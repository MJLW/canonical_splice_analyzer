#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#include <htslib/hts.h>
#include <htslib/faidx.h>
#include <htslib/kstring.h>
#include <cpliceai/predict.h>
#include <cpliceai/utils.h>

#include "bcftools/gff.h"
#include "bcftools/regidx.h"

#include "logging/log.h"

#define TENSORFLOW_LOG_LEVEL_SILENT 1 // Only warnings and above
#define SPLICE_SITE_MALLOC_START_COUNT 1000000

#define SPLICEAI_CONTEXT_PADDING 5000

#define HEADER_STRING "chr\tpos\tgene\ttid\tstrand\ttype\tneither\tacceptor\tdonor\n"

typedef enum {
    ACCEPTOR = ACCEPTOR_POS,
    DONOR = DONOR_POS
} SpliceSiteType;

typedef struct {
    const char *chr;
    uint32_t rid; // Chromosome/region encoding, for quick comparisons
    uint64_t pos;
    SpliceSiteType type;

    const char *gene_name;
    uint64_t gene_beg;
    uint64_t gene_end;
    uint32_t tid;
    int strand;

    uint32_t predicted;
    float score[NUM_SCORES]; // Should only be used if predicted==TRUE
} SpliceSite;

typedef struct {
    uint32_t n, m;
    SpliceSite *a;
} SpliceSites;

SpliceSite init_splice_site(const char *chr, const uint32_t rid, const uint64_t pos, const SpliceSiteType type, const char *gene_name, const uint64_t beg, const uint64_t end, const uint32_t tid, const int strand) {
    return (SpliceSite) { chr, rid, pos, type, gene_name, beg, end, tid, strand, 0};
}

void splice_sites_init(const uint32_t m, SpliceSites *sites) {
    sites->n = 0;
    sites->m = m;
    sites->a = malloc(m * sizeof(SpliceSite));
}

void splice_sites_destroy(SpliceSites *sites) {
    free(sites->a);
    sites->a = NULL;
    sites->n = 0;
    sites->m = 0;
}

SpliceSites get_splice_sites_from_gff(const gff_t *gff) {
    SpliceSites sites;
    splice_sites_init(SPLICE_SITE_MALLOC_START_COUNT, &sites);

    regidx_t *transcripts = gff_get((gff_t *) gff, idx_tscript); // Removes const qualification for this call as it's not typed const, but it is a simple getter without consequences
    regitr_t *itr = regitr_init(transcripts);
    while (regitr_loop(itr)) {
        const gf_tscript_t *tr = regitr_payload(itr, gf_tscript_t *);

        // Only want protein coding and stranded transcripts
        if ((tr->type != GF_PROTEIN_CODING) | (tr->gene->type != GF_PROTEIN_CODING)) continue;
        if (tr->strand == STRAND_UNK) log_error("Transcript has an UNKNOWN strand. Skipping...", itr->seq);

        const char *chr = itr->seq;
        const int rid = tr->gene->iseq;

        // Only want things located on chromosomes
        if (strncmp(chr, "chr", 3) != 0) continue; // WARN: This is based on RUMC GFF files, they may not always be prefixed with chr
        if (sites.n + 2 >= sites.m) sites.a = realloc(sites.a, (sites.m *= 2) * sizeof(SpliceSite));

        // Extract stuff from pointers, no practical benefit, just looks a bit cleaner
        const char *gene_name = tr->gene->name;
        const uint64_t tr_beg = tr->gene->beg, tr_end = tr->gene->end;
        const uint32_t tid = tr->id;
        const int strand = tr->strand;

        for (int i = 0; i < tr->ncds; i++) {
            const gf_cds_t *cds = tr->cds[i];
            const uint64_t cds_beg = cds->beg;
            const uint64_t cds_end = cds->beg + cds->len - 1; // Offset by -1 to get closed end coordinate

            if (tr->strand == STRAND_FWD) {
                if (i != 0) sites.a[sites.n++] = init_splice_site(chr, rid, cds_beg, ACCEPTOR, gene_name, tr_beg, tr_end, tid, strand); // As long as its not the first exon
                if (i != tr->ncds-1) sites.a[sites.n++] = init_splice_site(chr, rid, cds_end, DONOR, gene_name, tr_beg, tr_end, tid, strand); // As long as its not the last exon
            } else if (tr->strand == STRAND_REV) {
                if (i != 0) sites.a[sites.n++] = init_splice_site(chr, rid, cds_beg, DONOR, gene_name, tr_beg, tr_end, tid, strand); // As long as its not the last exon (approaches from 3' -> 5')
                if (i != tr->ncds-1) sites.a[sites.n++] = init_splice_site(chr, rid, cds_end, ACCEPTOR, gene_name, tr_beg, tr_end, tid, strand); // As long as its not the first exon (approaches from 3' -> 5')
            }
        }
    }

    return sites;
}

gff_t *read_gff(const char *gff_path) {
    log_debug("Loading GFF from: %s", gff_path);
    gff_t *gff = gff_init(gff_path);
    gff_parse(gff);
    return gff;
}

void parse_output_line(SpliceSite site, kstring_t *s) {
    kputs(site.chr, s);
    kputc('\t', s);
    kputl(site.pos, s);
    kputc('\t', s);
    kputs(site.gene_name, s);
    kputc('\t', s);
    kputw(site.tid, s);
    kputc('\t', s);
    kputs(site.strand == STRAND_FWD ? "fwd" : "rev", s);
    kputc('\t', s);
    kputs(site.type == ACCEPTOR ? "acceptor" : "donor", s);
    kputc('\t', s);
    kputd(site.score[0], s);
    kputc('\t', s);
    kputd(site.score[1], s);
    kputc('\t', s);
    kputd(site.score[2], s);
}

int main(int argc, char *argv[]) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", TENSORFLOW_LOG_LEVEL_SILENT);

    if (argc != 5) {
        fprintf(stdout, "Run as: ./canonical_splice_analyzer <spliceai_model_dir> <human_fa> <gff> <out.tsv>");
    }

    const char *models_dir_path = argv[1];
    log_debug("Loading SpliceAI models from: %s", models_dir_path);
    Model *models = load_models(argv[1]);

    const char *fasta_path = argv[2];
    log_debug("Loading FASTA from: %s", fasta_path);
    faidx_t *fai = fai_load(fasta_path);

    gff_t *gff = read_gff(argv[3]);
    SpliceSites sites = get_splice_sites_from_gff(gff);
    log_debug("Found %i splice sites", sites.n);

    FILE *out = fopen(argv[4], "w");
    fprintf(out, HEADER_STRING);

    for (int i = 0; i < sites.n; i++) {
        SpliceSite site = sites.a[i];

        // Fetch sequence for gene
        int window_len;
        const uint64_t window_beg = site.pos - SPLICEAI_CONTEXT_PADDING;
        const uint64_t window_end = site.pos + SPLICEAI_CONTEXT_PADDING;
        char *window_seq = faidx_fetch_seq(fai, site.chr, window_beg, window_end, &window_len);
        window_seq[window_len] = '\0';

        // Replace sequence beyond gene boundaries with N's
        const Range padding_size = {
            ((int64_t) site.gene_beg - (int64_t) window_beg) > 0 ? (site.gene_beg - window_beg) : 0,
            ((int64_t) window_end - (int64_t) site.gene_end) > 0 ? (window_end - site.gene_end) : 0
        };
        const char *padded_seq = pad_sequence(window_seq, padding_size, window_len);

        // Encode sequence
        float *encoded_seq = NULL;
        const int enc_len = one_hot_encode(padded_seq, window_len, (float **) &encoded_seq);
        if (site.strand == STRAND_REV) reverse_encoding(encoded_seq, enc_len);

        // Predict using encoding
        float *prediction = NULL;
        int num_predictions = 0;
        predict(models, enc_len, 1, &encoded_seq, &num_predictions, &prediction);
        free(encoded_seq);

        if (num_predictions != 3) {
            log_error("Incorrect number of predictions made on site %s:%li, %i predictions instead of 3. This is a bug. Exiting program.", site.chr, site.pos, num_predictions);
            return 1;
        }

        site.score[0] = prediction[0];
        site.score[1] = prediction[1];
        site.score[2] = prediction[2];

        kstring_t s = {0};
        parse_output_line(site, &s);
        fprintf(out, "%s\n", s.s);

        free(window_seq);
    }

    fclose(out);
    fai_destroy(fai);
    splice_sites_destroy(&sites);
    gff_destroy(gff);

    return 0;
}

