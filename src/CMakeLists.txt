set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -g -DLOG_USE_COLOR")
set(CMAKE_C_FLAGS_DEBUG_INIT "-Wall -g -DLOG_USE_COLOR")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# Create executable
add_executable(canonical_splice_analyzer canonical_splice_analyzer.c)

find_package(Sanitizers)

# Link external libraries
target_link_libraries(canonical_splice_analyzer ${HTSLIB_LIBRARIES})
target_link_libraries(canonical_splice_analyzer ${CPLICEAI_LIBRARIES})

# Add logging
add_library(Logging STATIC logging/log.c)
target_compile_definitions(Logging PRIVATE -DLOG_USE_COLOR)
target_include_directories(canonical_splice_analyzer PRIVATE Logging)
target_link_libraries(canonical_splice_analyzer Logging)

# Add BCFTools
add_library(Bcftools STATIC bcftools/regidx.c bcftools/gff.c)
target_link_libraries(Bcftools PRIVATE ${HTSLIB_LIBRARIES})
target_include_directories(canonical_splice_analyzer PRIVATE Bcftools)
target_link_libraries(canonical_splice_analyzer Bcftools)

