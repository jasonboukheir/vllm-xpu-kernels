cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)

# These two writers define the byte-level KVarN cache record contract. Keep
# their floating-point expressions ordered as written instead of inheriting
# reciprocal and reassociation transformations that can move intermediate
# values across q4 rounding and Sinkhorn best-iteration boundaries. Contraction
# remains enabled, matching the captured Triton writer device IR. This is a
# source property so the rest of the Xe2 attention library retains its normal
# -O3 floating-point optimization policy.
set(KVARN_PRECISE_FP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kvarn_sinkhorn_writer_xe2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kvarn_balanced_writer_xe2.cpp")
set_source_files_properties(${KVARN_PRECISE_FP_SOURCES}
                            PROPERTIES COMPILE_OPTIONS "-fp-model=precise")

# This target contains separately selectable 128- and 256-GRF KVarN kernels.
# The SYCL launch properties split them into distinct device images. Do not add
# the library-wide -cl-intel-256-GRF-per-thread backend option: it is forwarded
# unchanged to every AOT image and would defeat the 128-GRF experiment.
add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR MIXED_GRF_SIZE)
