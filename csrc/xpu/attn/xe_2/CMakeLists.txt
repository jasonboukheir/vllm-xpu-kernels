cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)

# This target contains separately selectable 128- and 256-GRF KVarN kernels.
# The SYCL launch properties split them into distinct device images. Do not add
# the library-wide -cl-intel-256-GRF-per-thread backend option: it is forwarded
# unchanged to every AOT image and would defeat the 128-GRF experiment.
add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR MIXED_GRF_SIZE)
