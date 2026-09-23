cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)

# KVarN decode/reduction select GRF size through their SYCL launch properties.
# Preserve compiler-selected GRF sizes for helper kernels in the same library.
add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR
                       MIXED_GRF_SIZE)
