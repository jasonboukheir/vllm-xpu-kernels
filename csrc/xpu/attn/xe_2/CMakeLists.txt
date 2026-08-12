cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)

add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR)

# Keep the KVarN development loop independent of the several hundred generated
# generic attention instantiations above. This library is intentionally not
# installed or linked into the production extension; the benchmark registration
# shim below is loaded explicitly by local correctness/performance tools.
if(NOT VLLM_XPU_LIBS_ONLY)
add_library(
  kvarn_decode_benchmark_xe_2 SHARED kvarn_decode_xe2.cpp
                                       kvarn_hadamard_scatter_xe2.cpp
                                       ../../../../benchmark/kvarn_benchmark_api.cpp)
target_include_directories(
  kvarn_decode_benchmark_xe_2
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
         ${CMAKE_SOURCE_DIR}
  PRIVATE ${SYCL_TLA_INCLUDE_DIRS})
set_property(TARGET kvarn_decode_benchmark_xe_2 PROPERTY CXX_STANDARD 20)
target_compile_options(kvarn_decode_benchmark_xe_2
                       PRIVATE ${SYCL_TLA_KERNELS_COMPILE_FLAGS} -fPIC)
target_compile_definitions(kvarn_decode_benchmark_xe_2
                           PRIVATE -DVLLM_XPU_ENABLE_XE2)
target_link_libraries(kvarn_decode_benchmark_xe_2 PRIVATE torch
                                                          ${TORCH_LIBRARIES})
target_link_options(kvarn_decode_benchmark_xe_2 PRIVATE
                    ${SYCL_DEVICE_LINK_FLAGS})
if(XE2_AOT_DEVICES)
  target_link_options(
    kvarn_decode_benchmark_xe_2 PRIVATE
    "-device ${XE2_AOT_DEVICES} -internal_options -cl-intel-256-GRF-per-thread")
endif()
set_target_properties(
  kvarn_decode_benchmark_xe_2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                         "${CMAKE_BINARY_DIR}/" NO_SONAME TRUE)
# The packaged registration extension depends on this SONAME.  Advertising it
# from the narrow decoder lets local serving tests preload the current KVarN
# implementation ahead of the packaged full attention library (which remains
# second in LD_PRELOAD to provide every unrelated attention symbol).
target_link_options(kvarn_decode_benchmark_xe_2 PRIVATE
                    "-Wl,-soname,libattn_kernels_xe_2.so")
endif()
