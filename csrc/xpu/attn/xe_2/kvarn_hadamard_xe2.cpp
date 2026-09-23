#include "kvarn_hadamard_xe2.h"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <type_traits>

namespace {

constexpr int kHeadDim = 256;
constexpr int kSubgroup = 16;
constexpr int kValuesPerLane = kHeadDim / kSubgroup;

template <typename input_t>
class KVarNHadamardKernel {
 public:
  KVarNHadamardKernel(
      const input_t* input,
      sycl::half* output,
      int64_t rows,
      int64_t input_row_stride,
      int64_t output_row_stride)
      : input_(input),
        output_(output),
        rows_(rows),
        input_row_stride_(input_row_stride),
        output_row_stride_(output_row_stride) {}

  [[sycl::reqd_sub_group_size(kSubgroup)]] void
  operator()(sycl::nd_item<1> item) const {
    const int64_t row = item.get_group(0);
    if (row >= rows_) return;
    const int lane = item.get_local_id(0);
    const input_t* src = input_ + row * input_row_stride_;
    float x[kValuesPerLane];
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      x[j] = static_cast<float>(src[lane + j * kSubgroup]);
    }

    // Sylvester H256. This is deliberately the same mapping as the validated
    // KVarN H256 scatter kernel: four dimensions cross subgroup lanes and the
    // upper four pair values already resident in each lane.
    const auto sg = item.get_sub_group();
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      const int peer_lane = lane ^ (1 << stage);
#pragma unroll
      for (int j = 0; j < kValuesPerLane; ++j) {
        const float peer = sycl::select_from_group(sg, x[j], peer_lane);
        x[j] = (lane & (1 << stage)) ? peer - x[j] : x[j] + peer;
      }
    }
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      const int span = 1 << stage;
#pragma unroll
      for (int base = 0; base < kValuesPerLane; base += 2 * span) {
#pragma unroll
        for (int offset = 0; offset < span; ++offset) {
          const float a = x[base + offset];
          const float b = x[base + offset + span];
          x[base + offset] = a + b;
          x[base + offset + span] = a - b;
        }
      }
    }

    sycl::half* dst = output_ + row * output_row_stride_;
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      dst[lane + j * kSubgroup] = sycl::half(x[j] * (1.0f / 16.0f));
    }
  }

 private:
  const input_t* input_;
  sycl::half* output_;
  int64_t rows_;
  int64_t input_row_stride_;
  int64_t output_row_stride_;
};

void check_inputs(const at::Tensor& input, const at::Tensor& output) {
  TORCH_CHECK(input.is_xpu() && output.is_xpu(), "input/output must be on XPU");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "input must have dtype float16 or bfloat16");
  TORCH_CHECK(
      output.scalar_type() == at::kHalf, "output must have dtype float16");
  TORCH_CHECK(
      input.dim() == 2 && input.size(1) == kHeadDim,
      "input must have shape [N, 256]");
  TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
  TORCH_CHECK(
      input.stride(1) == 1 && output.stride(1) == 1,
      "head dimension must be contiguous");
}

}  // namespace

void kvarn_hadamard_xe2(const at::Tensor& input, at::Tensor& output) {
  check_inputs(input, output);
  if (input.size(0) == 0) return;
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t rows = input.size(0);
  const auto launch = [&](auto* input_ptr) {
    using input_t = std::remove_pointer_t<decltype(input_ptr)>;
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(rows * kSubgroup, kSubgroup),
          KVarNHadamardKernel<input_t>(
              input_ptr,
              reinterpret_cast<sycl::half*>(output.data_ptr<at::Half>()),
              rows,
              input.stride(0),
              output.stride(0)));
    });
  };
  if (input.scalar_type() == at::kHalf) {
    launch(reinterpret_cast<const sycl::half*>(input.data_ptr<at::Half>()));
  } else {
    using bf16 = sycl::ext::oneapi::bfloat16;
    launch(reinterpret_cast<const bf16*>(input.data_ptr<at::BFloat16>()));
  }
}
