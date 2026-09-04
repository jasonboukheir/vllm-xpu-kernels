#include "kvarn_hadamard_scatter_xe2.h"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <type_traits>

namespace {

constexpr int kHeadDim = 256;
constexpr int kKvHeads = 4;
constexpr int kSubgroup = 16;
constexpr int kValuesPerLane = kHeadDim / kSubgroup;

template <typename input_t>
class KVarNHadamardScatterKernel {
 public:
  KVarNHadamardScatterKernel(
      const input_t* key,
      const input_t* value,
      const int64_t* slot_mapping,
      const int32_t* block_to_slot,
      sycl::half* tail_key,
      sycl::half* tail_value,
      int64_t tokens,
      int64_t lookup_size,
      int64_t pool_size,
      int64_t key_token_stride,
      int64_t key_head_stride,
      int64_t value_token_stride,
      int64_t value_head_stride,
      int64_t pool_block_stride,
      int64_t pool_token_stride,
      int64_t pool_head_stride)
      : key_(key),
        value_(value),
        slot_mapping_(slot_mapping),
        block_to_slot_(block_to_slot),
        tail_key_(tail_key),
        tail_value_(tail_value),
        tokens_(tokens),
        lookup_size_(lookup_size),
        pool_size_(pool_size),
        key_token_stride_(key_token_stride),
        key_head_stride_(key_head_stride),
        value_token_stride_(value_token_stride),
        value_head_stride_(value_head_stride),
        pool_block_stride_(pool_block_stride),
        pool_token_stride_(pool_token_stride),
        pool_head_stride_(pool_head_stride) {}

  [[sycl::reqd_sub_group_size(kSubgroup)]] void
  operator()(sycl::nd_item<1> item) const {
    const int64_t row = item.get_group(0);
    const int lane = item.get_local_id(0);
    const int kv = row & 1;
    const int head = (row >> 1) % kKvHeads;
    const int64_t token = row / (2 * kKvHeads);
    if (token >= tokens_) return;

    const int64_t logical_slot = slot_mapping_[token];
    if (logical_slot < 0) return;
    constexpr int group = 128;
    const int64_t block = logical_slot / group;
    if (block < 0 || block >= lookup_size_) return;
    const int32_t pool_slot = block_to_slot_[block];
    if (pool_slot < 0 || pool_slot >= pool_size_) return;
    const int64_t position = logical_slot % group;

    const input_t* src =
        (kv == 0 ? key_ + token * key_token_stride_ + head * key_head_stride_
                 : value_ + token * value_token_stride_ +
                       head * value_head_stride_);
    float x[kValuesPerLane];
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      x[j] = static_cast<float>(src[lane + j * kSubgroup]);
    }

    // Sylvester H256. The first four butterfly dimensions cross subgroup
    // lanes; the upper four pair values already resident in the same lane.
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

    sycl::half* dst_base =
        (kv == 0 ? tail_key_ : tail_value_) +
        static_cast<int64_t>(pool_slot) * pool_block_stride_ +
        position * pool_token_stride_ + head * pool_head_stride_;
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      dst_base[lane + j * kSubgroup] = sycl::half(x[j] * (1.0f / 16.0f));
    }
  }

 private:
  const input_t* key_;
  const input_t* value_;
  const int64_t* slot_mapping_;
  const int32_t* block_to_slot_;
  sycl::half* tail_key_;
  sycl::half* tail_value_;
  int64_t tokens_;
  int64_t lookup_size_;
  int64_t pool_size_;
  int64_t key_token_stride_;
  int64_t key_head_stride_;
  int64_t value_token_stride_;
  int64_t value_head_stride_;
  int64_t pool_block_stride_;
  int64_t pool_token_stride_;
  int64_t pool_head_stride_;
};

void check_inputs(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& slot_mapping,
    const at::Tensor& block_to_slot,
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    int64_t group) {
  TORCH_CHECK(
      key.is_xpu() && value.is_xpu() && slot_mapping.is_xpu() &&
          block_to_slot.is_xpu() && tail_key.is_xpu() && tail_value.is_xpu(),
      "all tensors must be on XPU");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
      "key and value must have dtype float16 or bfloat16");
  TORCH_CHECK(
      value.scalar_type() == key.scalar_type(), "K/V dtypes must match");
  TORCH_CHECK(
      slot_mapping.scalar_type() == at::kLong &&
          block_to_slot.scalar_type() == at::kInt,
      "slot_mapping must be int64 and block_to_slot must be int32");
  TORCH_CHECK(
      key.dim() == 3 && key.size(1) == kKvHeads && key.size(2) == kHeadDim &&
          value.sizes() == key.sizes(),
      "K/V must have shape [tokens, 4, 256]");
  TORCH_CHECK(
      slot_mapping.dim() == 1 && slot_mapping.size(0) == key.size(0),
      "slot_mapping must have shape [tokens]");
  TORCH_CHECK(group == 128, "only group=128 is supported");
  TORCH_CHECK(
      tail_key.dim() == 4 && tail_key.size(1) == group &&
          tail_key.size(2) == kKvHeads && tail_key.size(3) == kHeadDim &&
          tail_value.sizes() == tail_key.sizes(),
      "tail K/V must have shape [pool, 128, 4, 256]");
  TORCH_CHECK(
      tail_key.scalar_type() == at::kHalf &&
          tail_value.scalar_type() == at::kHalf,
      "tail K/V must have dtype float16");
  TORCH_CHECK(
      key.stride(2) == 1 && value.stride(2) == 1 && tail_key.stride(3) == 1 &&
          tail_value.strides() == tail_key.strides(),
      "head dimension must be contiguous and tail K/V strides must match");
}

}  // namespace

void kvarn_hadamard_scatter_xe2(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& slot_mapping,
    const at::Tensor& block_to_slot,
    at::Tensor& tail_key,
    at::Tensor& tail_value,
    int64_t group,
    bool dpas_layout) {
  // Tail records are unquantized and layout-independent. Keep the resolved
  // packed-cache layout in this writer ABI so every cache boundary is explicit
  // and future writer variants can branch without re-reading process state.
  (void)dpas_layout;
  check_inputs(
      key, value, slot_mapping, block_to_slot, tail_key, tail_value, group);
  if (key.size(0) == 0) return;
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t rows = key.size(0) * kKvHeads * 2;
  const auto launch = [&](auto* key_ptr, auto* value_ptr) {
    using input_t = std::remove_pointer_t<decltype(key_ptr)>;
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(rows * kSubgroup, kSubgroup),
          KVarNHadamardScatterKernel<input_t>(
              key_ptr,
              value_ptr,
              slot_mapping.data_ptr<int64_t>(),
              block_to_slot.data_ptr<int32_t>(),
              reinterpret_cast<sycl::half*>(tail_key.data_ptr<at::Half>()),
              reinterpret_cast<sycl::half*>(tail_value.data_ptr<at::Half>()),
              key.size(0),
              block_to_slot.numel(),
              tail_key.size(0),
              key.stride(0),
              key.stride(1),
              value.stride(0),
              value.stride(1),
              tail_key.stride(0),
              tail_key.stride(1),
              tail_key.stride(2)));
    });
  };
  if (key.scalar_type() == at::kHalf) {
    launch(
        reinterpret_cast<const sycl::half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const sycl::half*>(value.data_ptr<at::Half>()));
  } else {
    using bf16 = sycl::ext::oneapi::bfloat16;
    launch(
        reinterpret_cast<const bf16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(value.data_ptr<at::BFloat16>()));
  }
  const auto current_stream =
      c10::xpu::getCurrentXPUStream(key.device().index());
  const at::Tensor* tensors[] = {
      &key, &value, &slot_mapping, &block_to_slot, &tail_key, &tail_value};
  for (const at::Tensor* tensor : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), current_stream);
  }
}
