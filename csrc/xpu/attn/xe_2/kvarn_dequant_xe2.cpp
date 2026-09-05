#include "kvarn_dequant_xe2.h"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace {

constexpr int kHeadDim = 256;
constexpr int kGroup = 128;
constexpr int kPackedBytesPerItem = 4;
constexpr int kPackedBytes = kHeadDim * kGroup / 2;
constexpr int kSColOffset = kPackedBytes;
constexpr int kZpOffset = kSColOffset + kHeadDim * 2;
constexpr int kSRowOffset = kZpOffset + kHeadDim * 2;
constexpr int vPackedOffset = kSRowOffset + kGroup * 2;
constexpr int vSColOffset = vPackedOffset + kPackedBytes;
constexpr int vSRowOffset = vSColOffset + kHeadDim * 2;
constexpr int vZpOffset = vSRowOffset + kGroup * 2;
constexpr int kTileBytes = vZpOffset + kGroup * 2;

inline void check_inputs(
    const at::Tensor& packed_cache,
    const at::Tensor& key_out,
    const at::Tensor& value_out) {
  TORCH_CHECK(packed_cache.is_xpu(), "packed_cache must be on XPU");
  TORCH_CHECK(key_out.is_xpu(), "key_out must be on XPU");
  TORCH_CHECK(value_out.is_xpu(), "value_out must be on XPU");
  TORCH_CHECK(
      packed_cache.scalar_type() == at::kByte,
      "packed_cache must have dtype uint8");
  TORCH_CHECK(
      key_out.scalar_type() == at::kHalf &&
          value_out.scalar_type() == at::kHalf,
      "key_out and value_out must have dtype float16");
  TORCH_CHECK(
      packed_cache.dim() == 3,
      "packed_cache must have shape [blocks, heads, tile_bytes]");
  TORCH_CHECK(
      packed_cache.size(2) >= kTileBytes,
      "packed_cache record is shorter than the K4V4/D256/G128 layout");
  TORCH_CHECK(
      key_out.sizes() ==
          at::IntArrayRef(
              {packed_cache.size(0), packed_cache.size(1), kHeadDim, kGroup}),
      "key_out must have shape [blocks, heads, 256, 128]");
  TORCH_CHECK(
      value_out.sizes() ==
          at::IntArrayRef(
              {packed_cache.size(0), packed_cache.size(1), kGroup, kHeadDim}),
      "value_out must have shape [blocks, heads, 128, 256]");
  TORCH_CHECK(
      packed_cache.is_contiguous() && key_out.is_contiguous() &&
          value_out.is_contiguous(),
      "all tensors must be contiguous");
}

}  // namespace

void kvarn_dequant_xe2(
    const at::Tensor& packed_cache,
    at::Tensor& key_out,
    at::Tensor& value_out,
    bool dpas_layout) {
  check_inputs(packed_cache, key_out, value_out);

  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const auto* cache = packed_cache.data_ptr<uint8_t>();
  auto* key = reinterpret_cast<sycl::half*>(key_out.data_ptr<at::Half>());
  auto* value = reinterpret_cast<sycl::half*>(value_out.data_ptr<at::Half>());
  const int64_t records = packed_cache.size(0) * packed_cache.size(1);
  const int64_t record_stride = packed_cache.stride(1);
  const int64_t elements_per_record = kGroup * kHeadDim;
  const int64_t packed_elements_per_record = elements_per_record / 2;
  const int64_t items_per_record =
      packed_elements_per_record / kPackedBytesPerItem;
  const int64_t total = records * items_per_record;
  // One work-item consumes an aligned uint32 from K and V, producing eight
  // adjacent FP16 values from each. K remains [D,G] and V remains [G,D],
  // matching the QK/PV operand orientations. Four packed bytes never cross a
  // K channel row (64 bytes) or V token row (128 bytes), so row metadata is
  // loaded once per work-item.
  queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> item) {
    const int64_t linear = item[0];
    const int64_t record = linear / items_per_record;
    const int packed_element =
        (linear % items_per_record) * kPackedBytesPerItem;
    const auto* base = cache + record * record_stride;
    const auto* half_base = reinterpret_cast<const sycl::half*>(base);

    const int k_channel = packed_element / (kGroup / 2);
    const int k_token = (packed_element % (kGroup / 2)) * 2;
    const uint32_t k_word =
        *reinterpret_cast<const uint32_t*>(base + packed_element);
    const float k_s_col =
        static_cast<float>(half_base[kSColOffset / 2 + k_channel]);
    const float k_zp = static_cast<float>(half_base[kZpOffset / 2 + k_channel]);
    const int64_t k_out = record * elements_per_record + packed_element * 2;
    for (int nibble = 0; nibble < kPackedBytesPerItem * 2; ++nibble) {
      uint32_t qk_bits;
      if (dpas_layout) {
        int const token = k_token + nibble;
        int const local_token = token % 64;
        int const local_dim = k_channel % 64;
        int const lane = 2 * ((local_token % 16) % 8) + local_dim % 2;
        int const byte =
            ((((token / 64 * 4 + k_channel / 64) * 4 + local_token / 16) * 16 +
              lane) *
                 32 +
             local_dim / 2);
        qk_bits = (base[byte] >> (4 * ((local_token % 16) / 8))) & 0xF;
      } else {
        qk_bits = (k_word >> (nibble * 4)) & uint32_t{0xF};
      }
      const float qk = static_cast<float>(qk_bits);
      const float k_s_row =
          static_cast<float>(half_base[kSRowOffset / 2 + k_token + nibble]);
      key[k_out + nibble] = sycl::half((qk * k_s_col + k_zp) * k_s_row);
    }

    const int v_token = packed_element / (kHeadDim / 2);
    const int v_channel = (packed_element % (kHeadDim / 2)) * 2;
    const uint32_t v_word = *reinterpret_cast<const uint32_t*>(
        base + vPackedOffset + packed_element);
    const float v_s_row =
        static_cast<float>(half_base[vSRowOffset / 2 + v_token]);
    const float v_zp = static_cast<float>(half_base[vZpOffset / 2 + v_token]);
    const int64_t v_out = record * elements_per_record + packed_element * 2;
    for (int nibble = 0; nibble < kPackedBytesPerItem * 2; ++nibble) {
      uint32_t qv_bits;
      if (dpas_layout) {
        int const dim = v_channel + nibble;
        int const local_token = v_token % 64;
        int const local_dim = dim % 32;
        int const lane = 2 * (local_dim % 8) + local_token % 2;
        int const byte =
            vPackedOffset +
            ((((v_token / 64 * 8 + dim / 32) * 4 + local_token / 16) * 16 +
              lane) *
                 16 +
             8 * (local_dim / 16) + (local_token % 16) / 2);
        qv_bits = (base[byte] >> (4 * ((local_dim % 16) / 8))) & 0xF;
      } else {
        qv_bits = (v_word >> (nibble * 4)) & uint32_t{0xF};
      }
      const float qv = static_cast<float>(qv_bits);
      const float v_s_col =
          static_cast<float>(half_base[vSColOffset / 2 + v_channel + nibble]);
      value[v_out + nibble] = sycl::half((qv * v_s_row + v_zp) * v_s_col);
    }
  });
}
