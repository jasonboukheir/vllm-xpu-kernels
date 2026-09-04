#include "kvarn_balanced_writer_xe2.h"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

namespace {

// Frozen xe2_dpas cache ABI for the Brutus Qwen3.5 D256/G128/K4V4 profile.
constexpr int kHeadDim = 256;
constexpr int kGroup = 128;
constexpr int kKvHeads = 4;
constexpr int kSubgroup = 16;
constexpr int kQMax = 15;
constexpr int kPackedBytes = kHeadDim * kGroup / 2;
constexpr int kKSColOffset = kPackedBytes;
constexpr int kKZpOffset = kKSColOffset + kHeadDim * 2;
constexpr int kKSRowOffset = kKZpOffset + kHeadDim * 2;
constexpr int kVPackedOffset = kKSRowOffset + kGroup * 2;
constexpr int kVSColOffset = kVPackedOffset + kPackedBytes;
constexpr int kVSRowOffset = kVSColOffset + kHeadDim * 2;
constexpr int kVZpOffset = kVSRowOffset + kGroup * 2;
constexpr int kRecordBytes = kVZpOffset + kGroup * 2;

inline int round_to_even_q4(float value) {
  // torch.round uses round-to-nearest with ties to even. Keep this explicit so
  // the packed writer is independent of the device's ambient rounding mode.
  const float clipped = sycl::fmin(sycl::fmax(value, 0.0f), float(kQMax));
  const int lower = static_cast<int>(sycl::floor(clipped));
  const float fraction = clipped - float(lower);
  if (fraction > 0.5f || (fraction == 0.5f && (lower & 1))) {
    return lower < kQMax ? lower + 1 : kQMax;
  }
  return lower;
}

inline std::uint8_t pack_q4_pair(float x0, float x1, float lo, float scale) {
  const int q0 = round_to_even_q4((x0 - lo) / scale);
  const int q1 = round_to_even_q4((x1 - lo) / scale);
  return static_cast<std::uint8_t>(q0 | (q1 << 4));
}

class KVarNBalancedWriterKernel {
 public:
  KVarNBalancedWriterKernel(
      const float* key_balanced,
      const float* key_sinkhorn_col,
      const float* key_sinkhorn_row,
      const float* value_balanced,
      const float* value_sinkhorn_col,
      const float* value_sinkhorn_row,
      const int64_t* block_ids,
      std::uint8_t* packed_cache,
      int64_t tiles,
      int64_t blocks,
      int64_t record_stride,
      int64_t record_bytes)
      : key_balanced_(key_balanced),
        key_sinkhorn_col_(key_sinkhorn_col),
        key_sinkhorn_row_(key_sinkhorn_row),
        value_balanced_(value_balanced),
        value_sinkhorn_col_(value_sinkhorn_col),
        value_sinkhorn_row_(value_sinkhorn_row),
        block_ids_(block_ids),
        packed_cache_(packed_cache),
        tiles_(tiles),
        blocks_(blocks),
        record_stride_(record_stride),
        record_bytes_(record_bytes) {}

  [[sycl::reqd_sub_group_size(kSubgroup)]] void
  operator()(sycl::nd_item<1> item) const {
    const int64_t work_row = item.get_group(0);
    const int lane = item.get_local_id(0);
    const int row_kind = static_cast<int>(work_row % (kHeadDim + kGroup));
    const int64_t tile = work_row / (kHeadDim + kGroup);
    if (tile >= tiles_) return;

    const int64_t block = block_ids_[tile / kKvHeads];
    if (block < 0 || block >= blocks_) return;
    const int head = static_cast<int>(tile % kKvHeads);
    std::uint8_t* record = packed_cache_ + block * kKvHeads * record_stride_ +
                           head * record_stride_;
    const auto sg = item.get_sub_group();

    if (row_kind < kHeadDim) {
      write_key_row(record, tile, row_kind, lane, sg);
    } else {
      write_value_row(record, tile, row_kind - kHeadDim, lane, sg);
    }
  }

 private:
  template <typename Subgroup>
  void write_key_row(
      std::uint8_t* record,
      int64_t tile,
      int channel,
      int lane,
      const Subgroup& sg) const {
    const float* row =
        key_balanced_ + tile * kHeadDim * kGroup + channel * kGroup;
    float lane_lo = std::numeric_limits<float>::infinity();
    float lane_hi = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int token = lane; token < kGroup; token += kSubgroup) {
      const float x = row[token];
      lane_lo = sycl::fmin(lane_lo, x);
      lane_hi = sycl::fmax(lane_hi, x);
    }
    const float lo =
        sycl::reduce_over_group(sg, lane_lo, sycl::minimum<float>());
    const float hi =
        sycl::reduce_over_group(sg, lane_hi, sycl::maximum<float>());
    const float scale = sycl::fmax((hi - lo) / float(kQMax), 1.0e-10f);

    // DPAS K byte order is the exact inverse of
    // q.reshape(4,32,2,2,4,2,8).permute(3,0,4,6,2,1,5), followed by
    // nibble packing of the last axis. For one logical channel, the two
    // nibbles are tokens separated by eight positions.
    const int a = channel / 64;
    const int b = (channel % 64) / 2;
    const int c = channel % 2;
#pragma unroll
    for (int pair = lane; pair < kGroup / 2; pair += kSubgroup) {
      const int d = pair / 32;
      const int e = (pair % 32) / 8;
      const int g = pair % 8;
      const int token0 = d * 64 + e * 16 + g;
      const int token1 = token0 + 8;
      int packed = d;
      packed = packed * 4 + a;
      packed = packed * 4 + e;
      packed = packed * 8 + g;
      packed = packed * 2 + c;
      packed = packed * 32 + b;
      record[packed] = pack_q4_pair(row[token0], row[token1], lo, scale);
    }

    if (lane == 0) {
      auto* k_s_col = reinterpret_cast<sycl::half*>(record + kKSColOffset);
      auto* k_zp = reinterpret_cast<sycl::half*>(record + kKZpOffset);
      auto* v_s_col = reinterpret_cast<sycl::half*>(record + kVSColOffset);
      const float sinkhorn_row = key_sinkhorn_row_[tile * kHeadDim + channel];
      k_s_col[channel] = sycl::half(sinkhorn_row * scale);
      k_zp[channel] = sycl::half(sinkhorn_row * lo);
      v_s_col[channel] =
          sycl::half(value_sinkhorn_col_[tile * kHeadDim + channel]);

      // The physical record can have a large trailing power-of-two pad. The
      // reference writer zeroes it on every selected record; distribute that
      // work across the 256 K rows to preserve byte identity without a second
      // launch. Each byte has exactly one writer.
      for (int64_t offset = kRecordBytes + channel; offset < record_bytes_;
           offset += kHeadDim) {
        record[offset] = 0;
      }
    }
  }

  template <typename Subgroup>
  void write_value_row(
      std::uint8_t* record,
      int64_t tile,
      int token,
      int lane,
      const Subgroup& sg) const {
    const float* row =
        value_balanced_ + tile * kGroup * kHeadDim + token * kHeadDim;
    float lane_lo = std::numeric_limits<float>::infinity();
    float lane_hi = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int channel = lane; channel < kHeadDim; channel += kSubgroup) {
      const float x = row[channel];
      lane_lo = sycl::fmin(lane_lo, x);
      lane_hi = sycl::fmax(lane_hi, x);
    }
    const float lo =
        sycl::reduce_over_group(sg, lane_lo, sycl::minimum<float>());
    const float hi =
        sycl::reduce_over_group(sg, lane_hi, sycl::maximum<float>());
    const float scale = sycl::fmax((hi - lo) / float(kQMax), 1.0e-10f);

    // DPAS V byte order is the exact inverse of
    // q.reshape(2,4,8,2,8,2,2,8).permute(0,4,1,7,3,5,2,6), followed by
    // nibble packing. The paired logical channels differ by eight.
    const int a = token / 64;
    const int b = (token % 64) / 16;
    const int c = (token % 16) / 2;
    const int d = token % 2;
#pragma unroll
    for (int pair = lane; pair < kHeadDim / 2; pair += kSubgroup) {
      const int e = pair / 16;
      const int f = (pair % 16) / 8;
      const int h = pair % 8;
      const int channel0 = e * 32 + f * 16 + h;
      const int channel1 = channel0 + 8;
      int packed = a;
      packed = packed * 8 + e;
      packed = packed * 4 + b;
      packed = packed * 8 + h;
      packed = packed * 2 + d;
      packed = packed * 2 + f;
      packed = packed * 8 + c;
      record[kVPackedOffset + packed] =
          pack_q4_pair(row[channel0], row[channel1], lo, scale);
    }

    if (lane == 0) {
      auto* k_s_row = reinterpret_cast<sycl::half*>(record + kKSRowOffset);
      auto* v_s_row = reinterpret_cast<sycl::half*>(record + kVSRowOffset);
      auto* v_zp = reinterpret_cast<sycl::half*>(record + kVZpOffset);
      k_s_row[token] = sycl::half(key_sinkhorn_col_[tile * kGroup + token]);
      const float sinkhorn_row = value_sinkhorn_row_[tile * kGroup + token];
      v_s_row[token] = sycl::half(sinkhorn_row * scale);
      v_zp[token] = sycl::half(sinkhorn_row * lo);
    }
  }

  const float* key_balanced_;
  const float* key_sinkhorn_col_;
  const float* key_sinkhorn_row_;
  const float* value_balanced_;
  const float* value_sinkhorn_col_;
  const float* value_sinkhorn_row_;
  const int64_t* block_ids_;
  std::uint8_t* packed_cache_;
  int64_t tiles_;
  int64_t blocks_;
  int64_t record_stride_;
  int64_t record_bytes_;
};

void check_float_tensor(
    const at::Tensor& tensor,
    const at::Tensor& anchor,
    at::IntArrayRef shape,
    const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(
      tensor.device() == anchor.device(), name, " must share the cache device");
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat, name, " must have dtype float32");
  TORCH_CHECK(tensor.sizes() == shape, name, " has an invalid shape");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void kvarn_pack_balanced_kv_xe2(
    const at::Tensor& key_balanced,
    const at::Tensor& key_sinkhorn_col,
    const at::Tensor& key_sinkhorn_row,
    const at::Tensor& value_balanced,
    const at::Tensor& value_sinkhorn_col,
    const at::Tensor& value_sinkhorn_row,
    const at::Tensor& block_ids,
    at::Tensor& packed_cache,
    bool dpas_layout) {
  TORCH_CHECK(dpas_layout, "kvarn_pack_balanced_kv requires xe2_dpas layout");
  TORCH_CHECK(
      packed_cache.is_xpu() && packed_cache.scalar_type() == at::kByte,
      "packed_cache must be uint8 on XPU");
  TORCH_CHECK(
      packed_cache.dim() == 3 && packed_cache.size(1) == kKvHeads &&
          packed_cache.size(2) >= kRecordBytes,
      "packed_cache must have shape [blocks, 4, record_bytes>=35072]");
  TORCH_CHECK(packed_cache.is_contiguous(), "packed_cache must be contiguous");
  TORCH_CHECK(
      packed_cache.size(2) % alignof(sycl::half) == 0,
      "packed cache record stride must preserve fp16 alignment");
  TORCH_CHECK(
      block_ids.is_xpu() && block_ids.device() == packed_cache.device() &&
          block_ids.scalar_type() == at::kLong && block_ids.dim() == 1 &&
          block_ids.is_contiguous(),
      "block_ids must be contiguous int64 on the packed-cache XPU");

  const int64_t batches = block_ids.size(0);
  const int64_t tiles = batches * kKvHeads;
  check_float_tensor(
      key_balanced, packed_cache, {tiles, kHeadDim, kGroup}, "key_balanced");
  check_float_tensor(
      key_sinkhorn_col, packed_cache, {tiles, kGroup}, "key_sinkhorn_col");
  check_float_tensor(
      key_sinkhorn_row, packed_cache, {tiles, kHeadDim}, "key_sinkhorn_row");
  check_float_tensor(
      value_balanced,
      packed_cache,
      {tiles, kGroup, kHeadDim},
      "value_balanced");
  check_float_tensor(
      value_sinkhorn_col,
      packed_cache,
      {tiles, kHeadDim},
      "value_sinkhorn_col");
  check_float_tensor(
      value_sinkhorn_row, packed_cache, {tiles, kGroup}, "value_sinkhorn_row");
  if (tiles == 0) return;

  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  const int64_t rows = tiles * (kHeadDim + kGroup);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(rows * kSubgroup, kSubgroup),
        KVarNBalancedWriterKernel(
            key_balanced.data_ptr<float>(),
            key_sinkhorn_col.data_ptr<float>(),
            key_sinkhorn_row.data_ptr<float>(),
            value_balanced.data_ptr<float>(),
            value_sinkhorn_col.data_ptr<float>(),
            value_sinkhorn_row.data_ptr<float>(),
            block_ids.data_ptr<int64_t>(),
            packed_cache.data_ptr<std::uint8_t>(),
            tiles,
            packed_cache.size(0),
            packed_cache.stride(1),
            packed_cache.size(2)));
  });

  const auto current_stream =
      c10::xpu::getCurrentXPUStream(packed_cache.device().index());
  const at::Tensor* tensors[] = {
      &key_balanced,
      &key_sinkhorn_col,
      &key_sinkhorn_row,
      &value_balanced,
      &value_sinkhorn_col,
      &value_sinkhorn_row,
      &block_ids,
      &packed_cache};
  for (const at::Tensor* tensor : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), current_stream);
  }
}
