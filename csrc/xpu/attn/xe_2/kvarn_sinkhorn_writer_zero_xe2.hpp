#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

namespace vllm::kvarn::xe2 {

// Compile-time iterations=0 specialization.  Its work decomposition and
// arithmetic intentionally mirror KVarNBalancedWriterKernel: one subgroup per
// K channel or V token, subgroup min/max, then the same nearest-even q4 pack.
// The only difference is loading the row directly from the raw 16-bit tail.
// Keeping this separate prevents iterative Sinkhorn arithmetic from changing
// the zero-iteration control's code generation.
constexpr int kZeroHeadDim = 256;
constexpr int kZeroGroup = 128;
constexpr int kZeroKvHeads = 4;
constexpr int kZeroSubgroup = 16;
constexpr int kZeroQMax = 15;
constexpr int kZeroPackedBytes = kZeroHeadDim * kZeroGroup / 2;
constexpr int kZeroKSColOffset = kZeroPackedBytes;
constexpr int kZeroKZpOffset = kZeroKSColOffset + kZeroHeadDim * 2;
constexpr int kZeroKSRowOffset = kZeroKZpOffset + kZeroHeadDim * 2;
constexpr int kZeroVPackedOffset = kZeroKSRowOffset + kZeroGroup * 2;
constexpr int kZeroVSColOffset = kZeroVPackedOffset + kZeroPackedBytes;
constexpr int kZeroVSRowOffset = kZeroVSColOffset + kZeroHeadDim * 2;
constexpr int kZeroVZpOffset = kZeroVSRowOffset + kZeroGroup * 2;
constexpr int kZeroRecordBytes = kZeroVZpOffset + kZeroGroup * 2;
static_assert(
    kZeroRecordBytes == 35072,
    "zero-iteration writer must preserve the frozen xe2_dpas record ABI");

inline int zero_round_to_even_q4(float value) {
  const float clipped = sycl::fmin(sycl::fmax(value, 0.0f), float(kZeroQMax));
  const int lower = static_cast<int>(sycl::floor(clipped));
  const float fraction = clipped - float(lower);
  if (fraction > 0.5f || (fraction == 0.5f && (lower & 1))) {
    return lower < kZeroQMax ? lower + 1 : kZeroQMax;
  }
  return lower;
}

inline std::uint8_t
zero_pack_q4_pair(float x0, float x1, float lo, float scale) {
  const int q0 = zero_round_to_even_q4((x0 - lo) / scale);
  const int q1 = zero_round_to_even_q4((x1 - lo) / scale);
  return static_cast<std::uint8_t>(q0 | (q1 << 4));
}

template <typename input_t>
class KVarNSinkhornZeroWriterKernel {
 public:
  KVarNSinkhornZeroWriterKernel(
      const input_t* tail_key,
      const input_t* tail_value,
      const int64_t* pool_slots,
      const int64_t* block_ids,
      std::uint8_t* packed_cache,
      int64_t scheduled_blocks,
      int64_t pool_size,
      int64_t cache_blocks,
      int64_t record_stride,
      int64_t record_bytes)
      : tail_key_(tail_key),
        tail_value_(tail_value),
        pool_slots_(pool_slots),
        block_ids_(block_ids),
        packed_cache_(packed_cache),
        scheduled_blocks_(scheduled_blocks),
        pool_size_(pool_size),
        cache_blocks_(cache_blocks),
        record_stride_(record_stride),
        record_bytes_(record_bytes) {}

  [[sycl::reqd_sub_group_size(kZeroSubgroup)]] void
  operator()(sycl::nd_item<1> item) const {
    const int64_t work_row = item.get_group(0);
    const int lane = item.get_local_id(0);
    const int row_kind =
        static_cast<int>(work_row % (kZeroHeadDim + kZeroGroup));
    const int64_t tile = work_row / (kZeroHeadDim + kZeroGroup);
    if (tile >= scheduled_blocks_ * kZeroKvHeads) return;

    const int64_t schedule_index = tile / kZeroKvHeads;
    const int head = static_cast<int>(tile % kZeroKvHeads);
    const int64_t pool_slot = pool_slots_[schedule_index];
    const int64_t block = block_ids_[schedule_index];
    if (pool_slot < 0 || pool_slot >= pool_size_ || block < 0 ||
        block >= cache_blocks_) {
      return;
    }

    std::uint8_t* record = packed_cache_ +
                           block * kZeroKvHeads * record_stride_ +
                           head * record_stride_;
    const input_t* key_page =
        tail_key_ +
        (pool_slot * kZeroGroup * kZeroKvHeads + head) * kZeroHeadDim;
    const input_t* value_page =
        tail_value_ +
        (pool_slot * kZeroGroup * kZeroKvHeads + head) * kZeroHeadDim;
    const auto subgroup = item.get_sub_group();
    if (row_kind < kZeroHeadDim) {
      write_key_row(record, key_page, row_kind, lane, subgroup);
    } else {
      write_value_row(
          record, value_page, row_kind - kZeroHeadDim, lane, subgroup);
    }
  }

 private:
  template <typename Subgroup>
  void write_key_row(
      std::uint8_t* record,
      const input_t* page,
      int channel,
      int lane,
      const Subgroup& subgroup) const {
    float lane_lo = std::numeric_limits<float>::infinity();
    float lane_hi = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int token = lane; token < kZeroGroup; token += kZeroSubgroup) {
      const float value = static_cast<float>(
          page[token * kZeroKvHeads * kZeroHeadDim + channel]);
      lane_lo = sycl::fmin(lane_lo, value);
      lane_hi = sycl::fmax(lane_hi, value);
    }
    const float lo =
        sycl::reduce_over_group(subgroup, lane_lo, sycl::minimum<float>());
    const float hi =
        sycl::reduce_over_group(subgroup, lane_hi, sycl::maximum<float>());
    const float scale = sycl::fmax((hi - lo) / float(kZeroQMax), 1.0e-10f);

    const int a = channel / 64;
    const int b = (channel % 64) / 2;
    const int c = channel % 2;
#pragma unroll
    for (int pair = lane; pair < kZeroGroup / 2; pair += kZeroSubgroup) {
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
      record[packed] = zero_pack_q4_pair(
          static_cast<float>(
              page[token0 * kZeroKvHeads * kZeroHeadDim + channel]),
          static_cast<float>(
              page[token1 * kZeroKvHeads * kZeroHeadDim + channel]),
          lo,
          scale);
    }

    if (lane == 0) {
      auto* key_scale =
          reinterpret_cast<sycl::half*>(record + kZeroKSColOffset);
      auto* key_zero = reinterpret_cast<sycl::half*>(record + kZeroKZpOffset);
      auto* value_col =
          reinterpret_cast<sycl::half*>(record + kZeroVSColOffset);
      key_scale[channel] = sycl::half(scale);
      key_zero[channel] = sycl::half(lo);
      value_col[channel] = sycl::half(1.0f);
      for (int64_t offset = kZeroRecordBytes + channel; offset < record_bytes_;
           offset += kZeroHeadDim) {
        record[offset] = 0;
      }
    }
  }

  template <typename Subgroup>
  void write_value_row(
      std::uint8_t* record,
      const input_t* page,
      int token,
      int lane,
      const Subgroup& subgroup) const {
    const input_t* row = page + token * kZeroKvHeads * kZeroHeadDim;
    float lane_lo = std::numeric_limits<float>::infinity();
    float lane_hi = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int channel = lane; channel < kZeroHeadDim; channel += kZeroSubgroup) {
      const float value = static_cast<float>(row[channel]);
      lane_lo = sycl::fmin(lane_lo, value);
      lane_hi = sycl::fmax(lane_hi, value);
    }
    const float lo =
        sycl::reduce_over_group(subgroup, lane_lo, sycl::minimum<float>());
    const float hi =
        sycl::reduce_over_group(subgroup, lane_hi, sycl::maximum<float>());
    const float scale = sycl::fmax((hi - lo) / float(kZeroQMax), 1.0e-10f);

    const int a = token / 64;
    const int b = (token % 64) / 16;
    const int c = (token % 16) / 2;
    const int d = token % 2;
#pragma unroll
    for (int pair = lane; pair < kZeroHeadDim / 2; pair += kZeroSubgroup) {
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
      record[kZeroVPackedOffset + packed] = zero_pack_q4_pair(
          static_cast<float>(row[channel0]),
          static_cast<float>(row[channel1]),
          lo,
          scale);
    }

    if (lane == 0) {
      auto* key_row = reinterpret_cast<sycl::half*>(record + kZeroKSRowOffset);
      auto* value_scale =
          reinterpret_cast<sycl::half*>(record + kZeroVSRowOffset);
      auto* value_zero = reinterpret_cast<sycl::half*>(record + kZeroVZpOffset);
      key_row[token] = sycl::half(1.0f);
      value_scale[token] = sycl::half(scale);
      value_zero[token] = sycl::half(lo);
    }
  }

  const input_t* tail_key_;
  const input_t* tail_value_;
  const int64_t* pool_slots_;
  const int64_t* block_ids_;
  std::uint8_t* packed_cache_;
  int64_t scheduled_blocks_;
  int64_t pool_size_;
  int64_t cache_blocks_;
  int64_t record_stride_;
  int64_t record_bytes_;
};

template <typename input_t>
void submit_kvarn_sinkhorn_zero_writer(
    sycl::queue& queue,
    const input_t* tail_key,
    const input_t* tail_value,
    const int64_t* pool_slots,
    const int64_t* block_ids,
    std::uint8_t* packed_cache,
    int64_t scheduled_blocks,
    int64_t pool_size,
    int64_t cache_blocks,
    int64_t record_stride,
    int64_t record_bytes) {
  const int64_t tiles = scheduled_blocks * kZeroKvHeads;
  const int64_t rows = tiles * (kZeroHeadDim + kZeroGroup);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(rows * kZeroSubgroup, kZeroSubgroup),
        KVarNSinkhornZeroWriterKernel<input_t>(
            tail_key,
            tail_value,
            pool_slots,
            block_ids,
            packed_cache,
            scheduled_blocks,
            pool_size,
            cache_blocks,
            record_stride,
            record_bytes));
  });
}

}  // namespace vllm::kvarn::xe2
