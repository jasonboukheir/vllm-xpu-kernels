#include "kvarn_sinkhorn_writer_xe2.h"
#include "kvarn_sinkhorn_writer_zero_xe2.hpp"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

// Frozen xe2_dpas cache ABI for D256/G128/K4V4/Hkv4.
constexpr int kHeadDim = 256;
constexpr int kGroup = 128;
constexpr int kKvHeads = 4;
constexpr int kWorkgroup = 256;
constexpr int kSubgroup = 32;
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

constexpr float kClipStdMin = 1.0e-3f;
constexpr float kClipStdMax = 1.0e3f;
constexpr float kLogScaleMin = -0.3f;
constexpr float kLogScaleMax = 10.0f;
constexpr float kImbalanceFloor = 1.0e-8f;

// One SLM allocation contains all scalar Sinkhorn state.  The source tile is
// kept separately in its original 16-bit type, so no balanced fp32 tile is
// ever materialized in global memory.
constexpr int kLogCol = 0;
constexpr int kLogRow = kLogCol + kHeadDim;
constexpr int kLinearCol = kLogRow + kHeadDim;
constexpr int kLinearRow = kLinearCol + kHeadDim;
constexpr int kBestCol = kLinearRow + kHeadDim;
constexpr int kBestRow = kBestCol + kHeadDim;
constexpr int kBestImbalance = kBestRow + kHeadDim;
constexpr int kMomentSum = kBestImbalance + 1;
constexpr int kMomentSquareSum = kMomentSum + kHeadDim;
constexpr int kStateFloats = kMomentSquareSum + kHeadDim;
constexpr std::size_t kLocalMemoryBytes = kStateFloats * sizeof(float);
static_assert(
    kLocalMemoryBytes <= 64 * 1024,
    "KVarN fused writer must fit the Xe2 64 KiB SLM contract");

inline int round_to_even_q4(float value) {
  // Match torch.round: nearest integer, with exact halfway cases going to the
  // even code.  Clamp before inspecting the fractional part.
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

template <typename input_t>
class KVarNSinkhornWriterKernel {
 public:
  KVarNSinkhornWriterKernel(
      const input_t* tail_key,
      const input_t* tail_value,
      const int64_t* pool_slots,
      const int64_t* block_ids,
      std::uint8_t* packed_cache,
      int64_t scheduled_blocks,
      int64_t pool_size,
      int64_t cache_blocks,
      int64_t record_stride,
      int64_t record_bytes,
      int sinkhorn_iterations,
      sycl::local_accessor<float, 1> state)
      : tail_key_(tail_key),
        tail_value_(tail_value),
        pool_slots_(pool_slots),
        block_ids_(block_ids),
        packed_cache_(packed_cache),
        scheduled_blocks_(scheduled_blocks),
        pool_size_(pool_size),
        cache_blocks_(cache_blocks),
        record_stride_(record_stride),
        record_bytes_(record_bytes),
        sinkhorn_iterations_(sinkhorn_iterations),
        state_(state) {}

  [[sycl::reqd_sub_group_size(kSubgroup)]] void
  operator()(sycl::nd_item<1> item) const {
    const int64_t tile_index = item.get_group(0);
    const int thread = item.get_local_id(0);
    if (tile_index >= scheduled_blocks_ * kKvHeads) return;

    const int64_t schedule_index = tile_index / kKvHeads;
    const int head = static_cast<int>(tile_index % kKvHeads);
    const int64_t pool_slot = pool_slots_[schedule_index];
    const int64_t block = block_ids_[schedule_index];
    // Every work-item in this group observes the same ownership tuple, so a
    // group-wide early return cannot strand another lane at a barrier.
    if (pool_slot < 0 || pool_slot >= pool_size_ || block < 0 ||
        block >= cache_blocks_) {
      return;
    }

    std::uint8_t* record = packed_cache_ + block * kKvHeads * record_stride_ +
                           head * record_stride_;
    const input_t* key_page =
        tail_key_ + (pool_slot * kGroup * kKvHeads + head) * kHeadDim;
    const input_t* value_page =
        tail_value_ + (pool_slot * kGroup * kKvHeads + head) * kHeadDim;

    // Source pages stay in their compact 16-bit cacheable representation.
    // Only the O(D+G) Sinkhorn state occupies SLM; balanced fp32 pages are
    // never written to global memory.  Repeated scans therefore hit the same
    // 64 KiB source working set without violating the B70's 64 KiB SLM limit.
    balance<true, kHeadDim, kGroup>(item, key_page);
    write_key(record, thread, key_page);
    item.barrier(sycl::access::fence_space::local_space);
    balance<false, kGroup, kHeadDim>(item, value_page);
    write_value(record, thread, value_page);
  }

 private:
  template <bool Key>
  float source_value(const input_t* source, int row, int col) const {
    // Tail storage is [token, head, channel].  K's Sinkhorn matrix transposes
    // it to [channel, token]; V retains [token, channel].
    const int64_t offset =
        Key ? col * kKvHeads * kHeadDim + row : row * kKvHeads * kHeadDim + col;
    return static_cast<float>(source[offset]);
  }

  template <bool Key, int Rows, int Cols>
  float current_value(const input_t* source, int row, int col) const {
    return source_value<Key>(source, row, col) / state_[kLinearCol + col] /
           state_[kLinearRow + row];
  }

  template <bool Key, int Rows, int Cols>
  float best_value(const input_t* source, int row, int col) const {
    return source_value<Key>(source, row, col) / state_[kBestCol + col] /
           state_[kBestRow + row];
  }

  // Triton-XPU 3.7.2 lowers the R256/C128 K tile and R128/C256 V tile to
  // eight 32-lane subgroups.  A column reduction first forms an exact
  // 32-value binary tree per lane, then uses a SPIR-V ClusteredReduce across
  // the eight K row warps or four V row warps.  Keeping this structure is
  // important: a serial accumulator changes fp32 Sinkhorn factors enough to
  // alter q4 bytes after only two iterations.
  template <bool Square, bool Key, int Rows, int Cols>
  float column_tree(const input_t* source, int col, int row_chunk) const {
    constexpr int kCluster = Rows / kSubgroup;
    constexpr int kValuesPerLane = Rows / kCluster;
    static_assert(kValuesPerLane == kSubgroup);
    float values[kValuesPerLane];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const float value =
          current_value<Key, Rows, Cols>(source, row_chunk + i * kCluster, col);
      values[i] = Square ? value * value : value;
    }
#pragma unroll
    for (int stride = 1; stride < kValuesPerLane; stride *= 2) {
#pragma unroll
      for (int i = 0; i < kValuesPerLane; i += 2 * stride) {
        values[i] = values[i] + values[i + stride];
      }
    }
    return values[0];
  }

  template <bool Key, int Rows, int Cols>
  void
  compute_column_moments(sycl::nd_item<1> item, const input_t* source) const {
    constexpr int kCluster = Rows / kSubgroup;
    constexpr int kClustersPerSubgroup = kSubgroup / kCluster;
    constexpr int kColumnsPerRound =
        (kWorkgroup / kSubgroup) * kClustersPerSubgroup;
    static_assert(Rows == 128 || Rows == 256);
    static_assert(Cols % kColumnsPerRound == 0);

    const int thread = item.get_local_id(0);
    const int subgroup_id = thread / kSubgroup;
    const auto subgroup = item.get_sub_group();
    const auto cluster =
        sycl::ext::oneapi::experimental::chunked_partition<kCluster>(subgroup);
    const int cluster_id = cluster.get_group_linear_id();
    const int row_chunk = cluster.get_local_linear_id();

#pragma unroll
    for (int round = 0; round < Cols / kColumnsPerRound; ++round) {
      const int col = round * kColumnsPerRound +
                      subgroup_id * kClustersPerSubgroup + cluster_id;
      float sum = column_tree<false, Key, Rows, Cols>(source, col, row_chunk);
      float square_sum =
          column_tree<true, Key, Rows, Cols>(source, col, row_chunk);
      sum = sycl::reduce_over_group(cluster, sum, sycl::plus<float>());
      square_sum =
          sycl::reduce_over_group(cluster, square_sum, sycl::plus<float>());
      if (cluster.leader()) {
        state_[kMomentSum + col] = sum;
        state_[kMomentSquareSum + col] = square_sum;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  template <bool Key, int Rows, int Cols>
  void compute_row_moments(sycl::nd_item<1> item, const input_t* source) const {
    const int thread = item.get_local_id(0);
    const int lane = thread % kSubgroup;
    const int subgroup_id = thread / kSubgroup;
    const auto subgroup = item.get_sub_group();

    if constexpr (Cols == kGroup) {
      // K: four adjacent columns per lane, then one full-subgroup reduction.
#pragma unroll
      for (int round = 0; round < Rows / (kWorkgroup / kSubgroup); ++round) {
        const int row = round * (kWorkgroup / kSubgroup) + subgroup_id;
        const int col = lane * 4;
        const float x0 = current_value<Key, Rows, Cols>(source, row, col);
        const float x1 = current_value<Key, Rows, Cols>(source, row, col + 1);
        const float x2 = current_value<Key, Rows, Cols>(source, row, col + 2);
        const float x3 = current_value<Key, Rows, Cols>(source, row, col + 3);
        const float sum = sycl::reduce_over_group(
            subgroup, (x0 + x1) + (x2 + x3), sycl::plus<float>());
        const float square_sum = sycl::reduce_over_group(
            subgroup,
            (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3),
            sycl::plus<float>());
        if (lane == 0) {
          state_[kMomentSum + row] = sum;
          state_[kMomentSquareSum + row] = square_sum;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
    } else {
      static_assert(Cols == kHeadDim);
      // V: Triton's [4,2] warp layout reduces two 128-column halves in full
      // subgroups, transposes through SLM, then uses ClusteredReduce<2>.
#pragma unroll
      for (int round = 0; round < Rows / 4; ++round) {
        const int row = round * 4 + subgroup_id / 2;
        const int half = subgroup_id % 2;
        const int col = half * kGroup + lane * 4;
        const float x0 = current_value<Key, Rows, Cols>(source, row, col);
        const float x1 = current_value<Key, Rows, Cols>(source, row, col + 1);
        const float x2 = current_value<Key, Rows, Cols>(source, row, col + 2);
        const float x3 = current_value<Key, Rows, Cols>(source, row, col + 3);
        const float sum = sycl::reduce_over_group(
            subgroup, (x0 + x1) + (x2 + x3), sycl::plus<float>());
        const float square_sum = sycl::reduce_over_group(
            subgroup,
            (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3),
            sycl::plus<float>());
        if (lane == 0) {
          state_[kMomentSum + row * 2 + half] = sum;
          state_[kMomentSquareSum + row * 2 + half] = square_sum;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
      const int row = thread / 2;
      float sum = state_[kMomentSum + thread];
      float square_sum = state_[kMomentSquareSum + thread];
      // Every subgroup must capture its SLM inputs before another subgroup's
      // row leaders overwrite the front half of the scratch vectors.
      item.barrier(sycl::access::fence_space::local_space);
      const auto row_pair =
          sycl::ext::oneapi::experimental::chunked_partition<2>(subgroup);
      sum = sycl::reduce_over_group(row_pair, sum, sycl::plus<float>());
      square_sum =
          sycl::reduce_over_group(row_pair, square_sum, sycl::plus<float>());
      if (row_pair.leader()) {
        state_[kMomentSum + row] = sum;
        state_[kMomentSquareSum + row] = square_sum;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
  }

  template <int Extent>
  float moment_std(int index) const {
    const float mean = state_[kMomentSum + index] / float(Extent);
    float variance =
        state_[kMomentSquareSum + index] / float(Extent) - mean * mean;
    // Preserve Triton 3.7.2's two operations instead of pre-folding N/(N-1).
    variance *= float(Extent);
    variance /= float(Extent - 1);
    return sycl::sqrt(sycl::fmax(variance, 0.0f));
  }

  template <bool Key, int Rows, int Cols>
  float imbalance(sycl::nd_item<1> item, const input_t* source) const {
    const int thread = item.get_local_id(0);
    const auto group = item.get_group();
    compute_column_moments<Key, Rows, Cols>(item, source);
    const float col_std = thread < Cols ? moment_std<Rows>(thread) : 0.0f;
    const float col_min = sycl::reduce_over_group(
        group,
        thread < Cols ? col_std : std::numeric_limits<float>::infinity(),
        sycl::minimum<float>());
    const float col_max = sycl::reduce_over_group(
        group,
        thread < Cols ? col_std : -std::numeric_limits<float>::infinity(),
        sycl::maximum<float>());
    compute_row_moments<Key, Rows, Cols>(item, source);
    const float row_sigma = thread < Rows ? moment_std<Cols>(thread) : 0.0f;
    const float row_min = sycl::reduce_over_group(
        group,
        thread < Rows ? row_sigma : std::numeric_limits<float>::infinity(),
        sycl::minimum<float>());
    const float row_max = sycl::reduce_over_group(
        group,
        thread < Rows ? row_sigma : -std::numeric_limits<float>::infinity(),
        sycl::maximum<float>());
    return col_max / sycl::fmax(col_min, kImbalanceFloor) +
           row_max / sycl::fmax(row_min, kImbalanceFloor);
  }

  template <bool Key, int Rows, int Cols>
  void balance(sycl::nd_item<1> item, const input_t* source) const {
    const int thread = item.get_local_id(0);
    if (thread < Cols) {
      state_[kLogCol + thread] = 0.0f;
      state_[kLinearCol + thread] = 1.0f;
      state_[kBestCol + thread] = 1.0f;
    }
    if (thread < Rows) {
      state_[kLogRow + thread] = 0.0f;
      state_[kLinearRow + thread] = 1.0f;
      state_[kBestRow + thread] = 1.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float initial_imbalance = imbalance<Key, Rows, Cols>(item, source);
    if (thread == 0) state_[kBestImbalance] = initial_imbalance;
    item.barrier(sycl::access::fence_space::local_space);

    for (int iteration = 0; iteration < sinkhorn_iterations_; ++iteration) {
      compute_column_moments<Key, Rows, Cols>(item, source);
      if (thread < Cols) {
        const float sigma = sycl::fmin(
            sycl::fmax(moment_std<Rows>(thread), kClipStdMin), kClipStdMax);
        const float log_scale = sycl::fmin(
            sycl::fmax(
                state_[kLogCol + thread] + sycl::log(sigma), kLogScaleMin),
            kLogScaleMax);
        state_[kLogCol + thread] = log_scale;
        state_[kLinearCol + thread] =
            sycl::exp2(log_scale * 1.4426950216293335f);
      }
      item.barrier(sycl::access::fence_space::local_space);

      compute_row_moments<Key, Rows, Cols>(item, source);
      if (thread < Rows) {
        const float sigma = sycl::fmin(
            sycl::fmax(moment_std<Cols>(thread), kClipStdMin), kClipStdMax);
        const float log_scale = sycl::fmin(
            sycl::fmax(
                state_[kLogRow + thread] + sycl::log(sigma), kLogScaleMin),
            kLogScaleMax);
        state_[kLogRow + thread] = log_scale;
        state_[kLinearRow + thread] =
            sycl::exp2(log_scale * 1.4426950216293335f);
      }
      item.barrier(sycl::access::fence_space::local_space);

      const float candidate = imbalance<Key, Rows, Cols>(item, source);
      const bool better = candidate <= state_[kBestImbalance];
      if (better && thread < Cols) {
        state_[kBestCol + thread] = state_[kLinearCol + thread];
      }
      if (better && thread < Rows) {
        state_[kBestRow + thread] = state_[kLinearRow + thread];
      }
      if (better && thread == 0) state_[kBestImbalance] = candidate;
      item.barrier(sycl::access::fence_space::local_space);
    }
  }

  void
  write_key(std::uint8_t* record, int channel, const input_t* source) const {
    float lo = std::numeric_limits<float>::infinity();
    float hi = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < kGroup; ++token) {
      const float value =
          best_value<true, kHeadDim, kGroup>(source, channel, token);
      lo = sycl::fmin(lo, value);
      hi = sycl::fmax(hi, value);
    }
    const float scale = sycl::fmax((hi - lo) / float(kQMax), 1.0e-10f);

    const int a = channel / 64;
    const int b = (channel % 64) / 2;
    const int c = channel % 2;
    for (int pair = 0; pair < kGroup / 2; ++pair) {
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
      record[packed] = pack_q4_pair(
          best_value<true, kHeadDim, kGroup>(source, channel, token0),
          best_value<true, kHeadDim, kGroup>(source, channel, token1),
          lo,
          scale);
    }

    auto* k_s_col = reinterpret_cast<sycl::half*>(record + kKSColOffset);
    auto* k_zp = reinterpret_cast<sycl::half*>(record + kKZpOffset);
    k_s_col[channel] = sycl::half(state_[kBestRow + channel] * scale);
    k_zp[channel] = sycl::half(state_[kBestRow + channel] * lo);
    if (channel < kGroup) {
      auto* k_s_row = reinterpret_cast<sycl::half*>(record + kKSRowOffset);
      k_s_row[channel] = sycl::half(state_[kBestCol + channel]);
    }
    for (int64_t offset = kRecordBytes + channel; offset < record_bytes_;
         offset += kHeadDim) {
      record[offset] = 0;
    }
  }

  void
  write_value(std::uint8_t* record, int thread, const input_t* source) const {
    if (thread < kGroup) {
      const int token = thread;
      float lo = std::numeric_limits<float>::infinity();
      float hi = -std::numeric_limits<float>::infinity();
      for (int channel = 0; channel < kHeadDim; ++channel) {
        const float value =
            best_value<false, kGroup, kHeadDim>(source, token, channel);
        lo = sycl::fmin(lo, value);
        hi = sycl::fmax(hi, value);
      }
      const float scale = sycl::fmax((hi - lo) / float(kQMax), 1.0e-10f);
      const int a = token / 64;
      const int b = (token % 64) / 16;
      const int c = (token % 16) / 2;
      const int d = token % 2;
      for (int pair = 0; pair < kHeadDim / 2; ++pair) {
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
        record[kVPackedOffset + packed] = pack_q4_pair(
            best_value<false, kGroup, kHeadDim>(source, token, channel0),
            best_value<false, kGroup, kHeadDim>(source, token, channel1),
            lo,
            scale);
      }
      auto* v_s_row = reinterpret_cast<sycl::half*>(record + kVSRowOffset);
      auto* v_zp = reinterpret_cast<sycl::half*>(record + kVZpOffset);
      v_s_row[token] = sycl::half(state_[kBestRow + token] * scale);
      v_zp[token] = sycl::half(state_[kBestRow + token] * lo);
    }
    auto* v_s_col = reinterpret_cast<sycl::half*>(record + kVSColOffset);
    v_s_col[thread] = sycl::half(state_[kBestCol + thread]);
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
  int sinkhorn_iterations_;
  sycl::local_accessor<float, 1> state_;
};

void check_tail_pool(
    const at::Tensor& tensor, const at::Tensor& anchor, const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(
      tensor.device() == anchor.device(), name, " must share the cache device");
  TORCH_CHECK(
      tensor.scalar_type() == at::kHalf ||
          tensor.scalar_type() == at::kBFloat16,
      name,
      " must have dtype float16 or bfloat16");
  TORCH_CHECK(
      tensor.dim() == 4 && tensor.size(1) == kGroup &&
          tensor.size(2) == kKvHeads && tensor.size(3) == kHeadDim,
      name,
      " must have shape [pool, 128, 4, 256]");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void kvarn_sinkhorn_pack_kv_xe2(
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    const at::Tensor& pool_slots,
    const at::Tensor& block_ids,
    at::Tensor& packed_cache,
    int64_t sinkhorn_iterations,
    bool dpas_layout) {
  TORCH_CHECK(dpas_layout, "kvarn_sinkhorn_pack_kv requires xe2_dpas layout");
  TORCH_CHECK(
      packed_cache.is_xpu() && packed_cache.scalar_type() == at::kByte,
      "packed_cache must be uint8 on XPU");
  TORCH_CHECK(
      packed_cache.dim() == 3 && packed_cache.size(1) == kKvHeads &&
          packed_cache.size(2) >= kRecordBytes,
      "packed_cache must have shape [blocks, 4, record_bytes>=35072]");
  TORCH_CHECK(packed_cache.is_contiguous(), "packed_cache must be contiguous");
  TORCH_CHECK(
      packed_cache.size(2) % 4 == 0,
      "packed cache record stride must preserve the four-byte xe2_dpas ABI "
      "alignment");
  check_tail_pool(tail_key, packed_cache, "tail_key");
  check_tail_pool(tail_value, packed_cache, "tail_value");
  TORCH_CHECK(
      tail_value.sizes() == tail_key.sizes() &&
          tail_value.scalar_type() == tail_key.scalar_type(),
      "tail_key and tail_value must have identical shape and dtype");
  TORCH_CHECK(
      pool_slots.is_xpu() && pool_slots.device() == packed_cache.device() &&
          pool_slots.scalar_type() == at::kLong && pool_slots.dim() == 1 &&
          pool_slots.is_contiguous(),
      "pool_slots must be contiguous int64 on the packed-cache XPU");
  TORCH_CHECK(
      block_ids.is_xpu() && block_ids.device() == packed_cache.device() &&
          block_ids.scalar_type() == at::kLong && block_ids.dim() == 1 &&
          block_ids.is_contiguous(),
      "block_ids must be contiguous int64 on the packed-cache XPU");
  TORCH_CHECK(
      block_ids.sizes() == pool_slots.sizes(),
      "pool_slots and block_ids must have identical shape");
  TORCH_CHECK(
      sinkhorn_iterations >= 0 && sinkhorn_iterations <= 64,
      "sinkhorn_iterations must be between 0 and 64");
  if (block_ids.numel() == 0) return;

  const auto current_stream =
      c10::xpu::getCurrentXPUStream(packed_cache.device().index());
  auto& queue = current_stream.queue();
  const int64_t scheduled_blocks = block_ids.size(0);
  const int64_t tiles = scheduled_blocks * kKvHeads;
  const auto launch = [&](auto* key_ptr, auto* value_ptr) {
    using input_t =
        std::remove_const_t<std::remove_pointer_t<decltype(key_ptr)>>;
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> state(sycl::range<1>(kStateFloats), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(tiles * kWorkgroup, kWorkgroup),
          KVarNSinkhornWriterKernel<input_t>(
              key_ptr,
              value_ptr,
              pool_slots.data_ptr<int64_t>(),
              block_ids.data_ptr<int64_t>(),
              packed_cache.data_ptr<std::uint8_t>(),
              scheduled_blocks,
              tail_key.size(0),
              packed_cache.size(0),
              packed_cache.stride(1),
              packed_cache.size(2),
              static_cast<int>(sinkhorn_iterations),
              state));
    });
  };
  const auto launch_for_type = [&](auto* key_ptr, auto* value_ptr) {
    if (sinkhorn_iterations == 0) {
      vllm::kvarn::xe2::submit_kvarn_sinkhorn_zero_writer(
          queue,
          key_ptr,
          value_ptr,
          pool_slots.data_ptr<int64_t>(),
          block_ids.data_ptr<int64_t>(),
          packed_cache.data_ptr<std::uint8_t>(),
          scheduled_blocks,
          tail_key.size(0),
          packed_cache.size(0),
          packed_cache.stride(1),
          packed_cache.size(2));
    } else {
      launch(key_ptr, value_ptr);
    }
  };
  if (tail_key.scalar_type() == at::kHalf) {
    launch_for_type(
        reinterpret_cast<const sycl::half*>(tail_key.data_ptr<at::Half>()),
        reinterpret_cast<const sycl::half*>(tail_value.data_ptr<at::Half>()));
  } else {
    using bf16 = sycl::ext::oneapi::bfloat16;
    launch_for_type(
        reinterpret_cast<const bf16*>(tail_key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(tail_value.data_ptr<at::BFloat16>()));
  }

  const at::Tensor* tensors[] = {
      &tail_key, &tail_value, &pool_slots, &block_ids, &packed_cache};
  for (const at::Tensor* tensor : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), current_stream);
  }
}
