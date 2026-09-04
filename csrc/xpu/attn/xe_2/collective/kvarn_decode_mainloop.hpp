#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

#include "chunk_prefill_mainloop.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

/** Runtime description of one KVarN K4V4 cache record.
 *
 * Offsets are deliberately supplied by KVarNConfig.  In particular, the
 * historical 17,920-byte D=128 layout must never be baked into a D=256
 * kernel.  All offsets and strides below are bytes.
 */
struct KVarNK4V4Layout {
  std::uint8_t const* cache;
  std::int64_t block_stride;
  std::int64_t head_stride;
  int batch_size;
  int k_packed_offset;
  int k_s_col_offset;
  int k_zp_offset;
  int k_s_row_offset;
  int v_packed_offset;
  int v_s_col_offset;
  int v_s_row_offset;
  int v_zp_offset;
};

/** Optional full-precision pool for pages which must not be quantized.
 *
 * `block_to_slot[physical_block]` is -1 for a KVarN record and otherwise
 * indexes contiguous [slot, token, kv_head, dim] fp16 K/V storage.  Both
 * pools contain values in the same rotated frame as the packed cache.
 */
struct KVarNHybridTailLayout {
  int const* block_to_slot;
  cutlass::half_t const* key;
  cutlass::half_t const* value;
  std::int64_t slot_stride;
  std::int64_t token_stride;
  std::int64_t head_stride;
};

/** Register-fragment loader for the first native KVarN decode prototype.
 *
 * Logical cache geometry is fixed to D=256, G=128 and K4V4.  The enclosing
 * attention mainloop uses a K tile of 64, so each physical page is visited as
 * two consecutive tiles.  `CoordFragment` must be produced by partitioning a
 * CUTE identity tensor with the same MMA thread slice as `Fragment`; this is
 * what makes the register assignment independent of undocumented fragment
 * linear order.
 */
template <
    bool DpasPacked = false,
    bool VectorPackedLoads = false,
    bool SimdPackedUnpack = false>
struct KVarNK4V4FragmentLoader {
  static constexpr int kHeadDim = 256;
  static constexpr int kGroup = 128;
  static constexpr int kTileK = 64;
  static constexpr int kValuesPerWord = 8;
  static constexpr int kKRowBytes = kGroup / 2;
  static constexpr int kVRowBytes = kHeadDim / 2;
  static constexpr int kKPackedLaneWords = 8;
  static constexpr int kVPackedLaneWords = 4;
  static constexpr int kKPackedLaneBytes =
      kKPackedLaneWords * sizeof(std::uint32_t);
  static constexpr int kVPackedLaneBytes =
      kVPackedLaneWords * sizeof(std::uint32_t);
  static constexpr int kPackedBytes = kHeadDim * kGroup / 2;
  static constexpr int kColumnBytes = kHeadDim * sizeof(cutlass::half_t);
  static constexpr int kRowBytes = kGroup * sizeof(cutlass::half_t);
  static constexpr int kHalfRowBytes = kRowBytes / 2;
  static constexpr int kKMetadataBytes = 2 * kColumnBytes + kRowBytes;
  static constexpr int kVMetadataBytes = kColumnBytes + 2 * kRowBytes;
  static constexpr int kPackedHalfBytes = kPackedBytes / 2;
  static constexpr int kActiveRecordBytes =
      2 * kPackedBytes + kKMetadataBytes + kVMetadataBytes;
  static constexpr int kPagePrefetchThreads = 4 * cute::intel::sg_size;

  static_assert(!VectorPackedLoads || DpasPacked);
  static_assert(!SimdPackedUnpack || (DpasPacked && VectorPackedLoads));
  static_assert(kKPackedLaneBytes == 32);
  static_assert(kVPackedLaneBytes == 16);
  static_assert(
      sizeof(sycl::vec<std::uint32_t, kKPackedLaneWords>) == kKPackedLaneBytes);
  static_assert(
      sizeof(sycl::vec<std::uint32_t, kVPackedLaneWords>) == kVPackedLaneBytes);
  static_assert(kPackedBytes == 16384);
  static_assert(kActiveRecordBytes == 35072);
  static_assert(kPackedBytes % kPagePrefetchThreads == 0);
  static_assert(kKMetadataBytes % kPagePrefetchThreads == 0);
  static_assert(kVMetadataBytes % kPagePrefetchThreads == 0);

  KVarNK4V4Layout layout;
  KVarNHybridTailLayout tail;
  int const* page_table;
  int max_pages_per_seq;

  CUTLASS_DEVICE std::uint8_t const*
  record(int batch, int kv_head, int logical_tile) const {
    int const page = logical_tile / 2;
    int const block = page_table[batch * max_pages_per_seq + page];
    return layout.cache + std::int64_t(block) * layout.block_stride +
           std::int64_t(kv_head) * layout.head_stride;
  }

  CUTLASS_DEVICE int physical_block(int batch, int logical_tile) const {
    int const page = logical_tile / 2;
    return page_table[batch * max_pages_per_seq + page];
  }

  CUTLASS_DEVICE int tail_slot(int batch, int logical_tile) const {
    return tail.block_to_slot[physical_block(batch, logical_tile)];
  }

  CUTLASS_DEVICE float load_tail(
      cutlass::half_t const* pool,
      int slot,
      int token,
      int kv_head,
      int dim) const {
    auto offset = std::int64_t(slot) * tail.slot_stride +
                  std::int64_t(token) * tail.token_stride +
                  std::int64_t(kv_head) * tail.head_stride + dim;
    return static_cast<float>(pool[offset]);
  }

  CUTLASS_DEVICE static float load_f16(std::uint8_t const* ptr) {
    // KVarNConfig guarantees every fp16 field is two-byte aligned.
    auto bits = *reinterpret_cast<std::uint16_t const*>(ptr);
    return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
  }

  CUTLASS_DEVICE static std::uint32_t load_u32(std::uint8_t const* ptr) {
    return *reinterpret_cast<std::uint32_t const*>(ptr);
  }

  CUTLASS_DEVICE static int unpack_nibble(std::uint32_t word, int index) {
    return int((word >> (4 * index)) & 0xfu);
  }

  template <int RangeBytes>
  CUTLASS_DEVICE static void
  prefetch_lane_partition_l2(std::uint8_t const* range, int thread) {
    namespace syclex = sycl::ext::oneapi::experimental;
    static_assert(RangeBytes % kPagePrefetchThreads == 0);
    constexpr int kBytesPerThread = RangeBytes / kPagePrefetchThreads;
    syclex::prefetch(
        range + thread * kBytesPerThread,
        kBytesPerThread,
        syclex::properties{syclex::prefetch_hint_L2});
  }

  template <int RangeBytes>
  CUTLASS_DEVICE static void
  prefetch_lane_partition_l1(std::uint8_t const* range, int thread) {
    namespace syclex = sycl::ext::oneapi::experimental;
    static_assert(RangeBytes % kPagePrefetchThreads == 0);
    constexpr int kBytesPerThread = RangeBytes / kPagePrefetchThreads;
    syclex::prefetch(
        range + thread * kBytesPerThread,
        kBytesPerThread,
        syclex::properties{syclex::prefetch_hint_L1});
  }

  /** Prefetch exactly the packed-cache ranges used by the next physical page.
   *
   * All 64 work-items own disjoint byte ranges.  The partial-page form omits
   * the second K/V half and the row metadata belonging exclusively to it.
   * Column metadata remains necessary because the first half opens and closes
   * its own V scale frame at a split or sequence boundary.
   */
  template <bool BothHalves>
  CUTLASS_DEVICE void
  prefetch_dpas_page_l2(std::uint8_t const* rec, int thread) const {
    static_assert(DpasPacked);
    constexpr int kPackedRangeBytes =
        BothHalves ? kPackedBytes : kPackedBytes / 2;
    prefetch_lane_partition_l2<kPackedRangeBytes>(
        rec + layout.k_packed_offset, thread);

    if constexpr (BothHalves) {
      prefetch_lane_partition_l2<kKMetadataBytes>(
          rec + layout.k_s_col_offset, thread);
    } else {
      // K column scale + zero point + the first 64 row scales are contiguous.
      constexpr int kFirstHalfKMetadataBytes = 2 * kColumnBytes + kHalfRowBytes;
      prefetch_lane_partition_l2<kFirstHalfKMetadataBytes>(
          rec + layout.k_s_col_offset, thread);
    }

    prefetch_lane_partition_l2<kPackedRangeBytes>(
        rec + layout.v_packed_offset, thread);
    if constexpr (BothHalves) {
      prefetch_lane_partition_l2<kVMetadataBytes>(
          rec + layout.v_s_col_offset, thread);
    } else {
      // V column scale and first-half row scales are contiguous; first-half
      // zero points form a second exact range after the unused row-scale half.
      constexpr int kFirstHalfVScaleBytes = kColumnBytes + kHalfRowBytes;
      prefetch_lane_partition_l2<kFirstHalfVScaleBytes>(
          rec + layout.v_s_col_offset, thread);
      prefetch_lane_partition_l2<kHalfRowBytes>(
          rec + layout.v_zp_offset, thread);
    }
  }

  /** Stage only the V ranges consumed by one current K64 half into L1.
   *
   * Unlike next-page L2 prefetch, this is issued after QK has finished and
   * immediately before online softmax.  Softmax and probability rescaling
   * provide the overlap window while keeping unrelated K data out of L1.
   * Column scales are page-wide; packed V, row scales, and zero points are
   * half-local contiguous ranges in the immutable xe2_dpas cache ABI.
   */
  CUTLASS_DEVICE void prefetch_dpas_v_half_l1(
      std::uint8_t const* rec, int half, int thread) const {
    static_assert(DpasPacked);
    prefetch_lane_partition_l1<kPackedHalfBytes>(
        rec + layout.v_packed_offset + half * kPackedHalfBytes, thread);
    prefetch_lane_partition_l1<kColumnBytes>(
        rec + layout.v_s_col_offset, thread);
    prefetch_lane_partition_l1<kHalfRowBytes>(
        rec + layout.v_s_row_offset + half * kHalfRowBytes, thread);
    prefetch_lane_partition_l1<kHalfRowBytes>(
        rec + layout.v_zp_offset + half * kHalfRowBytes, thread);
  }

  template <int WordCount, class Fragment>
  CUTLASS_DEVICE static void
  fill_packed_lane_fragment(Fragment& dst, std::uint8_t const* lane_bytes) {
    static_assert(
        WordCount == kKPackedLaneWords || WordCount == kVPackedLaneWords);
    if constexpr (SimdPackedUnpack) {
      // The xe2_dpas lane ABI stores eight uint4 values in each little-endian
      // word.  View it as bytes so one SIMD operation extracts all low nibbles
      // and another extracts all high nibbles.  The two vector conversions are
      // exact for [0, 15] and avoid the old eight scalar shift/mask/converts
      // per packed word.  Interleaving low/high restores the immutable nibble
      // order consumed by the FP16 MMA-B fragment.
      constexpr int kBytesPerChunk = 16;
      constexpr int kLaneBytes = WordCount * sizeof(std::uint32_t);
      static_assert(kLaneBytes % kBytesPerChunk == 0);
      using ByteVector = sycl::vec<std::uint8_t, kBytesPerChunk>;
      using HalfVector = sycl::vec<sycl::half, kBytesPerChunk>;
      CUTLASS_PRAGMA_UNROLL
      for (int chunk = 0; chunk < kLaneBytes / kBytesPerChunk; ++chunk) {
        ByteVector const bytes = *reinterpret_cast<ByteVector const*>(
            lane_bytes + chunk * kBytesPerChunk);
        ByteVector const low_bits = bytes & ByteVector{0x0f};
        ByteVector const high_bits = bytes >> ByteVector{4};
        HalfVector const low = low_bits.template convert<sycl::half>();
        HalfVector const high = high_bits.template convert<sycl::half>();
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < kBytesPerChunk; ++byte) {
          int const output = chunk * (2 * kBytesPerChunk) + 2 * byte;
          dst(output) = static_cast<typename Fragment::value_type>(low[byte]);
          dst(output + 1) =
              static_cast<typename Fragment::value_type>(high[byte]);
        }
      }
    } else if constexpr (VectorPackedLoads) {
      // The host specialization validates the complete address induction:
      // cache base, block/head strides, field offset, and these fixed lane
      // extents.  Keep this as one explicit vector load per lane so this
      // candidate differs from the scalar DPAS baseline only at the load.
      using WordVector = sycl::vec<std::uint32_t, WordCount>;
      WordVector const words = *reinterpret_cast<WordVector const*>(lane_bytes);
      CUTLASS_PRAGMA_UNROLL
      for (int word_index = 0; word_index < WordCount; ++word_index) {
        std::uint32_t const word = words[word_index];
        CUTLASS_PRAGMA_UNROLL
        for (int nibble = 0; nibble < kValuesPerWord; ++nibble) {
          dst(word_index * kValuesPerWord + nibble) =
              static_cast<typename Fragment::value_type>(
                  unpack_nibble(word, nibble));
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int word_index = 0; word_index < WordCount; ++word_index) {
        std::uint32_t const word =
            load_u32(lane_bytes + word_index * sizeof(std::uint32_t));
        CUTLASS_PRAGMA_UNROLL
        for (int nibble = 0; nibble < kValuesPerWord; ++nibble) {
          dst(word_index * kValuesPerWord + nibble) =
              static_cast<typename Fragment::value_type>(
                  unpack_nibble(word, nibble));
        }
      }
    }
  }

  CUTLASS_DEVICE float
  load_k_quantized(std::uint8_t const* rec, int token, int dim) const {
    auto word = load_u32(
        rec + layout.k_packed_offset + dim * kKRowBytes + (token / 8) * 4);
    return float(unpack_nibble(word, token & 7));
  }

  CUTLASS_DEVICE float
  load_v_quantized(std::uint8_t const* rec, int token, int dim) const {
    auto word = load_u32(
        rec + layout.v_packed_offset + token * kVRowBytes + (dim / 8) * 4);
    return float(unpack_nibble(word, dim & 7));
  }

  CUTLASS_DEVICE float
  load_k_dpas_quantized(std::uint8_t const* rec, int token, int dim) const {
    int const half = token / 64;
    int const token16 = token % 16;
    int const subgroup = (token % 64) / 16;
    int const dim_tile = dim / 64;
    int const dim64 = dim % 64;
    int const lane = 2 * (token16 % 8) + dim64 % 2;
    int const slot = 2 * (dim64 / 2) + token16 / 8;
    auto const* lane_bytes =
        rec + layout.k_packed_offset +
        (((half * 4 + dim_tile) * 4 + subgroup) * 16 + lane) * 32;
    auto const word = load_u32(lane_bytes + (slot / 8) * 4);
    return float(unpack_nibble(word, slot % 8));
  }

  CUTLASS_DEVICE float
  load_v_dpas_quantized(std::uint8_t const* rec, int token, int dim) const {
    int const half = token / 64;
    int const token16 = token % 16;
    int const subgroup = (token % 64) / 16;
    int const value_tile = dim / 32;
    int const dim32 = dim % 32;
    int const lane = 2 * (dim32 % 8) + token16 % 2;
    int const inner = 2 * (token16 / 2) + (dim32 % 16) / 8;
    int const slot = 16 * (dim32 / 16) + inner;
    auto const* lane_bytes =
        rec + layout.v_packed_offset +
        (((half * 8 + value_tile) * 4 + subgroup) * 16 + lane) * 16;
    auto const word = load_u32(lane_bytes + (slot / 8) * 4);
    return float(unpack_nibble(word, slot % 8));
  }

  template <class Fragment>
  CUTLASS_DEVICE void fill_k_fragment(
      Fragment& dst,
      std::uint8_t const* rec,
      int slot,
      int kv_head,
      int logical_tile,
      int token_sg,
      int dim_tile) const {
    int token_base = (logical_tile & 1) * kTileK;
    int lane =
        sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];
    if constexpr (DpasPacked) {
      if (slot < 0) {
        int const subgroup = token_sg / 16;
        int const half = logical_tile & 1;
        auto const* lane_bytes =
            rec + layout.k_packed_offset +
            (((half * 4 + dim_tile / 64) * 4 + subgroup) * 16 + lane) * 32;
        fill_packed_lane_fragment<kKPackedLaneWords>(dst, lane_bytes);
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < dst.size(); ++i) {
          auto coord = dst.tv_layout()(lane, i);
          int token = token_base + token_sg + int(get<0>(coord));
          int dim = dim_tile + int(get<1>(coord));
          dst(i) = static_cast<typename Fragment::value_type>(
              load_tail(tail.key, slot, token, kv_head, dim));
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < dst.size(); ++i) {
        auto coord = dst.tv_layout()(lane, i);
        int token = token_base + token_sg + int(get<0>(coord));
        int dim = dim_tile + int(get<1>(coord));
        float value = slot < 0 ? load_k_quantized(rec, token, dim)
                               : load_tail(tail.key, slot, token, kv_head, dim);
        dst(i) = static_cast<typename Fragment::value_type>(value);
      }
    }
  }

  template <class Fragment>
  CUTLASS_DEVICE void fill_k_fragment_by_coordinate(
      Fragment& dst,
      std::uint8_t const* rec,
      int slot,
      int kv_head,
      int logical_tile,
      int token_sg,
      int dim_tile) const {
    static_assert(DpasPacked);
    int const token_base = (logical_tile & 1) * kTileK;
    int const lane =
        sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < dst.size(); ++i) {
      // The xe2_dpas cache ABI is frozen in the FP16 MMA-B fragment order.
      // Integer DPAS has a different MMA-B coordinate layout, so resolve each
      // destination coordinate through the frozen cache mapping instead of
      // treating its fragment slots as byte-order compatible.
      auto coord = dst.tv_layout()(lane, i);
      int const token = token_base + token_sg + int(get<0>(coord));
      int const dim = dim_tile + int(get<1>(coord));
      float const value = slot < 0
                              ? load_k_dpas_quantized(rec, token, dim)
                              : load_tail(tail.key, slot, token, kv_head, dim);
      dst(i) = static_cast<typename Fragment::value_type>(value);
    }
  }

  template <class Fragment>
  CUTLASS_DEVICE void fill_v_fragment(
      Fragment& dst,
      std::uint8_t const* rec,
      int slot,
      int kv_head,
      int logical_tile,
      int token_sg,
      int value_tile) const {
    int token_base = (logical_tile & 1) * kTileK;
    int lane =
        sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];
    if constexpr (DpasPacked) {
      if (slot < 0) {
        int const subgroup = token_sg / 16;
        int const half = logical_tile & 1;
        auto const* lane_bytes =
            rec + layout.v_packed_offset +
            (((half * 8 + value_tile / 32) * 4 + subgroup) * 16 + lane) * 16;
        fill_packed_lane_fragment<kVPackedLaneWords>(dst, lane_bytes);
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < dst.size(); ++i) {
          auto coord = dst.tv_layout()(lane, i);
          int token = token_base + token_sg + int(get<1>(coord));
          int dim = value_tile + int(get<0>(coord));
          dst(i) = static_cast<typename Fragment::value_type>(
              load_tail(tail.value, slot, token, kv_head, dim));
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < dst.size(); ++i) {
        auto coord = dst.tv_layout()(lane, i);
        int token = token_base + token_sg + int(get<1>(coord));
        int dim = value_tile + int(get<0>(coord));
        float value = slot < 0
                          ? load_v_quantized(rec, token, dim)
                          : load_tail(tail.value, slot, token, kv_head, dim);
        dst(i) = static_cast<typename Fragment::value_type>(value);
      }
    }
  }
};

/** D256/G128/K4V4 specialization of the Xe decode collective.
 *
 * The base collective is inherited only for its public fragment contract and
 * tested online-softmax implementation.  This operator never dereferences its
 * K_2D/V_2D arguments: K and V MMA-B fragments are reconstructed directly
 * from the packed KVarN record.
 */
template <
    class TiledMMAQK_,
    class TiledMMAQKInt_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    bool DpasPacked_ = false,
    bool VectorPackedLoads_ = false,
    bool QKInt8U4_ = false,
    bool ExactLiveRows_ = false,
    bool PagePair_ = false,
    bool NextPagePrefetch_ = false,
    bool SimdPackedUnpack_ = false,
    bool CurrentHalfVPrefetch_ = false,
    bool ReusePageRecordCursor_ = false>
struct KVarNDecodeFwdMainloop : DecodeFwdMainloop<
                                    XeDefault<1>,
                                    true,
                                    false,
                                    TiledMMAQK_,
                                    TiledMMAPV_,
                                    VTiles_,
                                    TensorQ_,
                                    TensorK_,
                                    TensorV_,
                                    void,
                                    void,
                                    void,
                                    false> {
  using Base = DecodeFwdMainloop<
      XeDefault<1>,
      true,
      false,
      TiledMMAQK_,
      TiledMMAPV_,
      VTiles_,
      TensorQ_,
      TensorK_,
      TensorV_,
      void,
      void,
      void,
      false>;

  using typename Base::ElementA;
  using typename Base::ElementS;
  using typename Base::FragA;
  using typename Base::FragARow;
  using typename Base::FragS;
  using typename Base::FragSCol;
  using typename Base::SGPerWG;
  using typename Base::SingleFragA;
  using typename Base::SubgroupLayoutQK;
  using typename Base::TensorK;
  using typename Base::TensorK2D;
  using typename Base::TensorQ;
  using typename Base::TensorQ2D;
  using typename Base::TensorV;
  using typename Base::TensorV2D;
  using typename Base::TiledMMAPV;
  using typename Base::TiledMMAQK;
  using typename Base::TileShapePV;
  using typename Base::TileShapeQK;
  using TiledMMAQKInt = TiledMMAQKInt_;

  static constexpr int VTiles = VTiles_;
  static constexpr bool PagedKV = true;
  static constexpr bool CausalMask = false;
  static constexpr bool LocalMask = false;
  static constexpr bool InitializeSplitScratchSentinels = true;
  static constexpr bool ExactLiveRows = ExactLiveRows_;
  static constexpr bool PagePair = PagePair_;
  static constexpr bool NextPagePrefetch = NextPagePrefetch_;
  static constexpr bool SimdPackedUnpack = SimdPackedUnpack_;
  static constexpr bool CurrentHalfVPrefetch = CurrentHalfVPrefetch_;
  static constexpr bool ReusePageRecordCursor = ReusePageRecordCursor_;
  static constexpr int QueryRows = cute::size<0>(TileShapeQK{});
  static constexpr int BiasRows = ExactLiveRows ? QueryRows : 8;
  static_assert(QueryRows <= 8);
  static_assert(!NextPagePrefetch || DpasPacked_);
  static_assert(!NextPagePrefetch || !PagePair);
  static_assert(!CurrentHalfVPrefetch || DpasPacked_);
  static_assert(!CurrentHalfVPrefetch || !PagePair);
  static_assert(
      !SimdPackedUnpack || (DpasPacked_ && VectorPackedLoads_ && !QKInt8U4_));
  // Each split stores a bounded normalized partial. KVarN reducers combine
  // it using weights reconstructed from the producer-written natural LSE.

  struct Arguments {
    typename Base::Arguments base;
    KVarNK4V4Layout kvarn;
    KVarNHybridTailLayout tail;
    int const* seq_lens;
  };

  struct Params {
    typename Base::Params base;
    KVarNK4V4Layout kvarn;
    KVarNHybridTailLayout tail;
    int const* seq_lens;
    // XeFMHAFwdSplitKVKernel reads these two scheduling fields directly from
    // the collective params. Keep them mirrored from the wrapped base params.
    int total_seqlen_kv;
    int window_size_left;
  };

  struct SharedStorage {
    typename Base::SharedStorage base;
  };

  Params params;

  KVarNDecodeFwdMainloop(Params const& params_, SharedStorage& storage)
      : Base(params_.base, storage.base), params(params_) {}

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const& args) {
    return args.kvarn.cache != nullptr && args.seq_lens != nullptr &&
           args.tail.block_to_slot != nullptr && args.tail.key != nullptr &&
           args.tail.value != nullptr && args.base.ptr_page_table != nullptr &&
           args.base.page_size == KVarNK4V4FragmentLoader<>::kGroup;
  }

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        Base::to_underlying_arguments(args.base, workspace),
        args.kvarn,
        args.tail,
        args.seq_lens,
        args.base.total_seqlen_kv,
        args.base.window_size_left};
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,
      TensorK2D const&,
      TensorV2D const&,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int const& idx_b,
      int blk_k0,
      int blk_k1,
      int,
      int thr_id,
      int seq_len,
      int,
      int) {
    // The generic kernel instantiates both surface branches. Packed K/V do
    // not use either surface path, and Q is always a small one-token surface.
    static_assert(get<1>(TileShapeQK{}) == 64);
    static_assert(get<2>(TileShapeQK{}) == 64);
    static_assert(get<2>(TileShapePV{}) == 64);

    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor cK = make_identity_tensor(select<1, 2>(TileShapeQK{}));  // (k,d)
    Tensor cV = make_identity_tensor(select<1, 2>(TileShapePV{}));  // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));

    Tensor gQ =
        local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});

    typename Base::TiledCopyQ copy_q{Q_2D};
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);
    auto qk_thr_coord = mma_qk.get_thr_layout_vmnk().get_flat_coord(thr_id);
    auto pv_thr_coord = mma_pv.get_thr_layout_vmnk().get_flat_coord(thr_id);
    int qk_token_sg = int(get<2>(qk_thr_coord)) * 16;
    int pv_token_sg = int(get<3>(pv_thr_coord)) * 16;

    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tSrK = thr_mma_qk.partition_sg_fragment_B(cK);
    auto tArV = thr_mma_pv.partition_sg_fragment_B(cV(_, _));

    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    for (int d = 0; d < size<3>(pQgQ); ++d) {
      prefetch(prefetch_q, pQgQ(_, _, _, d));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    KVarNK4V4FragmentLoader<DpasPacked_, VectorPackedLoads_, SimdPackedUnpack_>
        loader{
            params.kvarn,
            params.tail,
            params.base.ptr_page_table,
            params.base.max_pages_per_seq};
    int const actual_seq_len = params.seq_lens[idx_b];
    // Legacy single-split scheduler orders grid.z batch-major:
    //   flat = batch * num_kv_heads + kv_head.
    int kv_head = int(BlockIdxZ()) % 4;

    if constexpr (PagePair_) {
      // A scheduler work unit is one complete 128-token page for this
      // specialization.  Keep the established K64 MMA fragments, but build
      // both halves while the page lookup, query transform, and per-page
      // scalar metadata are live.  The two score fragments are then consumed
      // in their original order so online-softmax rounding remains aligned
      // with the scalar Q6 implementation.
      static_assert(DpasPacked_);
      static_assert(!VectorPackedLoads_);
      static_assert(!QKInt8U4_);
      auto tSrSSecond = thr_mma_qk.partition_sg_fragment_C(cP);

      for (int page = blk_k0; page < blk_k1; ++page) {
        int const first_k_tile = page * 2;
        if (page * KVarNK4V4FragmentLoader<>::kGroup >= actual_seq_len) {
          break;
        }
        // This predicate is uniform for the workgroup.  In particular, a
        // hybrid tail page ending in its first K64 half may leave the second
        // half entirely uninitialized; do not even materialize that half,
        // since a later PV MMA would otherwise allow 0 * NaN to poison the
        // output after its logits were masked.
        bool const has_live_second_half =
            (first_k_tile + 1) * 64 < actual_seq_len;

        // One lookup and address calculation serve both K64 halves.  This is
        // also important for the hybrid path: a page cannot change storage
        // class between its first and second half.
        int const physical = loader.physical_block(idx_b, first_k_tile);
        int const slot = params.tail.block_to_slot[physical];
        auto const* rec =
            slot < 0 ? params.kvarn.cache +
                           std::int64_t(physical) * params.kvarn.block_stride +
                           std::int64_t(kv_head) * params.kvarn.head_stride
                     : nullptr;

        clear(tSrS);
        clear(tSrSSecond);
        float k_zp_bias[BiasRows] = {};
        CUTLASS_PRAGMA_UNROLL
        for (int d_tile = 0; d_tile < 256; d_tile += 64) {
          // K column scale and zero point are page-wide.  Transform Q once,
          // then reuse the exact same fp16 fragment for both packed halves.
          copy(copy_q, tQgQ(_, _, _, d_tile / 64), tQrQ);
          reorder(tQrQ, tSrQ);
          if (slot < 0) {
            using KDimFragment = decltype(reduce<0>(tSrQ, sycl::plus<void>{}));
            static_assert(KDimFragment{}.size() == 4);
            KDimFragment k_dim_scale;
            KDimFragment k_dim_zp;
            int const lane = sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_id()[0];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < k_dim_scale.size(); ++i) {
              int const dim = d_tile + lane + i * intel::sg_size;
              k_dim_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.k_s_col_offset + 2 * dim);
              k_dim_zp(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.k_zp_offset + 2 * dim);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrQ.size(); ++i) {
              auto coord = tSrQ.tv_layout()(lane, i);
              int const query_row = int(get<0>(coord));
              float const query_value = static_cast<float>(tSrQ(i));
              float const dim_zp = broadcast<1>(k_dim_zp, tSrQ, i);
              k_zp_bias[query_row] += query_value * dim_zp;
              float const dim_scale = broadcast<1>(k_dim_scale, tSrQ, i);
              tSrQ(i) = static_cast<typename decltype(tSrQ)::value_type>(
                  query_value * dim_scale);
            }
          }

          loader.fill_k_fragment(
              tSrK, rec, slot, kv_head, first_k_tile, qk_token_sg, d_tile);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
          if (has_live_second_half) {
            loader.fill_k_fragment(
                tSrK,
                rec,
                slot,
                kv_head,
                first_k_tile + 1,
                qk_token_sg,
                d_tile);
            cute::gemm(mma_qk, tSrQ, tSrK, tSrSSecond);
          }
        }

        if (slot < 0) {
          auto subgroup = sycl::ext::oneapi::this_work_item::get_sub_group();
          CUTLASS_PRAGMA_UNROLL
          for (int query_row = 0; query_row < BiasRows; ++query_row) {
            k_zp_bias[query_row] = sycl::reduce_over_group(
                subgroup, k_zp_bias[query_row], sycl::plus<float>());
          }
          int const lane = subgroup.get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            auto coord = tSrS.tv_layout()(lane, i);
            float const bias = k_zp_bias[int(get<0>(coord))];
            tSrS(i) += bias;
            if (has_live_second_half) {
              tSrSSecond(i) += bias;
            }
          }

          FragSCol first_k_row_scale;
          FragSCol second_k_row_scale;
          int first_token = qk_token_sg + lane;
          int second_token = 64 + qk_token_sg + lane;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < first_k_row_scale.size(); ++i,
                   first_token += intel::sg_size,
                   second_token += intel::sg_size) {
            first_k_row_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                rec + params.kvarn.k_s_row_offset + 2 * first_token);
            if (has_live_second_half) {
              second_k_row_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.k_s_row_offset + 2 * second_token);
            }
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            tSrS(i) *= broadcast<1>(first_k_row_scale, tSrS, i);
            if (has_live_second_half) {
              tSrSSecond(i) *= broadcast<1>(second_k_row_scale, tSrSSecond, i);
            }
          }
        }

        // A physical page is always safe to read in full, including a hybrid
        // tail page.  Mask the unused tokens in the final partial page after
        // fragment materialization, exactly as the K64 baseline does.
        if ((first_k_tile + 1) * 64 > actual_seq_len) {
          FragSCol mask;
          int token = first_k_tile * 64 + qk_token_sg +
                      sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < mask.size(); ++i, token += intel::sg_size) {
            mask(i) = token < actual_seq_len ? ElementS(sycl::nan(0u))
                                             : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(mask, tSrS, i));
          }
        }
        if (has_live_second_half && (first_k_tile + 2) * 64 > actual_seq_len) {
          FragSCol mask;
          int token = (first_k_tile + 1) * 64 + qk_token_sg +
                      sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < mask.size(); ++i, token += intel::sg_size) {
            mask(i) = token < actual_seq_len ? ElementS(sycl::nan(0u))
                                             : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrSSecond.size(); ++i) {
            tSrSSecond(i) =
                sycl::fmin(tSrSSecond(i), broadcast<1>(mask, tSrSSecond, i));
          }
        }

        using VTokenFragment = decltype(reduce<0>(tArP, sycl::plus<void>{}));
        SingleFragA fragment_shape;
        using VDimFragment =
            decltype(reduce<0>(fragment_shape, sycl::plus<void>{}));
        VDimFragment page_v_dim_scale[VTiles];

        this->softmax(
            params.base.scale, page == blk_k0, tSrS, tA_max, tA_sum, tArA);
        reorder(tSrS, tArP);

        if (slot < 0) {
          float v_zp_bias[BiasRows] = {};
          int const lane = sycl::ext::oneapi::this_work_item::get_sub_group()
                               .get_local_id()[0];
          VTokenFragment v_token_scale;
          VTokenFragment v_token_zp;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < v_token_scale.size(); ++i) {
            int const token = pv_token_sg + lane + i * intel::sg_size;
            v_token_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                rec + params.kvarn.v_s_row_offset + 2 * token);
            v_token_zp(i) = KVarNK4V4FragmentLoader<>::load_f16(
                rec + params.kvarn.v_zp_offset + 2 * token);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArP.size(); ++i) {
            auto coord = tArP.tv_layout()(lane, i);
            int const query_row = int(get<0>(coord));
            float const probability = static_cast<float>(tArP(i));
            float const token_scale = broadcast<1>(v_token_scale, tArP, i);
            float const token_zp = broadcast<1>(v_token_zp, tArP, i);
            v_zp_bias[query_row] += probability * token_zp;
            tArP(i) = static_cast<typename decltype(tArP)::value_type>(
                probability * token_scale);
          }
          auto subgroup = sycl::ext::oneapi::this_work_item::get_sub_group();
          CUTLASS_PRAGMA_UNROLL
          for (int query_row = 0; query_row < BiasRows; ++query_row) {
            v_zp_bias[query_row] = sycl::reduce_over_group(
                subgroup, v_zp_bias[query_row], sycl::plus<float>());
          }

          CUTLASS_PRAGMA_UNROLL
          for (int vv = 0; vv < VTiles; ++vv) {
            auto& v_dim_scale = page_v_dim_scale[vv];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < v_dim_scale.size(); ++i) {
              int const dim =
                  vv * get<1>(TileShapePV{}) + lane + i * intel::sg_size;
              v_dim_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.v_s_col_offset + 2 * dim);
            }
            auto output_tile = tArA(_, _, _, vv);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < output_tile.size(); ++i) {
              output_tile(i) /= broadcast<1>(v_dim_scale, fragment_shape, i);
            }
            loader.fill_v_fragment(
                tArV,
                rec,
                slot,
                kv_head,
                first_k_tile,
                pv_token_sg,
                vv * get<1>(TileShapePV{}));
            cute::gemm(mma_pv, tArP, tArV, output_tile);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < output_tile.size(); ++i) {
              auto coord = fragment_shape.tv_layout()(lane, i);
              output_tile(i) += v_zp_bias[int(get<0>(coord))];
            }
          }
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int vv = 0; vv < VTiles; ++vv) {
            loader.fill_v_fragment(
                tArV,
                rec,
                slot,
                kv_head,
                first_k_tile,
                pv_token_sg,
                vv * get<1>(TileShapePV{}));
            cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, vv));
          }
        }

        if (!has_live_second_half) {
          // Packed V accumulation temporarily uses the page's inverse column
          // scale so both halves can share one scale frame.  Ordinarily the
          // second-half path restores that frame; do it here before leaving a
          // page whose second half is wholly inactive.
          if (slot < 0) {
            int const lane = sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_id()[0];
            CUTLASS_PRAGMA_UNROLL
            for (int vv = 0; vv < VTiles; ++vv) {
              auto output_tile = tArA(_, _, _, vv);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < output_tile.size(); ++i) {
                output_tile(i) *=
                    broadcast<1>(page_v_dim_scale[vv], fragment_shape, i);
              }
            }
          }
          continue;
        }

        // Consume the second half only after the first half's PV update.  The
        // output remains in this page's V-column-scale frame across the
        // second softmax rescale, preserving the established online ordering.
        this->softmax(
            params.base.scale, false, tSrSSecond, tA_max, tA_sum, tArA);
        reorder(tSrSSecond, tArP);

        if (slot < 0) {
          float v_zp_bias[BiasRows] = {};
          int const lane = sycl::ext::oneapi::this_work_item::get_sub_group()
                               .get_local_id()[0];
          VTokenFragment v_token_scale;
          VTokenFragment v_token_zp;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < v_token_scale.size(); ++i) {
            int const token = 64 + pv_token_sg + lane + i * intel::sg_size;
            v_token_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                rec + params.kvarn.v_s_row_offset + 2 * token);
            v_token_zp(i) = KVarNK4V4FragmentLoader<>::load_f16(
                rec + params.kvarn.v_zp_offset + 2 * token);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArP.size(); ++i) {
            auto coord = tArP.tv_layout()(lane, i);
            int const query_row = int(get<0>(coord));
            float const probability = static_cast<float>(tArP(i));
            float const token_scale = broadcast<1>(v_token_scale, tArP, i);
            float const token_zp = broadcast<1>(v_token_zp, tArP, i);
            v_zp_bias[query_row] += probability * token_zp;
            tArP(i) = static_cast<typename decltype(tArP)::value_type>(
                probability * token_scale);
          }
          auto subgroup = sycl::ext::oneapi::this_work_item::get_sub_group();
          CUTLASS_PRAGMA_UNROLL
          for (int query_row = 0; query_row < BiasRows; ++query_row) {
            v_zp_bias[query_row] = sycl::reduce_over_group(
                subgroup, v_zp_bias[query_row], sycl::plus<float>());
          }

          CUTLASS_PRAGMA_UNROLL
          for (int vv = 0; vv < VTiles; ++vv) {
            auto output_tile = tArA(_, _, _, vv);
            loader.fill_v_fragment(
                tArV,
                rec,
                slot,
                kv_head,
                first_k_tile + 1,
                pv_token_sg,
                vv * get<1>(TileShapePV{}));
            cute::gemm(mma_pv, tArP, tArV, output_tile);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < output_tile.size(); ++i) {
              auto coord = fragment_shape.tv_layout()(lane, i);
              output_tile(i) += v_zp_bias[int(get<0>(coord))];
              output_tile(i) *=
                  broadcast<1>(page_v_dim_scale[vv], fragment_shape, i);
            }
          }
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int vv = 0; vv < VTiles; ++vv) {
            loader.fill_v_fragment(
                tArV,
                rec,
                slot,
                kv_head,
                first_k_tile + 1,
                pv_token_sg,
                vv * get<1>(TileShapePV{}));
            cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, vv));
          }
        }
      }
      return;
    }

    for (int k_tile = blk_k0; k_tile < blk_k1; ++k_tile) {
      // The scheduler tiles to the batch maximum.  A workgroup owns exactly
      // one batch row, so this exit is uniform across all of its subgroups and
      // cannot strand a subgroup at the barrier below.  Besides avoiding
      // useless DPAS work, exiting before fragment materialization is
      // essential: padded block-table entries are not required to name valid
      // physical pages.
      if (k_tile * 64 >= actual_seq_len) {
        break;
      }
      int const physical = loader.physical_block(idx_b, k_tile);
      int const slot = params.tail.block_to_slot[physical];
      auto const* rec =
          slot < 0 ? params.kvarn.cache +
                         std::int64_t(physical) * params.kvarn.block_stride +
                         std::int64_t(kv_head) * params.kvarn.head_stride
                   : nullptr;
      if constexpr (NextPagePrefetch) {
        using Loader = KVarNK4V4FragmentLoader<
            DpasPacked_,
            VectorPackedLoads_,
            SimdPackedUnpack_>;
        static_assert(
            SGPerWG::value * cute::intel::sg_size ==
            Loader::kPagePrefetchThreads);
        // Issue one prefetch per physical page, early enough to overlap both
        // 64-token halves. A split beginning on an odd tile gets one-half-page
        // lead time. The two bounds prove the page-table entry is required by
        // both this split and this batch row before it is dereferenced.
        bool const first_owned_tile_in_page =
            (k_tile & 1) == 0 || k_tile == blk_k0;
        int const next_page_tile = (k_tile & ~1) + 2;
        bool const consumes_next_first_half =
            next_page_tile < blk_k1 && next_page_tile * 64 < actual_seq_len;
        if (first_owned_tile_in_page && consumes_next_first_half) {
          int const next_physical =
              loader.physical_block(idx_b, next_page_tile);
          int const next_slot = params.tail.block_to_slot[next_physical];
          if (next_slot < 0) {
            auto const* next_rec =
                params.kvarn.cache +
                std::int64_t(next_physical) * params.kvarn.block_stride +
                std::int64_t(kv_head) * params.kvarn.head_stride;
            bool const consumes_next_second_half =
                next_page_tile + 1 < blk_k1 &&
                (next_page_tile + 1) * 64 < actual_seq_len;
            if (consumes_next_second_half) {
              loader.template prefetch_dpas_page_l2<true>(next_rec, thr_id);
            } else {
              loader.template prefetch_dpas_page_l2<false>(next_rec, thr_id);
            }
          }
        }
      }
      clear(tSrS);
      float k_zp_bias[BiasRows] = {};
      if constexpr (QKInt8U4_) {
        if (slot < 0) {
          // Variant 1 keeps the PV path and cache ABI unchanged. Only packed
          // QK uses signed-int8 x unsigned-int4 DPAS. Quantize each 64-wide Q
          // slice independently so its scale can be folded back into the
          // corresponding integer accumulator before the four slices sum.
          TiledMMAQKInt mma_qk_int{};
          auto thr_mma_qk_int = mma_qk_int.get_slice(thr_id);
          auto tIrQ = thr_mma_qk_int.partition_sg_fragment_A(gQ(_, _, 0));
          auto tIrK = thr_mma_qk_int.partition_sg_fragment_B(cK);
          auto tIrS = thr_mma_qk_int.partition_sg_fragment_C(cP);
          auto tQrQInt8 = make_subgroup_tensor(
              make_tensor<std::int8_t>(tSrQ.layout()), tSrQ.tv_layout());
          auto tSrSPartial = make_subgroup_tensor(
              make_tensor<float>(tSrS.layout()), tSrS.tv_layout());

          CUTLASS_PRAGMA_UNROLL
          for (int d_tile = 0; d_tile < 256; d_tile += 64) {
            copy(copy_q, tQgQ(_, _, _, d_tile / 64), tQrQ);
            reorder(tQrQ, tSrQ);
            using KDimFragment = decltype(reduce<0>(tSrQ, sycl::plus<void>{}));
            static_assert(KDimFragment{}.size() == 4);
            KDimFragment k_dim_scale;
            int const lane = sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_id()[0];
            KDimFragment k_dim_zp;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < k_dim_scale.size(); ++i) {
              int const dim = d_tile + lane + i * intel::sg_size;
              k_dim_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.k_s_col_offset + 2 * dim);
              k_dim_zp(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.k_zp_offset + 2 * dim);
            }

            float q_amax = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrQ.size(); ++i) {
              auto coord = tSrQ.tv_layout()(lane, i);
              int const query_row = int(get<0>(coord));
              float const query_value = static_cast<float>(tSrQ(i));
              float const dim_zp = broadcast<1>(k_dim_zp, tSrQ, i);
              k_zp_bias[query_row] += query_value * dim_zp;
              float const dim_scale = broadcast<1>(k_dim_scale, tSrQ, i);
              float const scaled_query = query_value * dim_scale;
              tSrQ(i) = static_cast<typename decltype(tSrQ)::value_type>(
                  scaled_query);
              q_amax = sycl::fmax(q_amax, sycl::fabs(scaled_query));
            }

            auto subgroup = sycl::ext::oneapi::this_work_item::get_sub_group();
            q_amax = sycl::reduce_over_group(
                subgroup, q_amax, sycl::maximum<float>());
            float const q_scale = q_amax > 0.0f ? q_amax / 127.0f : 1.0f;
            float const q_inv_scale = 1.0f / q_scale;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrQ.size(); ++i) {
              float const quantized =
                  sycl::rint(static_cast<float>(tSrQ(i)) * q_inv_scale);
              tQrQInt8(i) = static_cast<std::int8_t>(
                  sycl::clamp(quantized, -127.0f, 127.0f));
            }
            reorder(tQrQInt8, tIrQ);
            loader.fill_k_fragment_by_coordinate(
                tIrK, rec, slot, kv_head, k_tile, qk_token_sg, d_tile);
            clear(tIrS);
            cute::gemm(mma_qk_int, tIrQ, tIrK, tIrS);
            reorder(tIrS, tSrSPartial);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
              tSrS(i) += tSrSPartial(i) * q_scale;
            }
          }
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int d_tile = 0; d_tile < 256; d_tile += 64) {
            copy(copy_q, tQgQ(_, _, _, d_tile / 64), tQrQ);
            reorder(tQrQ, tSrQ);
            loader.fill_k_fragment(
                tSrK, rec, slot, kv_head, k_tile, qk_token_sg, d_tile);
            cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
          }
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int d_tile = 0; d_tile < 256; d_tile += 64) {
          copy(copy_q, tQgQ(_, _, _, d_tile / 64), tQrQ);
          reorder(tQrQ, tSrQ);
          using KDimFragment = decltype(reduce<0>(tSrQ, sycl::plus<void>{}));
          static_assert(KDimFragment{}.size() == 4);
          KDimFragment k_dim_scale;
          if (slot < 0) {
            int const lane = sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_id()[0];
            KDimFragment k_dim_zp;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < k_dim_scale.size(); ++i) {
              int const dim = d_tile + lane + i * intel::sg_size;
              k_dim_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.k_s_col_offset + 2 * dim);
              k_dim_zp(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.k_zp_offset + 2 * dim);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrQ.size(); ++i) {
              auto coord = tSrQ.tv_layout()(lane, i);
              int const query_row = int(get<0>(coord));
              float const query_value = static_cast<float>(tSrQ(i));
              float const dim_zp = broadcast<1>(k_dim_zp, tSrQ, i);
              k_zp_bias[query_row] += query_value * dim_zp;
              // The column scale is constant across all 128 tokens in this
              // KVarN page. Apply it to the much smaller Q fragment instead of
              // scaling every unpacked K element.
              float const dim_scale = broadcast<1>(k_dim_scale, tSrQ, i);
              tSrQ(i) = static_cast<typename decltype(tSrQ)::value_type>(
                  query_value * dim_scale);
            }
          }
          loader.fill_k_fragment(
              tSrK, rec, slot, kv_head, k_tile, qk_token_sg, d_tile);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
      }

      if (slot < 0) {
        auto subgroup = sycl::ext::oneapi::this_work_item::get_sub_group();
        CUTLASS_PRAGMA_UNROLL
        for (int query_row = 0; query_row < BiasRows; ++query_row) {
          k_zp_bias[query_row] = sycl::reduce_over_group(
              subgroup, k_zp_bias[query_row], sycl::plus<float>());
        }
        int const lane = subgroup.get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          auto coord = tSrS.tv_layout()(lane, i);
          tSrS(i) += k_zp_bias[int(get<0>(coord))];
        }

        FragSCol k_row_scale;
        int token = (k_tile & 1) * 64 + qk_token_sg + lane;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_row_scale.size(); ++i, token += intel::sg_size) {
          k_row_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
              rec + params.kvarn.k_s_row_offset + 2 * token);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          tSrS(i) *= broadcast<1>(k_row_scale, tSrS, i);
        }
      }

      // The generic scheduler deliberately visits tiles up to max_seq_len.
      // Mask against this row's actual length on every tile, including tiles
      // wholly beyond the row, so a padded page table cannot contribute.
      if ((k_tile + 1) * 64 > actual_seq_len) {
        FragSCol mask;
        int token = k_tile * 64 + qk_token_sg +
                    sycl::ext::oneapi::this_work_item::get_sub_group()
                        .get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < mask.size(); ++i, token += intel::sg_size) {
          mask(i) = token < actual_seq_len ? ElementS(sycl::nan(0u))
                                           : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(mask, tSrS, i));
        }
      }

      if constexpr (CurrentHalfVPrefetch) {
        if (slot < 0) {
          loader.prefetch_dpas_v_half_l1(rec, k_tile & 1, thr_id);
        }
      }

      this->softmax(
          params.base.scale, k_tile == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      if (slot < 0) {
        bool const enter_v_scale_frame = (k_tile & 1) == 0 || k_tile == blk_k0;
        bool const leave_v_scale_frame = (k_tile & 1) == 1 ||
                                         k_tile + 1 == blk_k1 ||
                                         (k_tile + 1) * 64 >= actual_seq_len;
        float v_zp_bias[BiasRows] = {};
        int const lane = sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_id()[0];
        using VTokenFragment = decltype(reduce<0>(tArP, sycl::plus<void>{}));
        VTokenFragment v_token_scale;
        VTokenFragment v_token_zp;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < v_token_scale.size(); ++i) {
          int const token =
              (k_tile & 1) * 64 + pv_token_sg + lane + i * intel::sg_size;
          v_token_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
              rec + params.kvarn.v_s_row_offset + 2 * token);
          v_token_zp(i) = KVarNK4V4FragmentLoader<>::load_f16(
              rec + params.kvarn.v_zp_offset + 2 * token);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tArP.size(); ++i) {
          auto coord = tArP.tv_layout()(lane, i);
          int const query_row = int(get<0>(coord));
          float const probability = static_cast<float>(tArP(i));
          float const token_scale = broadcast<1>(v_token_scale, tArP, i);
          float const token_zp = broadcast<1>(v_token_zp, tArP, i);
          v_zp_bias[query_row] += probability * token_zp;
          tArP(i) = static_cast<typename decltype(tArP)::value_type>(
              probability * token_scale);
        }
        auto subgroup = sycl::ext::oneapi::this_work_item::get_sub_group();
        CUTLASS_PRAGMA_UNROLL
        for (int query_row = 0; query_row < BiasRows; ++query_row) {
          v_zp_bias[query_row] = sycl::reduce_over_group(
              subgroup, v_zp_bias[query_row], sycl::plus<float>());
        }

        CUTLASS_PRAGMA_UNROLL
        for (int vv = 0; vv < VTiles; ++vv) {
          loader.fill_v_fragment(
              tArV,
              rec,
              slot,
              kv_head,
              k_tile,
              pv_token_sg,
              vv * get<1>(TileShapePV{}));
          SingleFragA fragment_shape;
          auto output_tile = tArA(_, _, _, vv);
          using VDimFragment =
              decltype(reduce<0>(fragment_shape, sycl::plus<void>{}));
          VDimFragment v_dim_scale;
          if (enter_v_scale_frame || leave_v_scale_frame) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < v_dim_scale.size(); ++i) {
              int const dim =
                  vv * get<1>(TileShapePV{}) + lane + i * intel::sg_size;
              v_dim_scale(i) = KVarNK4V4FragmentLoader<>::load_f16(
                  rec + params.kvarn.v_s_col_offset + 2 * dim);
            }
          }
          // Accumulate directly into the online-softmax output fragment. Put
          // its existing value in this page's column-scale frame, add the raw
          // DPAS and zero-point contributions for both 64-token halves, then
          // restore the scale at the 128-token page (or split) boundary.
          // Online-softmax rescaling is linear, so it is valid in this frame.
          // For a single contribution the algebra is
          //   s * (old / s + q_acc + zp) = old + s * (q_acc + zp),
          // while avoiding a second full SingleFragA register fragment.
          if (enter_v_scale_frame) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < output_tile.size(); ++i) {
              float const dim_scale =
                  broadcast<1>(v_dim_scale, fragment_shape, i);
              output_tile(i) /= dim_scale;
            }
          }
          cute::gemm(mma_pv, tArP, tArV, output_tile);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < output_tile.size(); ++i) {
            auto coord = fragment_shape.tv_layout()(lane, i);
            int const query_row = int(get<0>(coord));
            output_tile(i) += v_zp_bias[query_row];
          }
          if (leave_v_scale_frame) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < output_tile.size(); ++i) {
              float const dim_scale =
                  broadcast<1>(v_dim_scale, fragment_shape, i);
              output_tile(i) *= dim_scale;
            }
          }
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int vv = 0; vv < VTiles; ++vv) {
          loader.fill_v_fragment(
              tArV,
              rec,
              slot,
              kv_head,
              k_tile,
              pv_token_sg,
              vv * get<1>(TileShapePV{}));
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, vv));
        }
      }
      // Packed K/V fragments and online-softmax accumulators are subgroup-
      // local registers. Unlike the generic surface mainloop, this path has
      // no shared-memory stage to publish before the next tile, so a
      // workgroup-wide barrier here only serializes independent query-head
      // subgroups.
    }
  }
};

}  // namespace cutlass::fmha::collective
