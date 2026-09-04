#pragma once

#include "collective/kvarn_decode_mainloop.hpp"
#include "paged_decode.hpp"

/** Host-facing arguments for the deliberately narrow native decode spike.
 *
 * The first integration should expose these fields through a caller-owned
 * output op. The generic kernel still requires valid row-statistic scratch
 * pointers even though num_kv_splits is fixed to one.
 */
struct kvarn_decode_args_t {
  void const* query;  // [B, 24, 256] fp16, Hadamard-rotated
  std::uint8_t const* cache;
  void* output;              // [B,24,256] fp16, or bf16 when fused/unrotated
  void* temp_output;         // [B, 24 * splits, 256] fp16 partials
  float* softmax_lse_accum;  // [B, 24, splits] natural-log LSE scratch
  float*
      legacy_max_logits;   // validated ABI scratch; unused after LSE migration
  int const* block_table;  // [B, max_pages_per_seq]
  int const* seq_lens;     // [B], actual length for each batch row
  int const* block_to_slot;  // [num_physical_blocks], -1 selects KVarN
  void const* tail_key;      // [slots, 128, 4, 256] fp16, rotated
  void const* tail_value;    // [slots, 128, 4, 256] fp16, rotated
  int batch_size;            // B1-B12 decode rows
  int max_seq_len;           // maximum length used by the scheduler
  int max_pages_per_seq;
  int num_kv_splits;
  float softmax_scale;
  bool unrotate_output;
  bool write_bf16_output;
  cutlass::fmha::collective::KVarNK4V4Layout layout;
};

/** Compile-time policy contract for the first KVarN kernel.
 *
 * Page 128 intentionally uses the existing tile-64 policy.  The repository's
 * tile-128/ReduceK=8 path has a known cross-subgroup reduction correctness
 * issue, while tile 64 visits each KVarN page in exactly two iterations.
 */
template <class QPacked>
struct KVarNDecodeD256G128PolicyImpl {
  static_assert(
      cute::is_same_v<QPacked, cute::Int<6>> ||
          cute::is_same_v<QPacked, cute::_8>,
      "KVarN D256/G128 supports only the exact GQA-6 and legacy Q8 tiles");
  using BasePolicy = decode_policy_qpacked_head<QPacked, cute::_256, cute::_64>;
  using ShapeQK = typename BasePolicy::ShapeQK;
  using ShapePV = typename BasePolicy::ShapePV;
  using ShapeOut = typename BasePolicy::ShapeOut;
  using SubgroupLayoutQK = typename BasePolicy::SubgroupLayoutQK;

  static constexpr int HeadDim = 256;
  static constexpr int PageSize = 128;
  static constexpr int KVTile = 64;
  static constexpr int NumQueryHeads = 24;
  static constexpr int NumKVHeads = 4;
  static constexpr int QueryHeadsPerKV = 6;
};

using KVarNDecodeD256G128Policy = KVarNDecodeD256G128PolicyImpl<cute::_8>;
using KVarNDecodeD256G128Q6Policy = KVarNDecodeD256G128PolicyImpl<cute::Int<6>>;

static_assert(cute::size<1>(KVarNDecodeD256G128Policy::ShapeQK{}) == 64);
static_assert(cute::size<1>(KVarNDecodeD256G128Policy::ShapeOut{}) == 256);

/** KVarN-only reduction for the serving-optimal fixed 16-way KV split.
 *
 * Keeping this kernel here is intentional: putting the specialization in the
 * generic paged-decode reducer forces every attention configuration to be
 * re-instantiated during each KVarN tuning iteration.
 */
template <int KVWorkUnitTokens>
struct KVarNReduceSplit16Kernel {
  using Element = cutlass::half_t;
  static_assert(cute::intel::sg_size == 16);

  struct Params {
    Element* output;
    Element const* partial_output;
    float const* softmax_lse_accum;
    int const* seq_lens;
    int kv_tiles_per_split;
  };

  struct SharedStorage {};
  static constexpr int SharedStorageSize = 0;

  CUTLASS_DEVICE
  void operator()(Params const& params, char*) {
    using namespace sycl::ext::oneapi::this_work_item;

    constexpr int kSplits = 16;
    constexpr int kThreads = 128;
    constexpr int kSubgroups = kThreads / cute::intel::sg_size;

    int const head = int(BlockIdxY());
    int const batch = int(BlockIdxZ());
    int const thread = int(ThreadIdxX());
    int const lane = thread % cute::intel::sg_size;
    int const subgroup_id = thread / cute::intel::sg_size;
    auto subgroup = get_sub_group();

    int const kv_tiles =
        cute::ceil_div(params.seq_lens[batch], KVWorkUnitTokens);
    int const active_splits =
        cute::ceil_div(kv_tiles, params.kv_tiles_per_split);
    int const stats_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) * kSplits +
        lane;

    float const invalid_lse =
        cutlass::platform::numeric_limits<float>::lowest();
    float const local_lse = lane < active_splits
                                ? params.softmax_lse_accum[stats_offset]
                                : invalid_lse;
    bool const valid = local_lse > invalid_lse;

    float const global_max_lse =
        sycl::reduce_over_group(subgroup, local_lse, sycl::maximum<>());
    constexpr float kLog2e = 1.4426950408889634f;
    float const weight =
        valid ? sycl::native::exp2((local_lse - global_max_lse) * kLog2e)
              : 0.0f;
    float const denominator =
        sycl::reduce_over_group(subgroup, weight, sycl::plus<>());
    float const inverse_denominator = 1.0f / denominator;

    for (int dim = subgroup_id; dim < KVarNDecodeD256G128Policy::HeadDim;
         dim += kSubgroups) {
      int const partial_offset =
          ((batch * kSplits + lane) * KVarNDecodeD256G128Policy::NumQueryHeads +
           head) *
              KVarNDecodeD256G128Policy::HeadDim +
          dim;
      float const partial =
          valid ? static_cast<float>(params.partial_output[partial_offset]) *
                      weight
                : 0.0f;
      float const numerator =
          sycl::reduce_over_group(subgroup, partial, sycl::plus<>());
      if (lane == 0) {
        int const output_offset =
            (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) *
                KVarNDecodeD256G128Policy::HeadDim +
            dim;
        params.output[output_offset] =
            static_cast<Element>(numerator * inverse_denominator);
      }
    }
  }
};

/** KVarN-only reduction for the B12 serving-optimal 32-way KV split.
 *
 * The generic reducer launches several workgroups per output row. KVarN's
 * fixed D256 shape instead fits one output dimension per thread and all 32
 * split statistics in a small shared array, so one workgroup completes each
 * (batch, query-head) row without another scheduling level.
 */
template <int KVWorkUnitTokens>
struct KVarNReduceSplit32Kernel {
  using Element = cutlass::half_t;

  struct Params {
    Element* output;
    Element const* partial_output;
    float const* softmax_lse_accum;
    int const* seq_lens;
    int kv_tiles_per_split;
  };

  static constexpr int kSplits = 32;
  static constexpr int kThreads = KVarNDecodeD256G128Policy::HeadDim;
  static constexpr int SharedStorageSize = kSplits * sizeof(float);

  CUTLASS_DEVICE
  void operator()(Params const& params, char* shared_storage) {
    using namespace sycl::ext::oneapi::this_work_item;

    int const head = int(BlockIdxY());
    int const batch = int(BlockIdxZ());
    int const thread = int(ThreadIdxX());
    auto workgroup = get_work_group<3>();
    auto* split_weights = reinterpret_cast<float*>(shared_storage);

    int const kv_tiles =
        cute::ceil_div(params.seq_lens[batch], KVWorkUnitTokens);
    int const active_splits =
        cute::ceil_div(kv_tiles, params.kv_tiles_per_split);
    int const stats_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) * kSplits +
        thread;
    float const invalid_lse =
        cutlass::platform::numeric_limits<float>::lowest();
    float const local_lse = thread < active_splits
                                ? params.softmax_lse_accum[stats_offset]
                                : invalid_lse;
    bool const valid = local_lse > invalid_lse;

    float const global_max_lse =
        sycl::reduce_over_group(workgroup, local_lse, sycl::maximum<>());
    constexpr float kLog2e = 1.4426950408889634f;
    if (thread < kSplits) {
      split_weights[thread] =
          valid ? sycl::native::exp2((local_lse - global_max_lse) * kLog2e)
                : 0.0f;
    }
    sycl::group_barrier(workgroup);

    if (thread < KVarNDecodeD256G128Policy::HeadDim) {
      float numerator = 0.0f;
      float denominator = 0.0f;
      for (int split = 0; split < active_splits; ++split) {
        float const weight = split_weights[split];
        if (weight <= 0.0f) continue;
        int const partial_offset =
            ((batch * kSplits + split) *
                 KVarNDecodeD256G128Policy::NumQueryHeads +
             head) *
                KVarNDecodeD256G128Policy::HeadDim +
            thread;
        numerator +=
            static_cast<float>(params.partial_output[partial_offset]) * weight;
        denominator += weight;
      }
      int const output_offset =
          (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) *
              KVarNDecodeD256G128Policy::HeadDim +
          thread;
      params.output[output_offset] =
          static_cast<Element>(numerator / denominator);
    }
  }
};

/** KVarN-only split reducer with a fused inverse H256 output transform.
 *
 * The standalone H256 kernel reads the fp16 result of split reduction, so the
 * fused path deliberately keeps that fp16 rounding boundary before applying
 * the float butterflies.  One workgroup owns a complete output row: its first
 * 32 threads load split statistics, all 256 threads reduce one output
 * dimension each, and the first 128 threads perform one butterfly per H256
 * stage.  This removes one launch without repeating the transform in every
 * producer split workgroup.
 */
template <int KVWorkUnitTokens>
struct KVarNReduceSplitOutputHadamardKernel {
  using Element = cutlass::half_t;

  static constexpr int kMaxSplits = 32;
  static constexpr int kThreads = KVarNDecodeD256G128Policy::HeadDim;

  struct Params {
    void* output;
    Element const* partial_output;
    float const* softmax_lse_accum;
    int num_kv_splits;
    int const* seq_lens;
    int kv_tiles_per_split;
    bool write_bf16_output;
  };

  struct SharedStorage {
    cutlass::Array<float, kMaxSplits> split_weights;
    cutlass::Array<float, KVarNDecodeD256G128Policy::HeadDim> output_row;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE
  void operator()(Params const& params, char* shared_storage) {
    using namespace sycl::ext::oneapi::this_work_item;

    int const head = int(BlockIdxY());
    int const batch = int(BlockIdxZ());
    int const thread = int(ThreadIdxX());
    auto workgroup = get_work_group<3>();
    auto& storage = *reinterpret_cast<SharedStorage*>(shared_storage);

    int const kv_tiles =
        cute::ceil_div(params.seq_lens[batch], KVWorkUnitTokens);
    int const active_splits =
        cute::ceil_div(kv_tiles, params.kv_tiles_per_split);
    int const stats_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) *
            params.num_kv_splits +
        thread;
    float const invalid_lse =
        cutlass::platform::numeric_limits<float>::lowest();
    float const local_lse = thread < active_splits
                                ? params.softmax_lse_accum[stats_offset]
                                : invalid_lse;
    bool const valid = local_lse > invalid_lse;

    float const global_max_lse =
        sycl::reduce_over_group(workgroup, local_lse, sycl::maximum<>());
    constexpr float kLog2e = 1.4426950408889634f;
    if (thread < params.num_kv_splits) {
      storage.split_weights[thread] =
          valid ? sycl::native::exp2((local_lse - global_max_lse) * kLog2e)
                : 0.0f;
    }
    sycl::group_barrier(workgroup);

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
      float const weight = storage.split_weights[split];
      if (weight <= 0.0f) continue;
      int const partial_offset = ((batch * params.num_kv_splits + split) *
                                      KVarNDecodeD256G128Policy::NumQueryHeads +
                                  head) *
                                     KVarNDecodeD256G128Policy::HeadDim +
                                 thread;
      numerator +=
          static_cast<float>(params.partial_output[partial_offset]) * weight;
      denominator += weight;
    }

    // Match the established reducer -> fp16 output -> H256 path exactly at
    // the operation boundary.  Keeping the half rounding here avoids a fused
    // mode silently changing the numerical contract.
    Element const reduced = static_cast<Element>(numerator / denominator);
    storage.output_row[thread] = static_cast<float>(reduced);
    sycl::group_barrier(workgroup);

    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < 8; ++stage) {
      int const span = 1 << stage;
      if (thread < KVarNDecodeD256G128Policy::HeadDim / 2) {
        int const pair = thread / span;
        int const offset = thread - pair * span;
        int const low = pair * 2 * span + offset;
        int const high = low + span;
        float const a = storage.output_row[low];
        float const b = storage.output_row[high];
        storage.output_row[low] = a + b;
        storage.output_row[high] = a - b;
      }
      sycl::group_barrier(workgroup);
    }

    int const output_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) *
            KVarNDecodeD256G128Policy::HeadDim +
        thread;
    // Preserve the second fp16 rounding boundary from the historical fused
    // reducer before optionally converting into the public bf16 output.  This
    // makes direct-bf16 output bitwise equivalent to fp16 output followed by
    // Tensor.copy_ into bf16.
    Element const transformed =
        static_cast<Element>(storage.output_row[thread] * (1.0f / 16.0f));
    if (params.write_bf16_output) {
      using BFloat16 = sycl::ext::oneapi::bfloat16;
      reinterpret_cast<BFloat16*>(params.output)[output_offset] =
          static_cast<BFloat16>(static_cast<float>(transformed));
    } else {
      reinterpret_cast<Element*>(params.output)[output_offset] = transformed;
    }
  }
};

/** Compile-time split-count version of the fused KVarN output reducer.
 *
 * B70 production uses a small fixed set of split counts.  Making NumSplits a
 * template argument gives the compiler a constant scratch stride and a bounded
 * accumulation loop while leaving the producer kernel and scratch layout
 * unchanged.  Inactive per-row splits still carry the established lowest-float
 * LSE sentinel and are skipped before their uninitialized partial is read.
 */
template <int NumSplits, int KVWorkUnitTokens>
struct KVarNReduceSplitOutputHadamardSpecializedKernel {
  using Element = cutlass::half_t;
  using Params =
      typename KVarNReduceSplitOutputHadamardKernel<KVWorkUnitTokens>::Params;

  static_assert(
      NumSplits == 2 || NumSplits == 4 || NumSplits == 8 || NumSplits == 16 ||
          NumSplits == 32,
      "specialized KVarN reducer supports only B70 production split counts");
  static constexpr int kThreads = KVarNDecodeD256G128Policy::HeadDim;

  struct SharedStorage {
    cutlass::Array<float, NumSplits> split_weights;
    cutlass::Array<float, KVarNDecodeD256G128Policy::HeadDim> output_row;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE
  void operator()(Params const& params, char* shared_storage) {
    using namespace sycl::ext::oneapi::this_work_item;

    int const head = int(BlockIdxY());
    int const batch = int(BlockIdxZ());
    int const thread = int(ThreadIdxX());
    auto workgroup = get_work_group<3>();
    auto& storage = *reinterpret_cast<SharedStorage*>(shared_storage);

    int const kv_tiles =
        cute::ceil_div(params.seq_lens[batch], KVWorkUnitTokens);
    int const active_splits =
        cute::ceil_div(kv_tiles, params.kv_tiles_per_split);
    int const stats_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) * NumSplits +
        thread;
    float const invalid_lse =
        cutlass::platform::numeric_limits<float>::lowest();
    float const local_lse = thread < active_splits
                                ? params.softmax_lse_accum[stats_offset]
                                : invalid_lse;
    bool const valid = local_lse > invalid_lse;

    float const global_max_lse =
        sycl::reduce_over_group(workgroup, local_lse, sycl::maximum<>());
    constexpr float kLog2e = 1.4426950408889634f;
    if (thread < NumSplits) {
      storage.split_weights[thread] =
          valid ? sycl::native::exp2((local_lse - global_max_lse) * kLog2e)
                : 0.0f;
    }
    sycl::group_barrier(workgroup);

    float numerator = 0.0f;
    float denominator = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int split = 0; split < NumSplits; ++split) {
      float const weight = storage.split_weights[split];
      if (weight <= 0.0f) continue;
      int const partial_offset = ((batch * NumSplits + split) *
                                      KVarNDecodeD256G128Policy::NumQueryHeads +
                                  head) *
                                     KVarNDecodeD256G128Policy::HeadDim +
                                 thread;
      numerator +=
          static_cast<float>(params.partial_output[partial_offset]) * weight;
      denominator += weight;
    }

    // Preserve the generic reducer's two fp16 rounding boundaries exactly.
    Element const reduced = static_cast<Element>(numerator / denominator);
    storage.output_row[thread] = static_cast<float>(reduced);
    sycl::group_barrier(workgroup);

    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < 8; ++stage) {
      int const span = 1 << stage;
      if (thread < KVarNDecodeD256G128Policy::HeadDim / 2) {
        int const pair = thread / span;
        int const offset = thread - pair * span;
        int const low = pair * 2 * span + offset;
        int const high = low + span;
        float const a = storage.output_row[low];
        float const b = storage.output_row[high];
        storage.output_row[low] = a + b;
        storage.output_row[high] = a - b;
      }
      sycl::group_barrier(workgroup);
    }

    int const output_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) *
            KVarNDecodeD256G128Policy::HeadDim +
        thread;
    Element const transformed =
        static_cast<Element>(storage.output_row[thread] * (1.0f / 16.0f));
    if (params.write_bf16_output) {
      using BFloat16 = sycl::ext::oneapi::bfloat16;
      reinterpret_cast<BFloat16*>(params.output)[output_offset] =
          static_cast<BFloat16>(static_cast<float>(transformed));
    } else {
      reinterpret_cast<Element*>(params.output)[output_offset] = transformed;
    }
  }
};

/** Concrete, intentionally narrow native decode configuration.
 *
 * This is kept separate from PagedDecodeConfig because K and V are not
 * tensors in memory: KVarNDecodeFwdMainloop materializes their MMA fragments
 * directly from the packed record.  The dummy K/V tensor types only satisfy
 * the surrounding XeFMHA kernel's type contract and are never dereferenced by
 * the custom mainloop.
 */
template <
    bool DpasPacked = false,
    class QPacked = cute::_8,
    bool VectorPackedLoads = false,
    bool QKInt8U4 = false,
    bool CacheScalarWeights = false,
    bool ExactLiveRows = false,
    bool PagePair = false,
    int MainKernelGrfSize = 256,
    bool SpecializedSplitReducer = false,
    bool NextPagePrefetch = false>
struct KVarNDecodeD256G128ConfigImpl {
  static_assert(!VectorPackedLoads || DpasPacked);
  static_assert(!QKInt8U4 || DpasPacked);
  static_assert(
      !QKInt8U4 || cute::is_same_v<QPacked, cute::_8>,
      "integer QK remains isolated from the exact-Q6 experiment");
  static_assert(
      !CacheScalarWeights || cute::is_same_v<QPacked, cute::Int<6>>,
      "cached scalar weights are an exact-Q6 experiment");
  static_assert(
      !ExactLiveRows || cute::is_same_v<QPacked, cute::Int<6>>,
      "exact live-row loops are an exact-Q6 experiment");
  static_assert(!PagePair || DpasPacked);
  static_assert(!PagePair || cute::is_same_v<QPacked, cute::Int<6>>);
  static_assert(!PagePair || !VectorPackedLoads);
  static_assert(!PagePair || !QKInt8U4);
  static_assert(
      MainKernelGrfSize == 128 || MainKernelGrfSize == 256,
      "KVarN main-kernel GRF size must be 128 or 256");
  static constexpr int MainGrfSize = MainKernelGrfSize;
  static_assert(
      !SpecializedSplitReducer ||
          (DpasPacked && cute::is_same_v<QPacked, cute::Int<6>>),
      "specialized split reduction is an exact-Q6 DPAS experiment");

  static constexpr bool UsesSpecializedSplitReducer = SpecializedSplitReducer;
  static_assert(
      !NextPagePrefetch ||
          (DpasPacked && cute::is_same_v<QPacked, cute::Int<6>>),
      "next-page prefetch is an exact-Q6 DPAS experiment");
  static_assert(!NextPagePrefetch || !PagePair);

  using Policy = KVarNDecodeD256G128PolicyImpl<QPacked>;
  using TileShapeQK = typename Policy::ShapeQK;
  using TileShapePV = typename Policy::ShapePV;
  using TileShapeO = typename Policy::ShapeOut;
  using SubgroupLayoutQK = typename Policy::SubgroupLayoutQK;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using Element = cutlass::half_t;
  using StrideQ = cute::Stride<int, cute::_1, int, int>;
  using StrideK = cute::Stride<int, cute::_1, int, int>;
  using StrideV = cute::Stride<cute::_1, int, int, int>;
  using StrideO = cute::Stride<int, cute::_1, int, int>;

  static constexpr int SGTileQ = cute::get<0>(
      cute::shape_div(TileShapeQK{}, cute::shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, Element>;
  using TiledMMAQK = typename TiledMMAHelper<
      cute::MMA_Atom<MMAOperation>,
      cute::Layout<TileShapeQK>,
      SubgroupLayoutQK>::TiledMMA;
  using MMAOperationQKInt = cute::XE_DPAS_TT<
      cute::gcd(SGTileQ, 8),
      std::int32_t,
      std::int8_t,
      cute::uint4_t,
      std::int32_t>;
  using TiledMMAQKInt = typename TiledMMAHelper<
      cute::MMA_Atom<MMAOperationQKInt>,
      cute::Layout<TileShapeQK>,
      SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<
      cute::MMA_Atom<MMAOperation>,
      cute::Layout<TileShapePV>,
      SubgroupLayoutPV>::TiledMMA;

  static constexpr int VTiles =
      cute::get<1>(TileShapeO{}) / cute::get<1>(TileShapePV{});

  template <class T, class Stride>
  using DummyTensor = decltype(cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T*>(nullptr)),
      cute::make_layout(cute::repeat<cute::rank_v<Stride>>(1), Stride{})));

  using TensorQ = DummyTensor<Element, StrideQ>;
  using TensorK = DummyTensor<Element, StrideK>;
  using TensorV = DummyTensor<Element, StrideV>;
  using TensorO = DummyTensor<Element, StrideO>;
  using TensorLSE = DummyTensor<float, StrideO>;

  using Mainloop = cutlass::fmha::collective::KVarNDecodeFwdMainloop<
      TiledMMAQK,
      TiledMMAQKInt,
      TiledMMAPV,
      VTiles,
      TensorQ,
      TensorK,
      TensorV,
      DpasPacked,
      VectorPackedLoads,
      QKInt8U4,
      ExactLiveRows,
      PagePair,
      NextPagePrefetch>;
  using Epilogue = cutlass::fmha::collective::DecodeFwdEpilogue<
      Mainloop,
      TileShapeO,
      TensorO,
      TensorLSE,
      void,
      false,
      cute::is_same_v<QPacked, cute::Int<6>>,
      CacheScalarWeights>;
  using ProblemShape = cutlass::fmha::kernel::DecodeProblemShape<false>;
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<
      ProblemShape,
      Mainloop,
      Epilogue,
      cutlass::fmha::kernel::DecodeTileScheduler>;
  using ReductionSplitKernel = cutlass::fmha::kernel::ReduceSplitK<
      ProblemShape,
      cutlass::fmha::kernel::XeReduceSplitKTileScheduler,
      Kernel>;
  static constexpr int KVWorkUnitTokens =
      PagePair ? Policy::PageSize : Policy::KVTile;
  using ReductionSplit16Kernel = KVarNReduceSplit16Kernel<KVWorkUnitTokens>;
  using ReductionSplit32Kernel = KVarNReduceSplit32Kernel<KVWorkUnitTokens>;
  using ReductionSplitOutputHadamardKernel =
      KVarNReduceSplitOutputHadamardKernel<KVWorkUnitTokens>;
  template <int NumSplits>
  using ReductionSplitOutputHadamardSpecializedKernel =
      KVarNReduceSplitOutputHadamardSpecializedKernel<
          NumSplits,
          KVWorkUnitTokens>;

  static cutlass::Status
  run(sycl::queue& queue, kvarn_decode_args_t const& args) {
    if (args.batch_size < 1 || args.batch_size > 12 || args.max_seq_len <= 0 ||
        args.max_pages_per_seq <= 0 || args.query == nullptr ||
        args.output == nullptr || args.temp_output == nullptr ||
        args.softmax_lse_accum == nullptr ||
        args.legacy_max_logits == nullptr || args.num_kv_splits < 1 ||
        args.block_table == nullptr || args.seq_lens == nullptr ||
        args.cache == nullptr || args.block_to_slot == nullptr ||
        args.tail_key == nullptr || args.tail_value == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    if constexpr (VectorPackedLoads) {
      constexpr std::uintptr_t kKVectorAlignment = 32;
      constexpr std::int64_t kVVectorAlignment = 16;
      auto const cache_address = reinterpret_cast<std::uintptr_t>(args.cache);
      if (cache_address % kKVectorAlignment != 0 ||
          args.layout.block_stride % kKVectorAlignment != 0 ||
          args.layout.head_stride % kKVectorAlignment != 0 ||
          args.layout.k_packed_offset % kKVectorAlignment != 0 ||
          args.layout.v_packed_offset % kVVectorAlignment != 0) {
        return cutlass::Status::kErrorInvalidProblem;
      }
    }

    // DecodeTileScheduler derives its work ranges from ShapeQK's K64 tile.
    // For PagePair, present one synthetic K64 scheduling unit per physical
    // 128-token page; the specialized mainloop expands each unit back into
    // the two immutable xe2_dpas cache halves.
    int const scheduler_seq_len =
        PagePair ? cute::ceil_div(args.max_seq_len, Policy::PageSize) *
                       Policy::KVTile
                 : args.max_seq_len;
    ProblemShape shape{
        args.batch_size,
        Policy::NumQueryHeads,
        Policy::NumKVHeads,
        1,
        scheduler_seq_len,
        Policy::HeadDim,
        Policy::HeadDim};

    auto stride_q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(
            1, Policy::HeadDim, Policy::NumQueryHeads, args.batch_size));
    // Type-contract-only layouts; the custom mainloop never dereferences K/V.
    auto stride_k = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(
            args.max_seq_len,
            Policy::HeadDim,
            Policy::NumKVHeads,
            args.batch_size));
    auto stride_v = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(
            Policy::HeadDim,
            args.max_seq_len,
            Policy::NumKVHeads,
            args.batch_size));
    auto stride_o = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(
            1, Policy::HeadDim, Policy::NumQueryHeads, args.batch_size));
    auto stride_o_accum = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(
            1,
            Policy::HeadDim,
            Policy::NumQueryHeads * args.num_kv_splits,
            args.batch_size));
    // LSE buffers are unused in the one-split direct-output path.  Preserve a
    // valid stride because it remains part of the generic kernel contract.
    auto stride_lse = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(
            1, args.num_kv_splits, Policy::NumQueryHeads, args.batch_size));

    auto const* q = reinterpret_cast<Element const*>(args.query);
    auto* out = reinterpret_cast<Element*>(args.output);
    // A one-split launch has no reduction kernel, so its mainloop must write
    // the final tensor directly.  Multi-split launches retain partials in the
    // caller-owned scratch buffer for the following reducer.
    auto* mainloop_out = args.num_kv_splits == 1
                             ? out
                             : reinterpret_cast<Element*>(args.temp_output);
    auto kvarn_layout = args.layout;
    kvarn_layout.cache = args.cache;
    cutlass::fmha::collective::KVarNHybridTailLayout tail_layout{
        args.block_to_slot,
        reinterpret_cast<Element const*>(args.tail_key),
        reinterpret_cast<Element const*>(args.tail_value),
        Policy::PageSize * Policy::NumKVHeads * Policy::HeadDim,
        Policy::NumKVHeads * Policy::HeadDim,
        Policy::HeadDim};
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    typename Kernel::Arguments kernel_args{
        {shape,
         q,
         stride_q,
         q,
         stride_k,
         q,
         stride_v,
         mainloop_out,
         stride_o_accum,
         args.softmax_lse_accum,
         stride_lse,
         nullptr,
         nullptr,
         nullptr,
         nullptr,
         0},
        {{args.softmax_scale,
          nullptr,
          nullptr,
          const_cast<int*>(args.block_table),
          Policy::PageSize,
          args.max_pages_per_seq,
          args.max_seq_len,
          -1,
          -1,
          0},
         kvarn_layout,
         tail_layout,
         args.seq_lens},
        {},
        hw_info,
        args.num_kv_splits};

    typename ReductionSplitKernel::Arguments reduce_args{
        {shape,
         out,
         stride_o,
         reinterpret_cast<Element const*>(args.temp_output),
         stride_o_accum,
         args.softmax_lse_accum,
         stride_lse,
         -1,
         nullptr,
         nullptr,
         nullptr,
         0},
        hw_info,
        args.num_kv_splits};

    if (!Kernel::can_implement(kernel_args)) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    auto params = Kernel::to_underlying_arguments(kernel_args, nullptr);
    auto reduce_params =
        ReductionSplitKernel::to_underlying_arguments(reduce_args, nullptr);
    int const max_kv_tiles = cute::ceil_div(args.max_seq_len, KVWorkUnitTokens);
    int const kv_tiles_per_split =
        cute::ceil_div(max_kv_tiles, args.num_kv_splits);
    typename ReductionSplit16Kernel::Params reduce16_params{
        out,
        reinterpret_cast<Element const*>(args.temp_output),
        args.softmax_lse_accum,
        args.seq_lens,
        kv_tiles_per_split};
    typename ReductionSplit32Kernel::Params reduce32_params{
        out,
        reinterpret_cast<Element const*>(args.temp_output),
        args.softmax_lse_accum,
        args.seq_lens,
        kv_tiles_per_split};
    typename ReductionSplitOutputHadamardKernel::Params reduce_hadamard_params{
        args.output,
        reinterpret_cast<Element const*>(args.temp_output),
        args.softmax_lse_accum,
        args.num_kv_splits,
        args.seq_lens,
        kv_tiles_per_split,
        args.write_bf16_output};
    launch(
        queue,
        params,
        reduce_params,
        reduce16_params,
        reduce32_params,
        reduce_hadamard_params,
        args.num_kv_splits,
        args.unrotate_output);
    return cutlass::Status::kSuccess;
  }

  template <class Reducer>
  static void launch_output_hadamard_reducer(
      sycl::queue& queue,
      typename ReductionSplitOutputHadamardKernel::Params const& reduce_params,
      int batch_size) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(Reducer::SharedStorageSize)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        compat::dim3(1, Policy::NumQueryHeads, batch_size),
        compat::dim3(Reducer::kThreads, 1, 1),
        launch_props,
        kernel_props};
    compat::experimental::launch<cutlass::device_kernel<Reducer>>(
        policy, queue, reduce_params);
  }

  static void launch(
      sycl::queue& queue,
      typename Kernel::Params params,
      typename ReductionSplitKernel::Params reduce_params,
      typename ReductionSplit16Kernel::Params reduce16_params,
      typename ReductionSplit32Kernel::Params reduce32_params,
      typename ReductionSplitOutputHadamardKernel::Params
          reduce_hadamard_params,
      int num_kv_splits,
      bool unrotate_output) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    dim3 const block = Kernel::get_block_shape();
    dim3 const grid = Kernel::get_grid_shape(params);
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
    // GRF128 is deliberately a producer-only occupancy experiment. Keeping
    // the reducer at its established GRF256 policy isolates register pressure
    // from every output and caller-owned scratch contract.
    compat::experimental::kernel_properties main_kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        intelex::grf_size<MainKernelGrfSize>};
    compat::experimental::kernel_properties reduce_kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z),
        compat::dim3(block.x, block.y, block.z),
        launch_props,
        main_kernel_props};
    compat::experimental::launch<cutlass::device_kernel<Kernel>>(
        policy, queue, params);

    if (unrotate_output) {
      if constexpr (SpecializedSplitReducer) {
        switch (num_kv_splits) {
          case 2:
            launch_output_hadamard_reducer<
                ReductionSplitOutputHadamardSpecializedKernel<2>>(
                queue, reduce_hadamard_params, params.kernel.shape.batch);
            break;
          case 4:
            launch_output_hadamard_reducer<
                ReductionSplitOutputHadamardSpecializedKernel<4>>(
                queue, reduce_hadamard_params, params.kernel.shape.batch);
            break;
          case 8:
            launch_output_hadamard_reducer<
                ReductionSplitOutputHadamardSpecializedKernel<8>>(
                queue, reduce_hadamard_params, params.kernel.shape.batch);
            break;
          case 16:
            launch_output_hadamard_reducer<
                ReductionSplitOutputHadamardSpecializedKernel<16>>(
                queue, reduce_hadamard_params, params.kernel.shape.batch);
            break;
          case 32:
            launch_output_hadamard_reducer<
                ReductionSplitOutputHadamardSpecializedKernel<32>>(
                queue, reduce_hadamard_params, params.kernel.shape.batch);
            break;
          default:
            // Legal non-policy counts (currently 17 and 24) retain the generic
            // fused reducer, so selecting the experiment never rejects a
            // service shape that the external ABI already accepts.
            launch_output_hadamard_reducer<ReductionSplitOutputHadamardKernel>(
                queue, reduce_hadamard_params, params.kernel.shape.batch);
            break;
        }
      } else {
        launch_output_hadamard_reducer<ReductionSplitOutputHadamardKernel>(
            queue, reduce_hadamard_params, params.kernel.shape.batch);
      }
    } else if (num_kv_splits == 32) {
      compat::experimental::launch_properties reduce32_launch_props{
          syclex::work_group_scratch_size(
              ReductionSplit32Kernel::SharedStorageSize)};
      compat::experimental::launch_policy reduce32_policy{
          compat::dim3(1, Policy::NumQueryHeads, params.kernel.shape.batch),
          compat::dim3(ReductionSplit32Kernel::kThreads, 1, 1),
          reduce32_launch_props,
          reduce_kernel_props};
      compat::experimental::launch<
          cutlass::device_kernel<ReductionSplit32Kernel>>(
          reduce32_policy, queue, reduce32_params);
    } else if (num_kv_splits == 16) {
      compat::experimental::launch_properties reduce16_launch_props{
          syclex::work_group_scratch_size(
              ReductionSplit16Kernel::SharedStorageSize)};
      compat::experimental::launch_policy reduce16_policy{
          compat::dim3(1, Policy::NumQueryHeads, params.kernel.shape.batch),
          compat::dim3(128, 1, 1),
          reduce16_launch_props,
          reduce_kernel_props};
      compat::experimental::launch<
          cutlass::device_kernel<ReductionSplit16Kernel>>(
          reduce16_policy, queue, reduce16_params);
    } else if (num_kv_splits > 1) {
      dim3 const reduce_block = ReductionSplitKernel::get_block_shape();
      dim3 const reduce_grid =
          ReductionSplitKernel::get_grid_shape(reduce_params);
      compat::experimental::launch_properties reduce_launch_props{
          syclex::work_group_scratch_size(
              ReductionSplitKernel::SharedStorageSize)};
      compat::experimental::launch_policy reduce_policy{
          compat::dim3(reduce_grid.x, reduce_grid.y, reduce_grid.z),
          compat::dim3(reduce_block.x, reduce_block.y, reduce_block.z),
          reduce_launch_props,
          reduce_kernel_props};
      compat::experimental::launch<
          cutlass::device_kernel<ReductionSplitKernel>>(
          reduce_policy, queue, reduce_params);
    }
  }
};

using KVarNDecodeD256G128Config = KVarNDecodeD256G128ConfigImpl<false>;
using KVarNDecodeD256G128DpasConfig = KVarNDecodeD256G128ConfigImpl<true>;
using KVarNDecodeD256G128DpasQKInt8U4Config =
    KVarNDecodeD256G128ConfigImpl<true, cute::_8, false, true>;
using KVarNDecodeD256G128DpasVectorLoadConfig =
    KVarNDecodeD256G128ConfigImpl<true, cute::_8, true>;
using KVarNDecodeD256G128DpasQ6Config =
    KVarNDecodeD256G128ConfigImpl<true, cute::Int<6>>;
using KVarNDecodeD256G128DpasQ6MainGrf128Config = KVarNDecodeD256G128ConfigImpl<
    true,
    cute::Int<6>,
    false,
    false,
    false,
    false,
    false,
    128>;
using KVarNDecodeD256G128DpasQ6VectorLoadConfig =
    KVarNDecodeD256G128ConfigImpl<true, cute::Int<6>, true>;
using KVarNDecodeD256G128DpasQ6CachedWeightsConfig =
    KVarNDecodeD256G128ConfigImpl<true, cute::Int<6>, false, false, true>;
using KVarNDecodeD256G128DpasQ6ExactRowsConfig = KVarNDecodeD256G128ConfigImpl<
    true,
    cute::Int<6>,
    false,
    false,
    false,
    true>;
using KVarNDecodeD256G128DpasQ6CachedWeightsExactRowsConfig =
    KVarNDecodeD256G128ConfigImpl<true, cute::Int<6>, false, false, true, true>;
using KVarNDecodeD256G128DpasQ6PagePairConfig = KVarNDecodeD256G128ConfigImpl<
    true,
    cute::Int<6>,
    false,
    false,
    false,
    false,
    true>;
using KVarNDecodeD256G128DpasQ6SplitReducerSpecializedConfig =
    KVarNDecodeD256G128ConfigImpl<
        true,
        cute::Int<6>,
        false,
        false,
        false,
        false,
        false,
        256,
        true>;
using KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig =
    KVarNDecodeD256G128ConfigImpl<
        true,
        cute::Int<6>,
        false,
        false,
        false,
        false,
        false,
        256,
        false,
        true>;

// Candidate r1-p2's scalar scatter assigns one 3x128 output subtile to each
// of the established four Reduce-K subgroups.  Prove against the actual CuTe
// fragment layout that a subgroup covers its subtile exactly once.  The shape
// assertions below then prove the 2x2 subtile grid partitions the 6x256 tile.
constexpr bool r1_p2_q6_fragment_is_bijective() {
  using Epilogue = KVarNDecodeD256G128DpasQ6Config::Epilogue;
  using Fragment = Epilogue::ReduceFragA;
  constexpr int kQ = cute::size<0>(typename Epilogue::SGTileShapeO{});
  constexpr int kV = cute::size<1>(typename Epilogue::SGTileShapeO{});
  constexpr int kSubgroupSize = cute::intel::sg_size;
  bool visited[kQ * kV]{};

  for (int lane = 0; lane < kSubgroupSize; ++lane) {
    for (int value = 0; value < Fragment{}.size(); ++value) {
      auto coord = Fragment{}.tv_layout()(lane, value);
      int const q = int(cute::get<0>(coord));
      int const v = int(cute::get<1>(coord));
      if (q < 0 || q >= kQ || v < 0 || v >= kV || visited[q * kV + v]) {
        return false;
      }
      visited[q * kV + v] = true;
    }
  }
  for (bool element : visited) {
    if (!element) return false;
  }
  return true;
}

// The old natural and q8 aliases retain their original template arguments,
// DPAS repeat, and generic block-copy epilogue.  Q6 alone opts into repeat-2
// DPAS and the scalar output adapter.
static_assert(cute::is_same_v<
              KVarNDecodeD256G128Config,
              KVarNDecodeD256G128ConfigImpl<false, cute::_8, false>>);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasConfig,
              KVarNDecodeD256G128ConfigImpl<true, cute::_8, false>>);
static_assert(KVarNDecodeD256G128DpasConfig::SGTileQ == 8);
static_assert(KVarNDecodeD256G128DpasQ6Config::SGTileQ == 6);
static_assert(KVarNDecodeD256G128DpasQ6Config::MainGrfSize == 256);
static_assert(KVarNDecodeD256G128DpasQ6MainGrf128Config::MainGrfSize == 128);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasQ6MainGrf128Config::Kernel,
              KVarNDecodeD256G128DpasQ6Config::Kernel>);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasConfig::MMAOperation,
              cute::XE_DPAS_TT<8, float, cutlass::half_t>>);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasQ6Config::MMAOperation,
              cute::XE_DPAS_TT<2, float, cutlass::half_t>>);
static_assert(!KVarNDecodeD256G128DpasConfig::Epilogue::ScalarOutput);
static_assert(KVarNDecodeD256G128DpasQ6Config::Epilogue::ScalarOutput);
static_assert(!KVarNDecodeD256G128DpasQ6Config::Epilogue::CacheScalarWeights);
static_assert(
    KVarNDecodeD256G128DpasQ6CachedWeightsConfig::Epilogue::CacheScalarWeights);
static_assert(!KVarNDecodeD256G128DpasQ6Config::Mainloop::ExactLiveRows);
static_assert(!KVarNDecodeD256G128DpasQ6Config::Mainloop::PagePair);
static_assert(
    KVarNDecodeD256G128DpasQ6ExactRowsConfig::Mainloop::ExactLiveRows);
static_assert(KVarNDecodeD256G128DpasQ6CachedWeightsExactRowsConfig::Epilogue::
                  CacheScalarWeights);
static_assert(KVarNDecodeD256G128DpasQ6CachedWeightsExactRowsConfig::Mainloop::
                  ExactLiveRows);
static_assert(KVarNDecodeD256G128DpasQ6PagePairConfig::Mainloop::PagePair);
static_assert(KVarNDecodeD256G128DpasQ6PagePairConfig::KVWorkUnitTokens == 128);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasQ6PagePairConfig::MMAOperation,
              KVarNDecodeD256G128DpasQ6Config::MMAOperation>);
static_assert(!KVarNDecodeD256G128DpasQ6Config::UsesSpecializedSplitReducer);
static_assert(KVarNDecodeD256G128DpasQ6SplitReducerSpecializedConfig::
                  UsesSpecializedSplitReducer);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasQ6SplitReducerSpecializedConfig::Mainloop,
              KVarNDecodeD256G128DpasQ6Config::Mainloop>);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasQ6SplitReducerSpecializedConfig::Kernel,
              KVarNDecodeD256G128DpasQ6Config::Kernel>);
static_assert(cute::is_same_v<
              KVarNReduceSplitOutputHadamardSpecializedKernel<8, 64>::Params,
              KVarNReduceSplitOutputHadamardKernel<64>::Params>);
static_assert(
    KVarNReduceSplitOutputHadamardSpecializedKernel<8, 64>::SharedStorageSize <
    KVarNReduceSplitOutputHadamardKernel<64>::SharedStorageSize);
static_assert(
    KVarNReduceSplitOutputHadamardSpecializedKernel<32, 64>::
        SharedStorageSize ==
    KVarNReduceSplitOutputHadamardKernel<64>::SharedStorageSize);
static_assert(!KVarNDecodeD256G128DpasQ6Config::Mainloop::NextPagePrefetch);
static_assert(KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::Mainloop::
                  NextPagePrefetch);
static_assert(
    !KVarNDecodeD256G128DpasQ6PagePairConfig::Mainloop::NextPagePrefetch);
static_assert(
    !KVarNDecodeD256G128DpasQ6MainGrf128Config::Mainloop::NextPagePrefetch);
static_assert(!KVarNDecodeD256G128DpasQ6SplitReducerSpecializedConfig::
                  Mainloop::NextPagePrefetch);
static_assert(
    !KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::Mainloop::PagePair);
static_assert(
    KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::MainGrfSize == 256);
static_assert(!KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::
                  UsesSpecializedSplitReducer);
static_assert(
    KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::KVWorkUnitTokens == 64);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::TiledMMAQK,
              KVarNDecodeD256G128DpasQ6Config::TiledMMAQK>);
static_assert(cute::is_same_v<
              KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::TiledMMAPV,
              KVarNDecodeD256G128DpasQ6Config::TiledMMAPV>);
static_assert(
    sizeof(KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::Mainloop::Params) ==
    sizeof(KVarNDecodeD256G128DpasQ6Config::Mainloop::Params));
static_assert(
    sizeof(
        KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::Mainloop::Arguments) ==
    sizeof(KVarNDecodeD256G128DpasQ6Config::Mainloop::Arguments));
static_assert(
    sizeof(KVarNDecodeD256G128DpasQ6NextPagePrefetchConfig::Mainloop::
               SharedStorage) ==
    sizeof(KVarNDecodeD256G128DpasQ6Config::Mainloop::SharedStorage));
static_assert(
    cute::size<0>(KVarNDecodeD256G128DpasQ6Config::Epilogue::TileShapeO{}) ==
    6);
static_assert(
    cute::size<1>(KVarNDecodeD256G128DpasQ6Config::Epilogue::TileShapeO{}) ==
    256);
static_assert(
    cute::size<0>(KVarNDecodeD256G128DpasQ6Config::Epilogue::SGTileShapeO{}) ==
    3);
static_assert(
    cute::size<1>(KVarNDecodeD256G128DpasQ6Config::Epilogue::SGTileShapeO{}) ==
    128);
static_assert(
    cute::size<0>(cute::shape(
        KVarNDecodeD256G128DpasQ6Config::Epilogue::ReduceSGLayout{})) == 2);
static_assert(
    cute::size<1>(cute::shape(
        KVarNDecodeD256G128DpasQ6Config::Epilogue::ReduceSGLayout{})) == 2);
static_assert(r1_p2_q6_fragment_is_bijective());
