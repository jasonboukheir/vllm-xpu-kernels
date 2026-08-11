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
  void* output;              // [B,24,256] fp16, Hadamard-rotated
  void* temp_output;         // [B, 24 * splits, 256] fp16 partials
  float* exp_sums;           // [B, 24, splits] scratch
  float* max_logits;         // [B, 24, splits] scratch
  int const* block_table;    // [B, max_pages_per_seq]
  int const* seq_lens;       // [B], actual length for each batch row
  int const* block_to_slot;  // [num_physical_blocks], -1 selects KVarN
  void const* tail_key;      // [slots, 128, 4, 256] fp16, rotated
  void const* tail_value;    // [slots, 128, 4, 256] fp16, rotated
  int batch_size;            // B1-B4 decode or B4 x qlen3 MTP virtual rows
  int max_seq_len;           // maximum length used by the scheduler
  int max_pages_per_seq;
  int num_kv_splits;
  float softmax_scale;
  cutlass::fmha::collective::KVarNK4V4Layout layout;
};

/** Compile-time policy contract for the first KVarN kernel.
 *
 * Page 128 intentionally uses the existing tile-64 policy.  The repository's
 * tile-128/ReduceK=8 path has a known cross-subgroup reduction correctness
 * issue, while tile 64 visits each KVarN page in exactly two iterations.
 */
struct KVarNDecodeD256G128Policy {
  using BasePolicy =
      decode_policy_qpacked_head<cute::_8, cute::_256, cute::_64>;
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

static_assert(cute::size<1>(KVarNDecodeD256G128Policy::ShapeQK{}) == 64);
static_assert(cute::size<1>(KVarNDecodeD256G128Policy::ShapeOut{}) == 256);

/** KVarN-only reduction for the serving-optimal fixed 16-way KV split.
 *
 * Keeping this kernel here is intentional: putting the specialization in the
 * generic paged-decode reducer forces every attention configuration to be
 * re-instantiated during each KVarN tuning iteration.
 */
struct KVarNReduceSplit16Kernel {
  using Element = cutlass::half_t;
  static_assert(cute::intel::sg_size == 16);

  struct Params {
    Element* output;
    Element const* partial_output;
    float const* exp_sums;
    float const* max_logits;
    int const* seq_lens;
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

    int const kv_blocks = cute::ceil_div(
        params.seq_lens[batch], KVarNDecodeD256G128Policy::KVTile);
    int const blocks_per_split = cute::ceil_div(kv_blocks, kSplits);
    bool valid = lane * blocks_per_split < kv_blocks;
    int const stats_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) * kSplits +
        lane;

    float local_max = cutlass::platform::numeric_limits<float>::lowest();
    float local_exp_sum = 0.0f;
    if (valid) {
      local_max = params.max_logits[stats_offset];
      local_exp_sum = params.exp_sums[stats_offset];
      valid = local_exp_sum > 0.0f;
      if (!valid) {
        local_max = cutlass::platform::numeric_limits<float>::lowest();
        local_exp_sum = 0.0f;
      }
    }

    float const global_max =
        sycl::reduce_over_group(subgroup, local_max, sycl::maximum<>());
    float const rescale =
        valid ? sycl::native::exp2(local_max - global_max) : 0.0f;
    float const denominator = sycl::reduce_over_group(
        subgroup, local_exp_sum * rescale, sycl::plus<>());
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
                      rescale
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
struct KVarNReduceSplit32Kernel {
  using Element = cutlass::half_t;

  struct Params {
    Element* output;
    Element const* partial_output;
    float const* exp_sums;
    float const* max_logits;
    int const* seq_lens;
  };

  static constexpr int kSplits = 32;
  static constexpr int kThreads = KVarNDecodeD256G128Policy::HeadDim;
  static constexpr int SharedStorageSize = 2 * kSplits * sizeof(float);

  CUTLASS_DEVICE
  void operator()(Params const& params, char* shared_storage) {
    using namespace sycl::ext::oneapi::this_work_item;

    int const head = int(BlockIdxY());
    int const batch = int(BlockIdxZ());
    int const thread = int(ThreadIdxX());
    auto workgroup = get_work_group<3>();
    auto* split_weights = reinterpret_cast<float*>(shared_storage);
    auto* split_exp_sums = split_weights + kSplits;

    int const kv_blocks = cute::ceil_div(
        params.seq_lens[batch], KVarNDecodeD256G128Policy::KVTile);
    int const blocks_per_split = cute::ceil_div(kv_blocks, kSplits);
    bool valid = thread < kSplits && thread * blocks_per_split < kv_blocks;
    int const stats_offset =
        (batch * KVarNDecodeD256G128Policy::NumQueryHeads + head) * kSplits +
        thread;
    float local_max = valid
                          ? params.max_logits[stats_offset]
                          : cutlass::platform::numeric_limits<float>::lowest();
    float local_exp_sum = valid ? params.exp_sums[stats_offset] : 0.0f;

    float const global_max =
        sycl::reduce_over_group(workgroup, local_max, sycl::maximum<>());
    if (thread < kSplits) {
      split_weights[thread] =
          valid ? sycl::native::exp2(local_max - global_max) : 0.0f;
      split_exp_sums[thread] = local_exp_sum;
    }
    sycl::group_barrier(workgroup);

    if (thread < KVarNDecodeD256G128Policy::HeadDim) {
      float numerator = 0.0f;
      float denominator = 0.0f;
      for (int split = 0; split < kSplits; ++split) {
        float const exp_sum = split_exp_sums[split];
        if (exp_sum <= 0.0f) continue;
        int const partial_offset =
            ((batch * kSplits + split) *
                 KVarNDecodeD256G128Policy::NumQueryHeads +
             head) *
                KVarNDecodeD256G128Policy::HeadDim +
            thread;
        numerator += static_cast<float>(params.partial_output[partial_offset]) *
                     split_weights[split];
        denominator += exp_sum * split_weights[split];
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

/** Concrete, intentionally narrow native decode configuration.
 *
 * This is kept separate from PagedDecodeConfig because K and V are not
 * tensors in memory: KVarNDecodeFwdMainloop materializes their MMA fragments
 * directly from the packed record.  The dummy K/V tensor types only satisfy
 * the surrounding XeFMHA kernel's type contract and are never dereferenced by
 * the custom mainloop.
 */
template <bool DpasPacked = false>
struct KVarNDecodeD256G128ConfigImpl {
  using Policy = KVarNDecodeD256G128Policy;
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
      TiledMMAPV,
      VTiles,
      TensorQ,
      TensorK,
      TensorV,
      DpasPacked>;
  using Epilogue = cutlass::fmha::collective::
      DecodeFwdEpilogue<Mainloop, TileShapeO, TensorO, TensorLSE, void, false>;
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

  static cutlass::Status
  run(sycl::queue& queue, kvarn_decode_args_t const& args) {
    if (args.batch_size < 1 || args.batch_size > 12 || args.max_seq_len <= 0 ||
        args.max_pages_per_seq <= 0 || args.query == nullptr ||
        args.output == nullptr || args.temp_output == nullptr ||
        args.exp_sums == nullptr || args.max_logits == nullptr ||
        args.num_kv_splits < 1 || args.block_table == nullptr ||
        args.seq_lens == nullptr || args.cache == nullptr ||
        args.block_to_slot == nullptr || args.tail_key == nullptr ||
        args.tail_value == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    ProblemShape shape{
        args.batch_size,
        Policy::NumQueryHeads,
        Policy::NumKVHeads,
        1,
        args.max_seq_len,
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
         args.exp_sums,
         stride_lse,
         args.max_logits,
         stride_lse,
         nullptr,
         nullptr,
         nullptr},
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
         args.exp_sums,
         stride_lse,
         args.max_logits,
         stride_lse,
         -1,
         nullptr,
         nullptr},
        hw_info,
        args.num_kv_splits};

    if (!Kernel::can_implement(kernel_args)) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    auto params = Kernel::to_underlying_arguments(kernel_args, nullptr);
    auto reduce_params =
        ReductionSplitKernel::to_underlying_arguments(reduce_args, nullptr);
    KVarNReduceSplit16Kernel::Params reduce16_params{
        out,
        reinterpret_cast<Element const*>(args.temp_output),
        args.exp_sums,
        args.max_logits,
        args.seq_lens};
    KVarNReduceSplit32Kernel::Params reduce32_params{
        out,
        reinterpret_cast<Element const*>(args.temp_output),
        args.exp_sums,
        args.max_logits,
        args.seq_lens};
    launch(
        queue,
        params,
        reduce_params,
        reduce16_params,
        reduce32_params,
        args.num_kv_splits);
    return cutlass::Status::kSuccess;
  }

  static void launch(
      sycl::queue& queue,
      typename Kernel::Params params,
      typename ReductionSplitKernel::Params reduce_params,
      KVarNReduceSplit16Kernel::Params reduce16_params,
      KVarNReduceSplit32Kernel::Params reduce32_params,
      int num_kv_splits) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    dim3 const block = Kernel::get_block_shape();
    dim3 const grid = Kernel::get_grid_shape(params);
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z),
        compat::dim3(block.x, block.y, block.z),
        launch_props,
        kernel_props};
    compat::experimental::launch<cutlass::device_kernel<Kernel>>(
        policy, queue, params);

    if (num_kv_splits == 32) {
      compat::experimental::launch_properties reduce32_launch_props{
          syclex::work_group_scratch_size(
              KVarNReduceSplit32Kernel::SharedStorageSize)};
      compat::experimental::launch_policy reduce32_policy{
          compat::dim3(1, Policy::NumQueryHeads, params.kernel.shape.batch),
          compat::dim3(KVarNReduceSplit32Kernel::kThreads, 1, 1),
          reduce32_launch_props,
          kernel_props};
      compat::experimental::launch<
          cutlass::device_kernel<KVarNReduceSplit32Kernel>>(
          reduce32_policy, queue, reduce32_params);
    } else if (num_kv_splits == 16) {
      compat::experimental::launch_properties reduce16_launch_props{
          syclex::work_group_scratch_size(
              KVarNReduceSplit16Kernel::SharedStorageSize)};
      compat::experimental::launch_policy reduce16_policy{
          compat::dim3(1, Policy::NumQueryHeads, params.kernel.shape.batch),
          compat::dim3(128, 1, 1),
          reduce16_launch_props,
          kernel_props};
      compat::experimental::launch<
          cutlass::device_kernel<KVarNReduceSplit16Kernel>>(
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
          kernel_props};
      compat::experimental::launch<
          cutlass::device_kernel<ReductionSplitKernel>>(
          reduce_policy, queue, reduce_params);
    }
  }
};

using KVarNDecodeD256G128Config = KVarNDecodeD256G128ConfigImpl<false>;
using KVarNDecodeD256G128DpasConfig = KVarNDecodeD256G128ConfigImpl<true>;
