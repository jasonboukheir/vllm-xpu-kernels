#pragma once

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "collective/chunk_prefill_epilogue.hpp"
#include "collective/chunk_prefill_scheduler.hpp"
#include "collective/kvarn_decode_mainloop.hpp"
#include "kernel/chunk_prefill_kernel.hpp"
#include "fmha_utils.hpp"

/** Narrow native compact cached-prefill contract for Qwen3.6 D256/G128.
 *
 * Q and O are packed varlen tensors [total_q, 24, 256]. `cu_seqlens_q` is
 * cumulative [B+1], while `seq_lens` contains each request's total KV length
 * after appending its current query chunk. K/V are reconstructed directly
 * from the compact records or the full-precision tail pool.
 */
struct kvarn_chunk_prefill_args_t {
  void const* query;
  std::uint8_t const* cache;
  void* output;
  int const* block_table;
  int const* seq_lens;
  int const* cu_seqlens_q;
  int const* block_to_slot;
  void const* tail_key;
  void const* tail_value;
  int batch_size;
  int total_query_tokens;
  int max_query_len;
  int max_seq_len;
  int max_pages_per_seq;
  float softmax_scale;
  cutlass::fmha::collective::KVarNK4V4Layout layout;
};

template <bool DpasPacked = false>
struct KVarNChunkPrefillD256G128ConfigImpl {
  using Element = cutlass::half_t;
  using TileShapeQK = cute::Shape<cute::_256, cute::_64, cute::_64>;
  using TileShapePV = cute::Shape<cute::_256, cute::_32, cute::_64>;
  using TileShapeO = cute::Shape<cute::_256, cute::_256>;
  // Thirty-two subgroups partition the 256 query rows. ReduceK remains one:
  // every subgroup reuses the same compact 64-token K/V fragment for eight Q
  // rows instead of materializing the prefix or rereading it per query row.
  using SubgroupLayoutQK = cute::Layout<cute::Shape<cute::_32, cute::_1, cute::_1>>;
  using SubgroupLayoutPV = decltype(
      cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using MMAOperation = cute::XE_DPAS_TT<8, float, Element>;
  using TiledMMAQK = typename TiledMMAHelper<
      cute::MMA_Atom<MMAOperation>,
      cute::Layout<TileShapeQK>,
      SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<
      cute::MMA_Atom<MMAOperation>,
      cute::Layout<TileShapePV>,
      SubgroupLayoutPV>::TiledMMA;

  static constexpr int VTiles = 8;
  using StrideQ = cute::Stride<int, cute::_1, int, int>;
  using StrideK = cute::Stride<int, cute::_1, int, int>;
  using StrideV = cute::Stride<cute::_1, int, int, int>;
  using StrideO = cute::Stride<int, cute::_1, int, int>;

  template <class T, class Stride>
  using DummyTensor = decltype(cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T*>(nullptr)),
      cute::make_layout(cute::repeat<cute::rank_v<Stride>>(1), Stride{})));

  using TensorQ = DummyTensor<Element, StrideQ>;
  using TensorK = DummyTensor<Element, StrideK>;
  using TensorV = DummyTensor<Element, StrideV>;
  using TensorO = DummyTensor<Element, StrideO>;
  using Mainloop = cutlass::fmha::collective::KVarNDecodeFwdMainloop<
      TiledMMAQK,
      TiledMMAPV,
      VTiles,
      TensorQ,
      TensorK,
      TensorV,
      DpasPacked,
      true,
      DpasPacked>;
  using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
      false,
      Mainloop,
      TileShapeO,
      TensorO>;
  using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler;
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
      ProblemShape,
      Mainloop,
      Epilogue,
      Scheduler,
      false>;

  static cutlass::Status
  run(sycl::queue& queue, kvarn_chunk_prefill_args_t const& args) {
    if (args.query == nullptr || args.cache == nullptr ||
        args.output == nullptr || args.block_table == nullptr ||
        args.seq_lens == nullptr || args.cu_seqlens_q == nullptr ||
        args.block_to_slot == nullptr || args.tail_key == nullptr ||
        args.tail_value == nullptr || args.batch_size < 1 ||
        args.total_query_tokens < 1 || args.max_query_len < 1 ||
        args.max_seq_len < args.max_query_len ||
        args.max_pages_per_seq < 1) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    ProblemShape shape;
    shape.batch = args.batch_size;
    shape.num_heads_q = 24;
    shape.num_heads_kv = 4;
    shape.seq_len_qo =
        cutlass::fmha::collective::VariableLength{args.max_query_len};
    shape.seq_len_qo.cumulative_length =
        const_cast<int*>(args.cu_seqlens_q);
    shape.seq_len_kv =
        cutlass::fmha::collective::VariableLength{args.max_seq_len};
    // Paged XeFMHAFwdKernel interprets this pointer as per-request used KV
    // lengths, rather than a cumulative tensor.
    shape.seq_len_kv.cumulative_length = const_cast<int*>(args.seq_lens);
    shape.head_size_qk = 256;
    shape.head_size_vo = 256;

    StrideQ stride_q{24 * 256, cute::_1{}, 256, 0};
    StrideO stride_o{24 * 256, cute::_1{}, 256, 0};
    // Contract-only layouts: the compact mainloop never dereferences K/V.
    StrideK stride_k{256, cute::_1{}, 256, 0};
    StrideV stride_v{cute::_1{}, 256, 256, 0};

    auto const* q = reinterpret_cast<Element const*>(args.query);
    auto* out = reinterpret_cast<Element*>(args.output);
    auto kvarn_layout = args.layout;
    kvarn_layout.cache = args.cache;
    cutlass::fmha::collective::KVarNHybridTailLayout tail_layout{
        args.block_to_slot,
        reinterpret_cast<Element const*>(args.tail_key),
        reinterpret_cast<Element const*>(args.tail_value),
        128 * 4 * 256,
        4 * 256,
        256};

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
         out,
         stride_o,
         nullptr,
         nullptr,
         0,
         nullptr},
        {{args.softmax_scale,
          nullptr,
          nullptr,
          const_cast<int*>(args.block_table),
          128,
          args.max_pages_per_seq,
          args.max_seq_len,
          -1,
          -1,
          0},
         kvarn_layout,
         tail_layout,
         args.seq_lens},
        {},
        hw_info};

    if (!Kernel::can_implement(kernel_args)) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    auto params = Kernel::to_underlying_arguments(kernel_args, nullptr);
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    auto block = Kernel::get_block_shape();
    auto grid = Kernel::get_grid_shape(params);
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
    return cutlass::Status::kSuccess;
  }
};

using KVarNChunkPrefillD256G128Config =
    KVarNChunkPrefillD256G128ConfigImpl<false>;
using KVarNChunkPrefillD256G128DpasConfig =
    KVarNChunkPrefillD256G128ConfigImpl<true>;

static_assert(KVarNChunkPrefillD256G128Config::Mainloop::CausalMask);
static_assert(
    cute::size<0>(KVarNChunkPrefillD256G128Config::TileShapeQK{}) == 256);
static_assert(
    cute::size<1>(KVarNChunkPrefillD256G128Config::TileShapeQK{}) == 64);
