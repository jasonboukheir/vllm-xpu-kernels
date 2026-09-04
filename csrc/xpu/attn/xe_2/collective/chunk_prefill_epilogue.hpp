/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"

#include "flash_attention_v2/collective/copy_block_slm.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template <
    bool Sink_,
    class CollectiveMainloop,  // Attention mainloop
    class TileShapeO_,         // Shape of output tile, may be larger than P*V
                               // GEMM
    class TensorO_,            // 2D slice of global output tensor
    class TiledCopyO_ = void>  // Optional TiledCopy for loading O
class FMHAFwdEpilogue {
 public:
  //
  // Type Aliases
  //
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D =
      decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

  // softmax sink, same dtype
  static constexpr bool Sink = Sink_;
  using ElementSink = typename CollectiveMainloop::TensorQ::element_type;

  // Split k-reduced tiles between participating subgroups.
  // Assumption: the A tile is contiguous.
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<
        (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg)
                                  : v_total_sg>{};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout =
      decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

  using SGTileShapeO =
      decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
      make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(
          TiledMMAPV{},
          ReduceFragA{}.tv_layout(),
          ReduceSGLayout{},
          TensorO2D{});
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO =
      conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  // Stateless design -- no arguments or parameters.
  struct Arguments {};
  struct Params {};

  // Shared memory storage
  // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe
  // block load infrastructure.
  using AlignedSGTileA_Q =
      C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) *
        intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data,
        a_max_data;
  };

  using SharedStorage = conditional_t<
      (ReduceK{} > _1{}),
      SharedStorageReduceK,
      SharedStorageNone>;

 private:
  SharedStorage& shared;

 public:
  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FMHAFwdEpilogue(Params const&, SharedStorage& shared_) : shared(shared_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,        // Global O tensor: (q,v)
      FragA& tArA,               // O accumulator:   (q,v)
      FragARow& tA_max,          // Softmax row-wise max accumulator
      FragARow& tA_sum,          // Softmax row-wise sum accumulator
      QVCoord blk_qv,            // WG tile indices: (q,v)
      ElementSink const& tSink,  // Sink for current head
      int thr_id) {              // Work-item ID

    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    /* Complete softmax, dividing out sums. */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++) {
      if constexpr (Sink) {
        constexpr double kLog2e = 1.4426950408889634074;
        rA_sum(i) += sycl::native::exp2(
            static_cast<ElementA>(tSink * kLog2e) - tA_max(i));
      }
      rA_sum(i) = ElementA(1) / rA_sum(i);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape());       // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  // Note that each k block has its own scale factor based on A_max,
  //   so A/A_sum contributions need to be rescaled to match.
  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA& tArA,       // O accumulator:   (q,v)
      FragARow& tA_max,  // Softmax row-wise max accumulator
      FragARow& tA_sum,  // Softmax row-wise sum accumulator
      int thr_id) {      // Work-item ID

    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, true);
    } else {
      /* Identify A tile ID and k block for this subgroup. */
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      /* Set up SLM tensors and partition A tiles among participating subgroups
       */
      auto shape_A =
          append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
      auto shape_A_row = make_shape(
          get<0>(SGTileShapeO{}),
          shape(ReduceSGLayout{}),
          ReduceK{},
          SGPerWG{} / ReduceK{});

      /* Physical layouts, with subtile modes broken out */
      auto sA_layout = group<2, 4>(flat_divide(
          make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}),
          SGTileShapeO{}));
      auto sA_row_stride = make_stride(
          _1{},
          make_stride(get<0>(shape_A_row), _0{}),
          AlignedSGTileA_Q{},
          AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      /* Coordinate layouts, with subtile modes broken out */
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(
          append(SGTileShapeO{}, shape(ReduceSGLayout{})),
          append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_data),
          sA_layout);  // (q,v,rblk_dst,rblk_src,a_tile)
      auto sA_max = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_max_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
      auto sA_sum = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_sum_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) ||
                    (ReduceK{} == size(ReduceSGLayout{}));  // help compiler out

      /* Wait for maxima to be available, signal other data available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        /* Read A_max back from SLM and reduce. */
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); ++i)
            rA_max(i) =
                (rA_max(i) < rA_kmax[kr](i)) ? rA_kmax[kr](i) : rA_max(i);
        }

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); ++i)
            rA_kmax[kr](i) = sycl::native::exp2(rA_kmax[kr](i) - rA_max(i));
        }
      }

      /* Wait for A/A_sum data to be available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce.
         */
        clear(rA_sum);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragARow rA_sum_read;
          copy_block_s2r(sA_sum(_, k_blk, kr, a_tile), rA_sum_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_sum_read.size(); i++) {
            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
          }
        }

        clear(rA);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragA rA_read;
          copy_block_s2r(
              sA(_, _, k_blk, kr, a_tile), sA_coords(_, _, 0), rA_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_read.size(); i++) {
            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
          }
        }
      }
      return std::make_tuple(rA, rA_sum, active);
    }
  }
};

template <
    class CollectiveMainloop,  // Attention mainloop
    class TileShapeO_,  // Shape of output tile, may be larger than P*V GEMM
    class TensorO_,     // 2D slice of global output tensor
    class TensorLSE_ = void,     // Optional intermediate natural-LSE tensor
    class TiledCopyO_ = void,    // Optional TiledCopy for loading O
    bool Sink_ = false,          // Whether to sink softmax into epilogue
    bool ScalarOutput_ = false>  // Whether to use coordinate scalar stores
class DecodeFwdEpilogue {
 public:
  //
  // Type Aliases
  //
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D =
      decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  using TensorLSE = TensorLSE_;
  using TensorLSE2D = conditional_t<
      is_void_v<TensorLSE_>,
      void,
      decltype(TensorLSE_{}(append<rank_v<TensorLSE_>>(make_coord(_, _), 0)))>;
  using ElementLSE = conditional_t<
      is_void_v<TensorLSE_>,
      void,
      typename TensorLSE_::value_type>;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

  // softmax sink, same dtype
  static constexpr bool Sink = Sink_;
  using ElementSink = typename CollectiveMainloop::TensorQ::element_type;

  // Split k-reduced tiles between participating subgroups.
  // Assumption: the A tile is contiguous.
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<
        (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg)
                                  : v_total_sg>{};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout =
      decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

  using SGTileShapeO =
      decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
      make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  // Xe block stores require a power-of-two-compatible Q extent. KVarN's
  // exact GQA-6 experiment deliberately uses a six-row DPAS tile, whose
  // cross-subgroup reduction produces 3x128 output subtiles. Keep the
  // existing block-store path identical for established policies and use a
  // narrow coordinate scatter only for that Q6 tile.
  static constexpr bool ScalarOutput = ScalarOutput_;
  struct ScalarCopyO {};

  static auto default_tiled_copy_O_helper() {
    if constexpr (ScalarOutput)
      return ScalarCopyO{};
    else if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(
          TiledMMAPV{},
          ReduceFragA{}.tv_layout(),
          ReduceSGLayout{},
          TensorO2D{});
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO =
      conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  template <typename OutputFragment, typename QVCoord>
  CUTLASS_DEVICE static void store_scalar_output(
      TensorO2D const& O,
      OutputFragment const& rA,
      QVCoord blk_qv,
      int thr_id) {
    auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                       .get_flat_coord(assert_uniform(thr_id));
    int q_subtile;
    int v_subtile;
    if constexpr (ReduceK{} == _1{}) {
      // group<1,3> flattens the MMA M/N subgroup coordinate as a_tile.
      // Q6 uses three M subgroups and one N subgroup, so a_tile is exactly
      // the two-row Q-subtile index.
      q_subtile = int(get<1>(thr_vak));
      v_subtile = 0;
    } else {
      auto reduction_subtile = get<2>(thr_vak);
      auto subtile_coord = idx2crd(reduction_subtile, shape(ReduceSGLayout{}));
      q_subtile = int(get<0>(subtile_coord));
      v_subtile = int(get<1>(subtile_coord));
    }
    int const lane =
        sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];
    int const q_base = int(get<0>(blk_qv)) * size<0>(TileShapeO{}) +
                       q_subtile * size<0>(SGTileShapeO{});
    int const v_base = int(get<1>(blk_qv)) * size<1>(TileShapeO{}) +
                       v_subtile * size<1>(SGTileShapeO{});

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); ++i) {
      auto coord = rA.tv_layout()(lane, i);
      int const q = q_base + int(get<0>(coord));
      int const v = v_base + int(get<1>(coord));
      if (q < size<0>(O.shape()) && v < size<1>(O.shape())) {
        O(q, v) = static_cast<ElementO>(rA(i));
      }
    }
  }

  // Stateless design -- no arguments or parameters.
  struct Arguments {};
  struct Params {};

  // Shared memory storage
  // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe
  // block load infrastructure.
  using AlignedSGTileA_Q =
      C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) *
        intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data,
        a_max_data;
  };

  using SharedStorage = conditional_t<
      (ReduceK{} > _1{}),
      SharedStorageReduceK,
      SharedStorageNone>;

 private:
  SharedStorage& shared;

 public:
  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  DecodeFwdEpilogue(Params const&, SharedStorage& shared_) : shared(shared_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,  // Global O tensor: (q,v)
      FragA& tArA,         // O accumulator:   (q,v)
      FragARow& tA_max,    // Softmax row-wise max accumulator
      FragARow& tA_sum,    // Softmax row-wise sum accumulator
      QVCoord blk_qv,      // WG tile indices: (q,v)
      int thr_id) {        // Work-item ID

    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_max_unused, rA_sum, active] =
        reduce_A(tArA, tA_max, tA_sum, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    /* Complete softmax, dividing out sums. */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      rA_sum(i) = ElementA(1) / rA_sum(i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape());       // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // splitK version
  template <typename QVCoord, class TensorSink>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,                    // Global O tensor: (q,v)
      FragA& tArA,                           // O accumulator:   (q,v)
      FragARow& tA_max,                      // Softmax row-wise max accumulator
      FragARow& tA_sum,                      // Softmax row-wise sum accumulator
      QVCoord blk_qv,                        // WG tile indices: (q,v)
      int thr_id,                            // Work-item ID
      const TensorLSE2D& softmax_lse_accum,  // Per-split natural-log LSE
      int idx_kv_split,
      int head_group_q,
      TensorSink& tSink,  // Sink for current head
      int num_kv_splits,
      ElementLSE* ptr_lse = nullptr,  // softmax_lse row base (null = disabled)
      int lse_stride = 0) {           // stride between packed GQA rows
    using namespace cute;
    using ElementA = typename FragA::element_type;
    constexpr float kLn2 = 0.6931471805599453f;

    auto [rA, rA_max, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    // Decode tiles head_group_q across the grid's Q dimension (blk_qv[0]).
    // The established epilogue assigns one reporting lane to every complete
    // Q row directly from thr_id. ScalarOutput redistributes Q6 as a 2x2 set
    // of 3x128 output subtiles, so its reporting row must instead follow the
    // active subgroup's output subtile. Only the first V subtile reports LSE,
    // avoiding same-address writes from both halves of the output tile.
    constexpr int q_tile_rows = cute::size<0>(TileShapeO{});
    int q_row;
    bool row_valid;
    if constexpr (ScalarOutput) {
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(thr_id));
      int const source_k = int(get<2>(thr_vak));
      auto output_subtile = idx2crd(source_k, shape(ReduceSGLayout{}));
      int const q_subtile = int(get<0>(output_subtile));
      int const v_subtile = int(get<1>(output_subtile));
      int const lane =
          sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];
      constexpr int q_rows_per_subtile = cute::size<0>(SGTileShapeO{});
      int const q_local = int(rA_max.tv_layout()(lane).value());
      bool first_lane_for_row = true;
      CUTLASS_PRAGMA_UNROLL
      for (int peer = 0; peer < cute::intel::sg_size; ++peer) {
        if (peer < lane && int(rA_max.tv_layout()(peer).value()) == q_local) {
          first_lane_for_row = false;
        }
      }
      q_row = get<0>(blk_qv) * q_tile_rows + q_subtile * q_rows_per_subtile +
              q_local;
      row_valid = active && v_subtile == 0 && first_lane_for_row &&
                  q_local < q_rows_per_subtile && q_row < head_group_q;
    } else {
      q_row = get<0>(blk_qv) * q_tile_rows + thr_id;
      row_valid = (thr_id < q_tile_rows) && (q_row < head_group_q);
    }

    // Softmax denominator for the row this work-item reports statistics for
    // (q_row). The Sink block below folds the sink into rA_sum(0) using the
    // *output tile* row mapping, which is what the O normalization needs but
    // does not agree with the q_row mapping used by the per-split/final LSE
    // writes. Snapshot the pre-sink denominator here and re-apply the sink for
    // q_row explicitly so the reported statistics stay correct.
    ElementA stats_sum = rA_sum(0);

    if constexpr (Sink) {
      static_assert(!ScalarOutput, "Q6 KVarN decode does not use softmax sink");
      Tensor cO = make_identity_tensor(O.shape());       // (q,v)
      Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)
      TiledCopyO copy_o{O};
      auto thr_copy_o = copy_o.get_slice(thr_id);
      auto tOgO = thr_copy_o.partition_D(gO);  // fragment coords (q,v)
      constexpr double kLog2e = 1.4426950408889634074;
      if (row_valid && idx_kv_split == 0) {
        stats_sum += sycl::native::exp2(
            static_cast<ElementA>(tSink(q_row) * kLog2e) - rA_max(0));
      }
      if (active && idx_kv_split == 0) {
        int base_row = cute::get<0>(tOgO(cute::_0{}, cute::_0{}, cute::_0{}));
        int lane =
            static_cast<int>(sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_id()[0]);
        int row_i = base_row + (lane % cute::size<0>(SGTileShapeO{}));
        if (row_i < head_group_q) {
          rA_sum(0) += sycl::native::exp2(
              static_cast<ElementA>(tSink(row_i) * kLog2e) - rA_max(0));
        }
      }
    }

    // Store one natural-log LSE for this KV split. ReduceSplitK uses it exactly
    // like CUDA FlashAttention: weight_i = exp(LSE_i - logsumexp(LSE)).
    // Assume seq_len_qo == 1.
    if (row_valid && (num_kv_splits > 1 || ptr_lse != nullptr)) {
      ElementLSE row_lse =
          stats_sum > ElementA(0)
              ? static_cast<ElementLSE>(
                    static_cast<float>(rA_max(0)) * kLn2 +
                    sycl::log(static_cast<float>(stats_sum)))
              : cutlass::platform::numeric_limits<ElementLSE>::lowest();
      if (num_kv_splits > 1) {
        softmax_lse_accum(q_row, idx_kv_split) = row_lse;
      }

      // Without a ReduceSplitK pass this work-item holds the complete
      // softmax statistics for its query row, so write softmax_lse here.
      if (ptr_lse != nullptr) {
        ptr_lse[q_row * lse_stride] = row_lse;
      }
    }

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    // Normalize every split independently. ReduceSplitK combines normalized
    // partial outputs with exp(LSE_i - LSE_global), matching CUDA
    // FlashAttention's split-K contract.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++) {
      rA_sum(i) =
          rA_sum(i) > ElementA(0) ? ElementA(1) / rA_sum(i) : ElementA(0);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      rA(i) *= broadcast<0>(rA_sum, rA, i);
    }

    if constexpr (ScalarOutput) {
      store_scalar_output(O, rA, blk_qv, thr_id);
    } else {
      Tensor cO = make_identity_tensor(O.shape());       // (q,v)
      Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)
      TiledCopyO copy_o{O};
      auto thr_copy_o = copy_o.get_slice(thr_id);
      auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
      auto tOgO = thr_copy_o.partition_D(gO);

      /* Reorder tile and write out */
      reorder(rA, tOrO);
      copy(copy_o, tOrO, tOgO);
    }
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  // Note that each k block has its own scale factor based on A_max,
  //   so A/A_sum contributions need to be rescaled to match.
  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA& tArA,       // O accumulator:   (q,v)
      FragARow& tA_max,  // Softmax row-wise max accumulator
      FragARow& tA_sum,  // Softmax row-wise sum accumulator
      int thr_id) {      // Work-item ID

    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_max, tA_sum, true);
    } else if constexpr (ScalarOutput) {
      // Q6 retains KVarN's established four-subgroup K64 decomposition. The
      // generic block-copy reduction requires power-of-two-compatible row
      // subtiles, while Q6 redistributes those subgroups as 2x2 output
      // subtiles of 3x128. Scatter the same source fragments to the existing
      // SLM storage and perform the same max/rescale/sum reduction explicitly.
      constexpr int kQ = size<0>(TileShapeO{});
      constexpr int kV = size<1>(TileShapeO{});
      constexpr int kAlignedQ = decltype(AlignedSGTileA_Q{})::value;
      constexpr int kReduceK = decltype(ReduceK{})::value;
      static_assert(kQ == 6 && kV == 256 && kReduceK == 4);

      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(thr_id));
      int const source_k = int(get<2>(thr_vak));
      int const lane =
          sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); ++i) {
        auto coord = tArA.tv_layout()(lane, i);
        int const q = int(get<0>(coord));
        int const v = int(get<1>(coord));
        shared.a_data[(source_k * kQ + q) * kV + v] = tArA(i);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA_max.size(); ++i) {
        int const q = int(tA_max.tv_layout()(lane).value());
        if (q < kQ) {
          shared.a_max_data[source_k * kAlignedQ + q] = tA_max(i);
          shared.a_sum_data[source_k * kAlignedQ + q] = tA_sum(i);
        }
      }

      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max;
      auto output_subtile = idx2crd(source_k, shape(ReduceSGLayout{}));
      int const q_offset =
          int(get<0>(output_subtile)) * size<0>(SGTileShapeO{});
      int const v_offset =
          int(get<1>(output_subtile)) * size<1>(SGTileShapeO{});

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA_max.size(); ++i) {
        int const q_local = int(rA_max.tv_layout()(lane).value());
        int const q = q_offset + q_local;
        if (q < kQ) {
          float global_max = shared.a_max_data[q];
          CUTLASS_PRAGMA_UNROLL
          for (int kr = 1; kr < kReduceK; ++kr) {
            global_max =
                sycl::fmax(global_max, shared.a_max_data[kr * kAlignedQ + q]);
          }
          float global_sum = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int kr = 0; kr < kReduceK; ++kr) {
            float const weight = sycl::native::exp2(
                shared.a_max_data[kr * kAlignedQ + q] - global_max);
            global_sum += shared.a_sum_data[kr * kAlignedQ + q] * weight;
          }
          rA_max(i) = global_max;
          rA_sum(i) = global_sum;
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rA.size(); ++i) {
        auto coord = rA.tv_layout()(lane, i);
        int const q = q_offset + int(get<0>(coord));
        int const v = v_offset + int(get<1>(coord));
        float global_max = shared.a_max_data[q];
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 1; kr < kReduceK; ++kr) {
          global_max =
              sycl::fmax(global_max, shared.a_max_data[kr * kAlignedQ + q]);
        }
        float value = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < kReduceK; ++kr) {
          float const weight = sycl::native::exp2(
              shared.a_max_data[kr * kAlignedQ + q] - global_max);
          value += shared.a_data[(kr * kQ + q) * kV + v] * weight;
        }
        rA(i) = value;
      }
      return std::make_tuple(rA, rA_max, rA_sum, true);
    } else {
      /* Identify A tile ID and k block for this subgroup. */
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      /* Set up SLM tensors and partition A tiles among participating subgroups
       */
      auto shape_A =
          append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
      auto shape_A_row = make_shape(
          get<0>(SGTileShapeO{}),
          shape(ReduceSGLayout{}),
          ReduceK{},
          SGPerWG{} / ReduceK{});

      /* Physical layouts, with subtile modes broken out */
      auto sA_layout = group<2, 4>(flat_divide(
          make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}),
          SGTileShapeO{}));
      auto sA_row_stride = make_stride(
          _1{},
          make_stride(get<0>(shape_A_row), _0{}),
          AlignedSGTileA_Q{},
          AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      /* Coordinate layouts, with subtile modes broken out */
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(
          append(SGTileShapeO{}, shape(ReduceSGLayout{})),
          append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_data),
          sA_layout);  // (q,v,rblk_dst,rblk_src,a_tile)
      auto sA_max = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_max_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
      auto sA_sum = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_sum_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) ||
                    (ReduceK{} == size(ReduceSGLayout{}));  // help compiler out

      /* Wait for maxima to be available, signal other data available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        /* Read A_max back from SLM and reduce. */
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); ++i)
            rA_max(i) =
                (rA_max(i) < rA_kmax[kr](i)) ? rA_kmax[kr](i) : rA_max(i);
        }

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); ++i)
            rA_kmax[kr](i) = sycl::native::exp2(rA_kmax[kr](i) - rA_max(i));
        }
      }

      /* Wait for A/A_sum data to be available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce.
         */
        clear(rA_sum);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragARow rA_sum_read;
          copy_block_s2r(sA_sum(_, k_blk, kr, a_tile), rA_sum_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_sum_read.size(); i++) {
            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
          }
        }

        clear(rA);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragA rA_read;
          copy_block_s2r(
              sA(_, _, k_blk, kr, a_tile), sA_coords(_, _, 0), rA_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_read.size(); i++) {
            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
          }
        }
      }
      return std::make_tuple(rA, rA_max, rA_sum, active);
    }
  }
};

}  // namespace cutlass::fmha::collective
