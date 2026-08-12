#include "kvarn_decode.hpp"
#include "kvarn_chunk_prefill.hpp"

namespace {

static_assert(KVarNDecodeD256G128Policy::HeadDim == 256);
static_assert(KVarNDecodeD256G128Policy::PageSize == 128);
static_assert(
    cutlass::fmha::collective::KVarNK4V4FragmentLoader<>::kValuesPerWord == 8);

using DecodeConfig = KVarNDecodeD256G128Config;
using CausalMainloop = cutlass::fmha::collective::KVarNDecodeFwdMainloop<
    DecodeConfig::TiledMMAQK,
    DecodeConfig::TiledMMAPV,
    DecodeConfig::VTiles,
    DecodeConfig::TensorQ,
    DecodeConfig::TensorK,
    DecodeConfig::TensorV,
    false,
    true>;

static_assert(CausalMainloop::CausalMask);

// Keep the cached-prefill specialization type-checked before its public
// launcher is wired in. Merely naming a class template does not instantiate
// its device operator, so this compile-only call deliberately exercises the
// bottom-right causal path with the existing D256/G128 fragment contract.
[[maybe_unused]] void instantiate_causal_mainloop(
    CausalMainloop::Params const& params,
    CausalMainloop::TensorQ2D const& q,
    CausalMainloop::TensorK2D const& k,
    CausalMainloop::TensorV2D const& v) {
  CausalMainloop::SharedStorage storage;
  CausalMainloop mainloop(params, storage);
  CausalMainloop::FragA out;
  CausalMainloop::FragARow row_max;
  CausalMainloop::FragARow row_sum;
  mainloop.template operator()<false>(
      q,
      k,
      v,
      out,
      row_max,
      row_sum,
      cute::make_coord(0, 0),
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0);
}

}  // namespace

cutlass::Status kvarn_decode_compile_spike(
    sycl::queue& queue, kvarn_decode_args_t const& args) {
  return KVarNDecodeD256G128Config::run(queue, args);
}

cutlass::Status kvarn_chunk_prefill_compile_spike(
    sycl::queue& queue, kvarn_chunk_prefill_args_t const& args) {
  return KVarNChunkPrefillD256G128Config::run(queue, args);
}
