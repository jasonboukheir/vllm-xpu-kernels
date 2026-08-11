#include "kvarn_decode.hpp"

namespace {

static_assert(KVarNDecodeD256G128Policy::HeadDim == 256);
static_assert(KVarNDecodeD256G128Policy::PageSize == 128);
static_assert(
    cutlass::fmha::collective::KVarNK4V4FragmentLoader<>::kValuesPerWord == 8);

}  // namespace

cutlass::Status kvarn_decode_compile_spike(
    sycl::queue& queue, kvarn_decode_args_t const& args) {
  return KVarNDecodeD256G128Config::run(queue, args);
}
