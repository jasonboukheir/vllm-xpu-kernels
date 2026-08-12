#include <torch/library.h>

#include "xpu/attn/xe_2/kvarn_decode_xe2.h"

// Standalone registration for the narrow KVarN development library. Keep it
// outside csrc: the production extension collects csrc/flash_attn/*.cpp and
// owns the namespace's sole TORCH_LIBRARY block in flash_api.cpp.
TORCH_LIBRARY(_vllm_fa2_C, ops) {
  ops.def(
      "kvarn_decode(Tensor query, Tensor packed_cache, Tensor block_table, "
      "Tensor seq_lens, Tensor block_to_slot, Tensor tail_key, Tensor "
      "tail_value, Tensor! output, int max_seq_len, float softmax_scale) -> "
      "()");
  ops.impl("kvarn_decode", torch::kXPU, &kvarn_decode_xe2);
  ops.def(
      "kvarn_decode_with_scratch(Tensor query, Tensor packed_cache, Tensor "
      "block_table, Tensor seq_lens, Tensor block_to_slot, Tensor tail_key, "
      "Tensor tail_value, Tensor(a!) temp_output, Tensor(b!) exp_sums, "
      "Tensor(c!) max_logits, Tensor(d!) output, int max_seq_len, float "
      "softmax_scale) -> ()");
  ops.impl(
      "kvarn_decode_with_scratch", torch::kXPU,
      &kvarn_decode_with_scratch_xe2);
  ops.def(
      "kvarn_chunk_prefill(Tensor query, Tensor packed_cache, Tensor "
      "block_table, Tensor seq_lens, Tensor cu_seqlens_q, Tensor "
      "block_to_slot, Tensor tail_key, Tensor tail_value, Tensor! output, "
      "int max_query_len, int max_seq_len, float softmax_scale) -> ()");
  ops.impl("kvarn_chunk_prefill", torch::kXPU, &kvarn_chunk_prefill_xe2);
  ops.def(
      "kvarn_materialize_packed_kv(Tensor packed_cache, Tensor block_table, "
      "Tensor seq_lens, Tensor cu_seqlens_k, Tensor block_to_slot, Tensor "
      "tail_key, Tensor tail_value, Tensor(a!) key_output, Tensor(b!) "
      "value_output, int max_seq_len) -> ()");
  ops.impl(
      "kvarn_materialize_packed_kv", torch::kXPU,
      &kvarn_materialize_packed_kv_xe2);
}
