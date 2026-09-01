#pragma once

#include <torch/all.h>

std::tuple<at::Tensor, at::Tensor>
kvarn_fragment_coords_xe2(const at::Tensor& device_anchor);

void kvarn_decode_xe2(
    const at::Tensor& query,
    const at::Tensor& packed_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& block_to_slot,
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    at::Tensor& output,
    int64_t max_seq_len,
    double softmax_scale,
    bool unrotate_output);

void kvarn_decode_with_scratch_xe2(
    const at::Tensor& query,
    const at::Tensor& packed_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& block_to_slot,
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    at::Tensor& temp_output,
    at::Tensor& exp_sums,
    at::Tensor& max_logits,
    at::Tensor& output,
    int64_t max_seq_len,
    double softmax_scale,
    bool unrotate_output);

void kvarn_materialize_packed_kv_xe2(
    const at::Tensor& packed_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& block_to_slot,
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    at::Tensor& key_output,
    at::Tensor& value_output,
    int64_t max_seq_len);
