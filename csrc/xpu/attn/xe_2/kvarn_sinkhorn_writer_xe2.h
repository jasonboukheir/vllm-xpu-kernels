#pragma once

#include <torch/all.h>

// Balance full rotated K/V pages and write the immutable Xe2 DPAS KVarN
// record directly.  This deliberately accepts the fp16/bf16 tail pools rather
// than the six fp32 tensors produced by the reference Sinkhorn path.
void kvarn_sinkhorn_pack_kv_xe2(
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    const at::Tensor& pool_slots,
    const at::Tensor& block_ids,
    at::Tensor& packed_cache,
    int64_t sinkhorn_iterations,
    bool dpas_layout);
