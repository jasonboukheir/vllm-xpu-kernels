#pragma once

#include <torch/all.h>

void kvarn_pack_balanced_kv_xe2(
    const at::Tensor& key_balanced,
    const at::Tensor& key_sinkhorn_col,
    const at::Tensor& key_sinkhorn_row,
    const at::Tensor& value_balanced,
    const at::Tensor& value_sinkhorn_col,
    const at::Tensor& value_sinkhorn_row,
    const at::Tensor& block_ids,
    at::Tensor& packed_cache,
    bool dpas_layout);
