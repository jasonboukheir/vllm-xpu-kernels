#pragma once

#include <torch/all.h>

void kvarn_hadamard_scatter_xe2(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& slot_mapping,
    const at::Tensor& block_to_slot,
    at::Tensor& tail_key,
    at::Tensor& tail_value,
    int64_t group,
    bool dpas_layout);
