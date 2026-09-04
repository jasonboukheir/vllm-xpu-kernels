#pragma once

#include <torch/all.h>

void kvarn_dequant_xe2(
    const at::Tensor& packed_cache,
    at::Tensor& key_out,
    at::Tensor& value_out,
    bool dpas_layout);
