# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.utils import format_tc, seed_everything
from vllm_xpu_kernels.fused_moe_interface import (cutlass_grouped_gemm,
                                                  cutlass_grouped_gemm_xe2)

DEVICE = "xpu"

# shape for Llama-4-scout
FUSED_MOE_MNK_FACTORS = [
    (1, 5120, 8192),
    (4, 5120, 8192),
    (16, 5120, 8192),
    (8192, 5120, 8192),
]
NUM_EXPERTS = [16]
TOP_KS = [1]


def random_partition(size_a: int, target: int):
    cuts = sorted(random.sample(range(target + size_a - 1), size_a - 1))
    cuts = [-1] + cuts + [target + size_a - 1]
    result = [cuts[i + 1] - cuts[i] - 1 for i in range(size_a)]
    return result


MINI_PYTEST_PARAMS = {
    "default": {
        "m,n,k": [(1, 256, 128)],
        "e": [2],
        "topk": [1],
        "dtype": [torch.bfloat16],
        "has_bias": [True]
    }
}


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.skipif(True, reason="Need Fix API, skip for now")
def test_grouped_gemm(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    token_per_group = random_partition(e, m * topk)
    assert (len(token_per_group) == e)
    # input
    input_A = torch.randn((sum(token_per_group), k),
                          dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B = torch.randn((num_experts, n, k), dtype=dtype, device=DEVICE)
    input_B = input_B.transpose(-1, -2).contiguous()
    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE)
    else:
        bias = None

    # output offset
    output = torch.empty((sum(token_per_group), n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm(input_A, input_B, bias, output, token_per_group, n, k,
                         num_experts)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = token_per_group[i]
        if cur_token_num == 0:
            continue
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :]
        weight = input_B[i, :, :]
        expert_output = input @ weight
        if has_bias:
            expert_output += bias[i]
        ref.append(expert_output)
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=2e-2, atol=1e-2)


def init_rows_for_experts(tokens, topk, num_rows_per_expert):
    if num_rows_per_expert.shape[0] == 1:
        num_rows_per_expert[0] = tokens * topk
        return
    n_experts = num_rows_per_expert.numel()
    rand = torch.rand(tokens, n_experts, device=num_rows_per_expert.device)
    topk_idx = torch.topk(rand, topk, dim=1).indices  # [tokens, topk]
    flat_idx = topk_idx.flatten()
    num_rows_per_expert += torch.bincount(flat_idx, minlength=n_experts)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [True, False])
def test_xe_grouped_gemm(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B = torch.randn((num_experts, k, n), dtype=dtype, device=DEVICE)
    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE)
    else:
        bias = None

    # output offset
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)
    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)

    cutlass_grouped_gemm_xe2(input_A, input_B, None, bias, output,
                             num_rows_per_expert, n, k, num_experts)

    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
        if cur_token_num == 0:
            continue
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :].to(
            torch.float32)
        weight = input_B[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight
        if has_bias:
            expert_output_fp32 += bias[i]
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e5m2, torch.float8_e4m3fn],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_fp8(m, n, k, e, topk, dtype, fp8_dtype, has_bias):
    seed_everything(7)
    num_experts = e
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B = torch.randn((num_experts, k, n), dtype=dtype, device=DEVICE)
    # scale
    random_exponents = torch.randint(-3, 4, (num_experts, ), device=DEVICE)
    scale_B = torch.pow(2.0, random_exponents.float())
    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) * 100
    else:
        bias = None

    # quantize weight
    input_B_fp8 = torch.empty_like(input_B, dtype=fp8_dtype)
    for i in range(num_experts):
        input_B_fp8[i], _ = scaled_fp8_quant(input_B[i],
                                             scale_B[i].to(torch.float32),
                                             False,
                                             False,
                                             fp8_dtype=fp8_dtype)
    input_B_dequatize = torch.empty_like(input_B, dtype=dtype)
    for i in range(num_experts):
        input_B_dequatize[i] = (input_B_fp8[i].to(torch.float32) *
                                scale_B[i]).to(dtype)

    # output offset
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)
    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)

    cutlass_grouped_gemm_xe2(input_A, input_B_fp8, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
        if cur_token_num == 0:
            continue
        # mma uses fp32 as calculate dtype
        # so here use fp32 to avoid accuracy error
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :].to(
            torch.float32)
        weight = input_B_dequatize[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight
        if has_bias:
            expert_output_fp32 += bias[i]
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)


def dequantize_uint4(qweight, scales, group_size):
    import numpy as np
    k = qweight.shape[1] * 2
    n = qweight.shape[0]
    unpack_idx = np.array([0, 1])
    data = qweight[:, [i // 2 for i in range(k)]]
    shift = (torch.tensor(unpack_idx[[i % 2 for i in range(k)]],
                          dtype=torch.int32,
                          device="xpu")[None, :].expand([n, -1]) * 4)
    dst_data = (data >> shift) & 0xF
    expand_scales = scales[:, [i // group_size for i in range(k)]]
    weight_16 = (dst_data - 8) * expand_scales

    return weight_16.to(scales.dtype)


def implement_zp(qweight, zp=None):
    assert qweight.dtype == torch.uint8, "Input tensor must be uint8"

    high_u4 = (qweight >> 4) & 0x0F
    low_u4 = qweight & 0x0F

    high_s8 = high_u4.to(torch.int8)
    low_s8 = low_u4.to(torch.int8)

    high_s8 = high_s8 - 8
    low_s8 = low_s8 - 8

    def pack_compact(a, b):

        def process_number(x):
            sign = (x < 0).to(torch.uint8)
            abs_low3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
            return (sign << 3) | abs_low3

        packed_a = process_number(a)
        packed_b = process_number(b)

        return (packed_a << 4) | packed_b

    result = pack_compact(high_s8, low_s8)

    return result


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_int4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    group_size = 128
    group_num = k // group_size
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B_uint4 = (torch.randint(0,
                                   0xff, [num_experts, n, k // 2],
                                   device=DEVICE)).to(torch.uint8)
    # scale
    random_exponents = torch.randint(-3,
                                     4, (num_experts, n, group_num),
                                     device=DEVICE)
    scale_B = torch.pow(2.0, random_exponents.float()).to(dtype)

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) * 100
    else:
        bias = None

    input_B_16 = torch.empty(num_experts, n, k, dtype=dtype, device=DEVICE)
    input_B_int4 = torch.empty_like(input_B_uint4).to(torch.int8)
    for i in range(num_experts):
        # default zp=8
        input_B_16[i] = dequantize_uint4(input_B_uint4[i], scale_B[i],
                                         group_size)
        input_B_int4[i] = implement_zp(input_B_uint4[i], None)

    # output offset
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)

    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm_xe2(input_A, input_B_int4, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
        if cur_token_num == 0:
            continue
        # mma uses fp32 as calculate dtype
        # so here use fp32 to avoid accuracy error
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :].to(
            torch.float32)
        weight = input_B_16[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight.T
        if has_bias:
            expert_output_fp32 += bias[i]
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("m,k", [(2048, 5120), (1535, 17408), (2047, 17408)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_int4_dense_policies(m, k, has_bias, monkeypatch):
    """Real prefill M/K, bounded N, BF16 oracle and caller-owned tails."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    seed_everything(7)
    n, group_size = 512, 128
    dtype = torch.bfloat16
    input_A = torch.randn((m, k), dtype=dtype, device=DEVICE) / 10
    raw = torch.randint(0, 256, (1, n, k // 2), dtype=torch.uint8,
                        device=DEVICE)
    input_B = implement_zp(raw).view(torch.int8)
    scales = (torch.rand((1, n, k // group_size), device=DEVICE) / 10
              + 0.01).to(dtype)
    weights = dequantize_uint4(raw[0], scales[0], group_size)
    bias = (torch.linspace(-0.1, 0.1, n, device=DEVICE).to(dtype)[None, :]
            if has_bias else None)
    reference = input_A.float() @ weights.float().T
    if bias is not None:
        reference += bias.float()
    reference = reference.to(dtype)
    rows = torch.tensor([m], dtype=torch.int32, device=DEVICE)

    def run(activations):
        storage = torch.full((m + 2, n), -42, dtype=dtype, device=DEVICE)
        output = storage[1:-1]
        output.fill_(float("nan"))
        result = torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            activations, None, input_B, scales, bias, output, rows, n, k, 1)
        torch.xpu.synchronize()
        assert result.data_ptr() == output.data_ptr()
        assert torch.all(storage[[0, -1]] == -42)
        assert torch.isfinite(output).all()
        return output

    # The original kernel truncates dequantized BF16 weights; its distinct
    # arithmetic is checked exactly by the identity test below. Preserve its
    # output here while requiring both candidates to meet the RNE oracle.
    monkeypatch.delenv("VLLM_XPU_INT4_DENSE_POLICY", raising=False)
    original = run(input_A)
    for policy in ("128x128", "256x128", "128x128"):
        monkeypatch.setenv("VLLM_XPU_INT4_DENSE_POLICY", policy)
        output = run(input_A)
        torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
        for _ in range(3):
            decoy = run(-input_A)
            pressure = torch.empty((1048573,), dtype=torch.uint8,
                                   device=DEVICE)
            repeated = run(input_A)
            torch.testing.assert_close(repeated, output, atol=0, rtol=0)
            del decoy, pressure, repeated

    monkeypatch.delenv("VLLM_XPU_INT4_DENSE_POLICY", raising=False)
    torch.testing.assert_close(run(input_A), original, atol=0, rtol=0)
    monkeypatch.setenv("VLLM_XPU_INT4_DENSE_POLICY", "invalid")
    with pytest.raises(RuntimeError, match="VLLM_XPU_INT4_DENSE_POLICY"):
        run(input_A)


def test_xe_grouped_gemm_int4_dense_rounding(monkeypatch):
    """Identity inputs isolate RNE candidates from original RTZ dequant."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    m, n, k = 256, 256, 128
    dtype = torch.bfloat16
    # Build the independent oracle on CPU, including positive/negative scales
    # and the halfway products that distinguish ties-to-even from truncation.
    unsigned = (torch.arange(n)[:, None] + torch.arange(k)[None, :]) % 16
    raw = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).to(torch.uint8)
    packed = raw.bitwise_xor(0x88).view(torch.int8)[None, :].to(DEVICE)
    scale_values = [0.0634765625, -0.0634765625, 0.10107421875,
                    -0.10107421875]
    scales_cpu = torch.tensor(scale_values, dtype=dtype).repeat(n // 4)
    products = (unsigned - 8).float() * scales_cpu[:, None].float()
    rne_weights = products.to(dtype)
    rtz_weights = (products.view(torch.int32).bitwise_and(-65536)
                   .view(torch.float32).to(dtype))
    assert not torch.equal(rne_weights, rtz_weights)
    references = {
        "rne": torch.cat((rne_weights.T, -rne_weights.T)),
        "rtz": torch.cat((rtz_weights.T, -rtz_weights.T)),
    }
    input_A = torch.cat((torch.eye(k), -torch.eye(k))).to(dtype).to(DEVICE)
    scales = scales_cpu.reshape(1, n, 1).to(DEVICE)
    rows = torch.tensor([m], dtype=torch.int32, device=DEVICE)

    for policy in (None, "128x128", None, "256x128", "128x128", ""):
        if policy is None:
            monkeypatch.delenv("VLLM_XPU_INT4_DENSE_POLICY", raising=False)
        else:
            monkeypatch.setenv("VLLM_XPU_INT4_DENSE_POLICY", policy)
        storage = torch.full((m + 2, n), -42, dtype=dtype, device=DEVICE)
        output = storage[1:-1]
        output.fill_(float("nan"))
        result = torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            input_A, None, packed, scales, None, output, rows, n, k, 1)
        torch.xpu.synchronize()
        assert result.data_ptr() == output.data_ptr()
        assert torch.all(storage[[0, -1]] == -42)
        expected = references["rne" if policy else "rtz"]
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("m,e,dtype,group_size", [
    (128, 1, torch.bfloat16, 128),
    (129, 2, torch.bfloat16, 128),
    (129, 1, torch.float16, 128),
    (129, 1, torch.bfloat16, 64),
])
def test_xe_grouped_gemm_int4_dense_scope(m, e, dtype, group_size,
                                        monkeypatch):
    """An invalid selector is ignored outside the dense BF16 G128 scope."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    seed_everything(7)
    n, k = 256, 512
    input_A = torch.randn((m * e, k), dtype=dtype, device=DEVICE) / 10
    raw = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8,
                        device=DEVICE)
    input_B = implement_zp(raw).view(torch.int8)
    scales = torch.full((e, n, k // group_size), 0.125, dtype=dtype,
                        device=DEVICE)
    rows = torch.full((e,), m, dtype=torch.int32, device=DEVICE)

    def run():
        output = torch.empty((m * e, n), dtype=dtype, device=DEVICE)
        return torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            input_A, None, input_B, scales, None, output, rows, n, k, e)

    monkeypatch.delenv("VLLM_XPU_INT4_DENSE_POLICY", raising=False)
    original = run()
    monkeypatch.setenv("VLLM_XPU_INT4_DENSE_POLICY", "invalid")
    torch.testing.assert_close(run(), original, atol=0, rtol=0)


def dequantize_mxfp4(qweight, scales, group_size, dtype):
    import numpy as np
    k = qweight.shape[1] * 2
    n = qweight.shape[0]
    unpack_idx = np.array([0, 1])
    data = qweight[:, [i // 2 for i in range(k)]]
    shift = (torch.tensor(unpack_idx[[i % 2 for i in range(k)]],
                          dtype=torch.int32,
                          device="xpu")[None, :].expand([n, -1]) * 4)
    dst_data = (data >> shift) & 0xF

    table = torch.tensor([
        +0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
                         dtype=dtype,
                         device="xpu")
    dst_data = table[dst_data]
    expand_scales = scales[:, [i // group_size for i in range(k)]]
    dst_scale = (expand_scales.to(torch.int32) << 7).to(torch.uint16).view(
        torch.bfloat16).to(dtype)
    weight_16 = dst_data * dst_scale
    # weight_16 = dst_data

    return weight_16.to(dtype)


@pytest.mark.parametrize("m,n,k", FUSED_MOE_MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_mxfp4(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    group_size = 32
    group_num = k // group_size
    total_m = m * topk
    # input
    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous()
    ref_A = input_A
    # weight
    input_B_mxfp4 = (torch.randint(0,
                                  0xff, [num_experts, n, k // 2],
                                  device=DEVICE)).to(torch.uint8)
    # scale
    scale_B = torch.randint(0,
                            0x7f, (num_experts, n, group_num),
                            dtype=torch.uint8,
                            device="xpu")

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) * 100
    else:
        bias = None

    input_B_16 = torch.empty(num_experts, n, k, dtype=dtype, device=DEVICE)
    for i in range(num_experts):
        input_B_16[i] = dequantize_mxfp4(input_B_mxfp4[i], scale_B[i],
                                         group_size, dtype)

    # output offset
    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)

    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)
    input_B_mxfp4 = input_B_mxfp4.view(torch.float4_e2m1fn_x2)
    cutlass_grouped_gemm_xe2(input_A, input_B_mxfp4, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts)
    # ref gg
    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = num_rows_per_expert[i]
        if cur_token_num == 0:
            continue
        # mma uses fp32 as calculate dtype
        # so here use fp32 to avoid accuracy error
        input = ref_A[pre_token_sum:pre_token_sum + cur_token_num, :].to(
            torch.float32)
        weight = input_B_16[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight.T
        if has_bias:
            expert_output_fp32 += bias[i]
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)


def dequantize_mxfp8_wei_kn(wei, wei_scale, group_size=32):
    """Dequant MXFP8 weight [K, N] with scales [K/group, N] (E8M0 bits)."""
    scale_f = wei_scale.view(torch.float8_e8m0fnu).to(torch.float32)
    return wei.to(torch.float32) * scale_f.repeat_interleave(group_size, dim=0)


@pytest.mark.parametrize("m,n,k", [(1, 256, 128), (4, 256, 128)])
@pytest.mark.parametrize("e", [2])
@pytest.mark.parametrize("topk", [1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_mxfp8(m, n, k, e, topk, dtype, has_bias):
    """Native MXFP8 W8A16 grouped GEMM vs dequant+matmul gold."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    seed_everything(7)
    num_experts = e
    group_size = 32
    assert k % group_size == 0
    group_num = k // group_size
    total_m = m * topk

    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous() / 10
    weight_hp = torch.randn((num_experts, k, n), dtype=torch.float32,
                            device=DEVICE) / 10
    input_B = torch.empty(num_experts, k, n, dtype=torch.float8_e4m3fn,
                          device=DEVICE)
    scale_B = torch.empty(num_experts,
                          group_num,
                          n,
                          dtype=torch.uint8,
                          device=DEVICE)

    from tests.ops.mx_utils import to_mxfp
    for i in range(num_experts):
        # to_mxfp blocks the last dim; we need E8M0 along K → quantize [N, K].
        sc, lp = to_mxfp(weight_hp[i].transpose(0, 1).contiguous().cpu(),
                         format="mxfp8")
        input_B[i] = lp.transpose(0, 1).contiguous().to(device=DEVICE)
        scale_B[i] = sc.transpose(0, 1).contiguous().view(
            torch.uint8).to(device=DEVICE)

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) / 10
    else:
        bias = None

    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)

    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm_xe2(input_A, input_B, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts)

    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = int(num_rows_per_expert[i].item())
        if cur_token_num == 0:
            continue
        inp = input_A[pre_token_sum:pre_token_sum + cur_token_num].to(
            torch.float32)
        wei = dequantize_mxfp8_wei_kn(input_B[i], scale_B[i], group_size)
        expert_out = inp @ wei
        if has_bias:
            expert_out = expert_out + bias[i].to(torch.float32)
        ref.append(expert_out.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("m,n,k", [(1, 256, 256), (4, 256, 256)])
@pytest.mark.parametrize("e", [2])
@pytest.mark.parametrize("topk", [1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=format_tc)
@pytest.mark.parametrize("has_bias", [False, True])
def test_xe_grouped_gemm_block_fp8(m, n, k, e, topk, dtype, has_bias):
    """Native block-FP8 W8A16 grouped GEMM vs dequant+matmul gold.

    Keeps FP8 weights + float32 2D scales [E, K/128, N/128] in memory.
    """
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    seed_everything(7)
    from vllm_xpu_kernels.moe_utils import dequant_fp8_block_wei

    num_experts = e
    group_size = 128
    assert k % group_size == 0 and n % group_size == 0
    total_m = m * topk

    input_A = torch.randn((total_m, k), dtype=dtype,
                          device=DEVICE).contiguous() / 10
    input_B = torch.empty(num_experts, k, n, dtype=torch.float8_e4m3fn,
                          device=DEVICE)
    scale_B = (torch.randn(num_experts,
                           k // group_size,
                           n // group_size,
                           dtype=torch.float32,
                           device=DEVICE).abs() + 0.01)

    for i in range(num_experts):
        hp = torch.randn(k, n, dtype=torch.float32, device=DEVICE) / 10
        input_B[i] = hp.to(torch.float8_e4m3fn)

    if has_bias:
        bias = torch.randn((num_experts, n), dtype=dtype, device=DEVICE) / 10
    else:
        bias = None

    num_rows_per_expert = torch.zeros(num_experts,
                                      device=DEVICE,
                                      dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)

    output = torch.empty((total_m, n), dtype=dtype, device=DEVICE)
    cutlass_grouped_gemm_xe2(input_A, input_B, scale_B, bias, output,
                             num_rows_per_expert, n, k, num_experts)

    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = int(num_rows_per_expert[i].item())
        if cur_token_num == 0:
            continue
        inp = input_A[pre_token_sum:pre_token_sum + cur_token_num].to(
            torch.float32)
        wei = dequant_fp8_block_wei(input_B[i], scale_B[i])
        expert_out = inp @ wei
        if has_bias:
            expert_out = expert_out + bias[i].to(torch.float32)
        ref.append(expert_out.to(dtype))
        pre_token_sum += cur_token_num
    ref = torch.cat(ref, dim=0)

    torch.testing.assert_close(output, ref, rtol=2e-2, atol=2e-2)
