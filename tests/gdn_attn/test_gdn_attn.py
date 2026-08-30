# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import random

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import format_tc

# QWEN NEXT shape
NUM_TOKENS = [1, 32, 1024, 8192]
BATCH_SIZE = [32]
NUM_K_HEADS = [16]
NUM_K_DIMS = [128]
NUM_V_HEADS = [32]
NUM_V_DIMS = [128]
WIDTH = [4]
TP_SIZE = [1]
HAS_BIAS = [True, False]
ACTIVATION = ["silu"]
MODE = ["prefill", "decode", "mix_mode"]
REORDER_INPUT = [True, False]
DTYPES = [torch.float16, torch.bfloat16]
SSM_STATE_IS_FP32 = [False, True]

# Override pytest parameters when enabling mini pytest
MINI_PYTEST_PARAMS = {
"default": {
    "num_actual_tokens": [16],
    "batch_size": [16],
    "num_k_heads": [1],
    "head_k_dim": [32],
    "num_v_heads": [1],
    "head_v_dim": [32],
    "width": [2],
    "tp_size": [1],
    "has_bias": [False],
    "activation": ["silu"],
    "mode": ["prefill", "decode", "mix_mode"],
    "reorder_input": [False],
    "dtype": [torch.float16], 
    "ssm_state_is_fp32": [False],
    },
}


def ref_gdn_attention(
    core_attn_out,
    z,
    projected_states_qkvz,
    projected_states_ba,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    conv_state,
    ssm_state,
    conv_weights,
    conv_bias,
    activation,
    A_log,
    dt_bias,
    num_prefills,
    num_decodes,
    has_initial_state,
    non_spec_query_start_loc,
    non_spec_state_indices_tensor,
    num_actual_tokens,
    tp_size,
    reorder_input,
):
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = projected_states_qkvz.dtype
    batch_size = non_spec_query_start_loc.shape[0] - 1

    qkv, b, a, z_global = _extract_qkv_b_a_z(
        projected_states_qkvz, projected_states_ba, num_actual_tokens,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size,
        reorder_input)
    qkv_elems_size = qkv.shape[-1]
    z.copy_(z_global)

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)
    if conv_bias is not None:
        conv_bias = conv_bias.to(torch.float)

    # Cache lines may reserve extra spec-decode rows; non-spec uses only the
    # leading (width - 1) history rows.
    width = conv_weights.shape[-1]

    for batch in range(batch_size):
        if has_initial_state[batch]:
            conv_state_batch = conv_state[
                non_spec_state_indices_tensor[batch]][:width - 1]
        else:
            conv_state_batch = torch.zeros_like(conv_state[0][:width - 1])

        batch_start_id = non_spec_query_start_loc[batch]
        batch_end_id = non_spec_query_start_loc[batch + 1]
        batch_num_tokens = batch_end_id - batch_start_id

        qkv_batch = qkv[batch_start_id:batch_end_id]
        qkv_conv_input = torch.cat([conv_state_batch, qkv_batch], dim=0)
        conv_state[non_spec_state_indices_tensor[batch]][:width - 1] = (
            qkv_conv_input[batch_num_tokens:])

        qkv_conv_input = qkv_conv_input.transpose(0, 1).unsqueeze(0)

        qkv_conv_out = F.conv1d(qkv_conv_input.to(torch.float32),
                                conv_weights.unsqueeze(1).to(torch.float32),
                                conv_bias,
                                padding=0,
                                groups=qkv_elems_size)
        qkv_conv_out = (qkv_conv_out if activation is None else
                        F.silu(qkv_conv_out)).to(dtype=dtype)
        qkv_conv_out = qkv_conv_out.transpose(-2, -1).reshape(
            batch_num_tokens, qkv_elems_size)

        split_arg_list_qkv = [
            num_k_heads // tp_size * head_k_dim,
            num_k_heads // tp_size * head_k_dim,
            num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim,
        ]
        (q_out, k_out, v_out) = torch.split(qkv_conv_out,
                                            split_arg_list_qkv,
                                            dim=-1)
        q_out = q_out.reshape(batch_num_tokens, num_k_heads // tp_size,
                              head_k_dim)
        k_out = k_out.reshape(batch_num_tokens, num_k_heads // tp_size,
                              head_k_dim)
        v_out = v_out.reshape(batch_num_tokens, num_v_heads // tp_size,
                              head_v_dim)

        if has_initial_state[batch]:
            ssm_state_batch = ssm_state[
                non_spec_state_indices_tensor[batch]].to(
                    torch.float32
                )  # [num_v_heads // tp_size, head_v_dim, head_k_dim]
        else:
            ssm_state_batch = torch.zeros_like(ssm_state[0],
                                               dtype=torch.float32)

        # ------------------------------------------------------------------
        # Hoist all per-token elementwise work out of the recurrence loop.
        # Only the SSM state update itself is sequential.
        # ------------------------------------------------------------------
        rep = num_v_heads // num_k_heads
        b_batch = b[batch_start_id:batch_end_id].to(
            torch.float32)  # [T, NV]
        a_batch = a[batch_start_id:batch_end_id].to(torch.float32)
        beta_batch = torch.sigmoid(b_batch)  # [T, NV]
        g_batch = torch.exp(A_log_exp *
                            softplus(a_batch + dt_bias))  # [T, NV]

        q_all = q_out.to(torch.float32)  # [T, NK, Hk]
        k_all = k_out.to(torch.float32)
        v_all = v_out.to(torch.float32)  # [T, NV, Hv]

        # l2norm along head dim, then scale q.
        q_all = q_all * torch.rsqrt(q_all.pow(2).sum(-1, keepdim=True) + eps)
        k_all = k_all * torch.rsqrt(k_all.pow(2).sum(-1, keepdim=True) + eps)
        q_all = q_all * scale

        # GQA: replicate K/Q heads NK -> NV.
        if rep > 1:
            q_all = q_all.repeat_interleave(rep, dim=1)  # [T, NV, Hk]
            k_all = k_all.repeat_interleave(rep, dim=1)

        out_buf = torch.empty(batch_num_tokens,
                              num_v_heads // tp_size,
                              head_v_dim,
                              dtype=torch.float32,
                              device=core_attn_out.device)

        # O(t) = S(t) * q(t)
        # S(t) = g(t)*S(t - 1) + (v(t) - g(t)*S(t - 1)*k(t))*beta(t)*k(t)
        for token_id in range(batch_num_tokens):
            g_t = g_batch[token_id]  # [NV]
            beta_t = beta_batch[token_id]  # [NV]
            q_t = q_all[token_id]  # [NV, Hk]
            k_t = k_all[token_id]  # [NV, Hk]
            v_t = v_all[token_id]  # [NV, Hv]

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)

            # kv_mem_t[v, h] = sum_k S[v, h, k] * k_t[v, k]
            kv_mem_t = torch.einsum("vhk,vk->vh", ssm_state_batch, k_t)
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)  # [NV, Hv]

            # outer product update: S[v, h, k] += delta[v, h] * k_t[v, k]
            ssm_state_batch.add_(
                torch.einsum("vh,vk->vhk", delta_t, k_t))

            out_buf[token_id] = torch.einsum("vhk,vk->vh", ssm_state_batch,
                                             q_t)

        core_attn_out[batch_start_id:batch_end_id] = out_buf.to(dtype)
        ssm_state[non_spec_state_indices_tensor[batch]] = ssm_state_batch.to(
            ssm_state.dtype)


def simple_random_distribute(N, batch_size):
    distribution = torch.ones([batch_size])
    for i in range(N - batch_size):
        selected_idx = random.randint(0, batch_size - 1)
        distribution[selected_idx] += 1

    return distribution


@pytest.mark.parametrize("num_actual_tokens", NUM_TOKENS)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("num_k_heads", NUM_K_HEADS)
@pytest.mark.parametrize("head_k_dim", NUM_K_DIMS)
@pytest.mark.parametrize("num_v_heads", NUM_V_HEADS)
@pytest.mark.parametrize("head_v_dim", NUM_V_DIMS)
@pytest.mark.parametrize("width", WIDTH)
@pytest.mark.parametrize("tp_size", TP_SIZE)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("activation", ACTIVATION)
@pytest.mark.parametrize("mode", MODE)
@pytest.mark.parametrize("reorder_input", REORDER_INPUT)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("ssm_state_is_fp32", SSM_STATE_IS_FP32)
@torch.inference_mode()
def test_gdn_attention(num_actual_tokens, batch_size, num_k_heads, head_k_dim,
                       num_v_heads, head_v_dim, width, tp_size, has_bias,
                       activation, reorder_input, mode, dtype,
                       ssm_state_is_fp32):
    # FIXME: remove skip
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(42)
    torch.manual_seed(42)
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

    assert head_k_dim == head_v_dim

    if batch_size > num_actual_tokens:
        batch_size = num_actual_tokens

    if mode == "prefill":
        num_prefills = batch_size
    elif mode == "decode":
        num_prefills = 0
        if batch_size < num_actual_tokens:
            return
    else:
        num_prefills = random.randint(1, batch_size -
                                      1) if batch_size > 1 else 1

    num_decodes = batch_size - num_prefills
    cache_batch_size = 200

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype,
                                      device=device)

    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype,
                             device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=ssm_state_dtype,
        device=device)
    ref_ssm_state = ssm_state.clone()

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

    A_log = torch.randn((num_v_heads // tp_size),
                        dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)

    prefill_batches = simple_random_distribute(num_actual_tokens - num_decodes,
                                               batch_size - num_decodes)
    token_batches = torch.cat([torch.ones([num_decodes]),
                               prefill_batches]).to(device)
    perm = torch.randperm(token_batches.size(0)).to(device)
    shuffled_tensor = token_batches[perm]
    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)
    has_initial_state = perm >= num_decodes
    non_spec_state_indices_tensor = torch.tensor(random.sample(
        range(cache_batch_size), batch_size),
                                                 device=device,
                                                 dtype=torch.int32)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype,
        device=device,
    )
    z = torch.empty_like(core_attn_out)

    intermediates = torch.ops._xpu_C.causal_conv1d_non_spec(
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input)

    torch.ops._xpu_C.gated_delta_rule_non_spec(
        core_attn_out,
        *intermediates,
        num_v_heads,
        head_v_dim,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)

    ref_gdn_attention(
        ref_core_attn_out,
        ref_z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)

    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)
    for i in range(batch_size):
        state_id = non_spec_state_indices_tensor[i]
        torch.testing.assert_close(conv_state[state_id],
                                   ref_conv_state[state_id],
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(ssm_state[state_id],
                                   ref_ssm_state[state_id],
                                   atol=atol,
                                   rtol=rtol)


@pytest.mark.parametrize("num_actual_tokens", NUM_TOKENS)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("num_k_heads", NUM_K_HEADS)
@pytest.mark.parametrize("head_k_dim", NUM_K_DIMS)
@pytest.mark.parametrize("num_v_heads", NUM_V_HEADS)
@pytest.mark.parametrize("head_v_dim", NUM_V_DIMS)
@pytest.mark.parametrize("width", WIDTH)
@pytest.mark.parametrize("tp_size", TP_SIZE)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("activation", ACTIVATION)
@pytest.mark.parametrize("mode", MODE)
@pytest.mark.parametrize("reorder_input", REORDER_INPUT)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@pytest.mark.parametrize("ssm_state_is_fp32", SSM_STATE_IS_FP32)
@torch.inference_mode()
def test_gdn_attention_legacy(num_actual_tokens, batch_size, num_k_heads,
                              head_k_dim, num_v_heads, head_v_dim, width,
                              tp_size, has_bias, activation, reorder_input,
                              mode, dtype, ssm_state_is_fp32):
    # Backward-compat test for the legacy fused gdn_attention op, which is now
    # a thin wrapper that chains causal_conv1d and gated_delta_rule. This is
    # the original test_gdn_attention case kept verbatim against the fused
    # entry point.
    # FIXME: remove skip
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(42)
    torch.manual_seed(42)
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

    assert head_k_dim == head_v_dim

    if batch_size > num_actual_tokens:
        batch_size = num_actual_tokens

    if mode == "prefill":
        num_prefills = batch_size
    elif mode == "decode":
        num_prefills = 0
        if batch_size < num_actual_tokens:
            return
    else:
        num_prefills = random.randint(1, batch_size -
                                      1) if batch_size > 1 else 1

    num_decodes = batch_size - num_prefills
    cache_batch_size = 200

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype,
                                      device=device)

    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype,
                             device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=ssm_state_dtype,
        device=device)
    ref_ssm_state = ssm_state.clone()

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

    A_log = torch.randn((num_v_heads // tp_size),
                        dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)

    prefill_batches = simple_random_distribute(num_actual_tokens - num_decodes,
                                               batch_size - num_decodes)
    token_batches = torch.cat([torch.ones([num_decodes]),
                               prefill_batches]).to(device)
    perm = torch.randperm(token_batches.size(0)).to(device)
    shuffled_tensor = token_batches[perm]
    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)
    has_initial_state = perm >= num_decodes
    non_spec_state_indices_tensor = torch.tensor(random.sample(
        range(cache_batch_size), batch_size),
                                                 device=device,
                                                 dtype=torch.int32)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype,
        device=device,
    )
    z = torch.empty_like(core_attn_out)

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)

    ref_gdn_attention(
        ref_core_attn_out,
        ref_z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)

    if num_actual_tokens == 8192:
        pytest.skip("FIXME, skip core_attn_out test because of random error")

    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)
    for i in range(batch_size):
        state_id = non_spec_state_indices_tensor[i]
        torch.testing.assert_close(conv_state[state_id],
                                   ref_conv_state[state_id],
                                   atol=atol,
                                   rtol=rtol)
        # FIXME: the ssm_state check is skipped for num_actual_tokens == 8192
        # due to random error; will be fixed in future. (8192 is already
        # skipped earlier, so the assertion runs for the reached cases.)
        if num_actual_tokens != 8192:
            torch.testing.assert_close(ssm_state[state_id],
                                       ref_ssm_state[state_id],
                                       atol=atol,
                                       rtol=rtol)


NUM_SPEC_DECODES = [1, 4]
NUM_SPEC_TOKENS = [2, 3]  # num_speculative_tokens + 1


def _extract_qkv_b_a_z(projected_states_qkvz, projected_states_ba,
                       num_actual_tokens, num_k_heads, num_v_heads,
                       head_k_dim, head_v_dim, tp_size, reorder_input):
    """Replicate the qkv/ba split done in `ref_gdn_attention` and return
    the post-reorder qkv (for conv1d), b, a, z tensors plus split sizes."""
    if reorder_input:
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        q_size = key_dim // tp_size
        k_size = q_size
        v_size = value_dim // tp_size
        z_size = v_size
        q_tmp, k_tmp, v_tmp, z_tmp = projected_states_qkvz.split(
            [q_size, k_size, v_size, z_size], dim=-1)
        q_tmp = q_tmp.reshape(q_tmp.size(0), -1, head_k_dim)
        k_tmp = k_tmp.reshape(k_tmp.size(0), -1, head_k_dim)
        v_tmp = v_tmp.reshape(v_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        z_tmp = z_tmp.reshape(z_tmp.size(0), -1,
                              num_v_heads // num_k_heads * head_v_dim)
        projected_states_qkvz = torch.cat(
            [q_tmp, k_tmp, v_tmp, z_tmp],
            dim=-1).reshape(q_tmp.size(0), -1).contiguous()

        b, a = projected_states_ba.chunk(2, dim=-1)
        b = b.reshape(b.size(0), -1, num_v_heads // num_k_heads)
        a = a.reshape(a.size(0), -1, num_v_heads // num_k_heads)
        projected_states_ba = torch.cat(
            [b, a], dim=-1).reshape(b.size(0), -1).contiguous()

    projected_states_ba = projected_states_ba.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * num_v_heads // num_k_heads))
    b, a = torch.split(
        projected_states_ba,
        [num_v_heads // num_k_heads, num_v_heads // num_k_heads],
        dim=-1)
    b = b.reshape(num_actual_tokens, num_v_heads // tp_size)
    a = a.reshape(num_actual_tokens, num_v_heads // tp_size)

    split_qkvz = [
        head_k_dim,
        head_k_dim,
        num_v_heads // num_k_heads * head_v_dim,
        num_v_heads // num_k_heads * head_v_dim,
    ]
    projected_states_qkvz = projected_states_qkvz.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * head_k_dim + 2 * num_v_heads // num_k_heads * head_v_dim))
    q_split, k_split, v_split, z_split = torch.split(
        projected_states_qkvz, split_qkvz, dim=-1)
    q_split = q_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    k_split = k_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    v_split = v_split.reshape(
        num_actual_tokens,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim)
    qkv = torch.cat((q_split, k_split, v_split), dim=-1).reshape(
        num_actual_tokens,
        num_k_heads // tp_size *
        (2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim))
    z_global = z_split.reshape(num_actual_tokens, num_v_heads // tp_size,
                               head_v_dim)
    return qkv, b, a, z_global


def ref_gdn_attention_spec(
    core_attn_out,
    z,
    projected_states_qkvz,
    projected_states_ba,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    conv_state,
    ssm_state,
    conv_weights,
    conv_bias,
    activation,
    A_log,
    dt_bias,
    num_spec_decodes,
    spec_query_start_loc,
    spec_token_indx,
    spec_state_indices_tensor,
    num_accepted_tokens,
    num_actual_tokens,
    tp_size,
    reorder_input,
):
    """Spec-decode reference. Conv uses the single-cache-line sliding-window
    convention (column 0 + row offset); ssm keeps the token-indexed
    convention (per-step writeback to every column)."""
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = projected_states_qkvz.dtype
    width = conv_weights.shape[-1]
    rep = num_v_heads // num_k_heads
    K = spec_state_indices_tensor.shape[1]
    qkv_elems_size = num_k_heads // tp_size * (
        2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim)

    qkv, b, a, z_global = _extract_qkv_b_a_z(
        projected_states_qkvz, projected_states_ba, num_actual_tokens,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size,
        reorder_input)

    # Scatter z into output at the spec token positions.
    spec_indx_long = spec_token_indx.to(torch.long)
    z[spec_indx_long] = z_global[spec_indx_long]

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)
    conv_bias_f = (conv_bias.to(torch.float)
                   if conv_bias is not None else None)

    split_qkv = [
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * head_k_dim,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim,
    ]

    for n in range(num_spec_decodes):
        start = int(spec_query_start_loc[n].item())
        end = int(spec_query_start_loc[n + 1].item())
        assert end - start == K, (end - start, K)
        globals_ = spec_token_indx[start:end].to(torch.long)

        naccepted = int(num_accepted_tokens[n].item())
        init_col = max(naccepted - 1, 0)
        init_slot = int(spec_state_indices_tensor[n, init_col].item())  # ssm
        conv_slot = int(spec_state_indices_tensor[n, 0].item())  # conv, col 0
        conv_state_len = conv_state.shape[1]
        conv_init_row = init_col

        # conv1d: window = rows [init_row, init_row + width - 1), col 0
        conv_line = conv_state[conv_slot]
        prior = conv_line[conv_init_row:conv_init_row + (width - 1)].clone()
        qkv_batch = qkv[globals_]  # [K, qkv_elems]
        qkv_conv_input = torch.cat([prior, qkv_batch], dim=0)
        # Roll the line: history from init_row+1, then the K draft inputs.
        new_conv_line = torch.empty_like(conv_line)
        hist_rows = conv_state_len - K
        if hist_rows > 0:
            new_conv_line[:hist_rows] = conv_line[
                conv_init_row + 1:conv_init_row + 1 + hist_rows]
        new_conv_line[hist_rows:] = qkv_batch
        conv_state[conv_slot] = new_conv_line

        qkv_conv_in = qkv_conv_input.transpose(0, 1).unsqueeze(0).to(
            torch.float32)
        qkv_conv_out = F.conv1d(qkv_conv_in,
                                conv_weights.unsqueeze(1).to(torch.float32),
                                conv_bias_f,
                                padding=0,
                                groups=qkv_elems_size)
        qkv_conv_out = (qkv_conv_out if activation is None else
                        F.silu(qkv_conv_out)).to(dtype=dtype)
        qkv_conv_out = qkv_conv_out.transpose(-2, -1).reshape(
            K, qkv_elems_size)

        q_out, k_out, v_out = torch.split(qkv_conv_out, split_qkv, dim=-1)
        q_out = q_out.reshape(K, num_k_heads // tp_size, head_k_dim)
        k_out = k_out.reshape(K, num_k_heads // tp_size, head_k_dim)
        v_out = v_out.reshape(K, num_v_heads // tp_size, head_v_dim)

        # ---- SSM recurrence (same as non-spec, just per-step writeback) ----
        ssm_state_batch = ssm_state[init_slot].to(torch.float32).clone()

        b_batch = b[globals_].to(torch.float32)
        a_batch = a[globals_].to(torch.float32)
        beta_batch = torch.sigmoid(b_batch)
        g_batch = torch.exp(A_log_exp * softplus(a_batch + dt_bias))

        q_all = q_out.to(torch.float32)
        k_all = k_out.to(torch.float32)
        v_all = v_out.to(torch.float32)
        q_all = q_all * torch.rsqrt(q_all.pow(2).sum(-1, keepdim=True) + eps)
        k_all = k_all * torch.rsqrt(k_all.pow(2).sum(-1, keepdim=True) + eps)
        q_all = q_all * scale
        if rep > 1:
            q_all = q_all.repeat_interleave(rep, dim=1)
            k_all = k_all.repeat_interleave(rep, dim=1)

        for t in range(K):
            g_t = g_batch[t]
            beta_t = beta_batch[t]
            q_t = q_all[t]
            k_t = k_all[t]
            v_t = v_all[t]

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)
            kv_mem_t = torch.einsum("vhk,vk->vh", ssm_state_batch, k_t)
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)
            ssm_state_batch.add_(torch.einsum("vh,vk->vhk", delta_t, k_t))

            out_t = torch.einsum("vhk,vk->vh", ssm_state_batch, q_t).to(dtype)
            core_attn_out[globals_[t]] = out_t
            # Per-step ssm-state writeback to cache_indices[n, t].
            ssm_state[int(spec_state_indices_tensor[n, t].item())] = (
                ssm_state_batch.to(ssm_state.dtype))


@pytest.mark.parametrize("num_spec_decodes", NUM_SPEC_DECODES)
@pytest.mark.parametrize("num_spec_tokens", NUM_SPEC_TOKENS)
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("activation", ["silu"])
@pytest.mark.parametrize("reorder_input", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=format_tc)
@pytest.mark.parametrize("ssm_state_is_fp32", [False, True])
@torch.inference_mode()
def test_gdn_attention_mtp(num_spec_decodes, num_spec_tokens, num_k_heads,
                           head_k_dim, num_v_heads, head_v_dim, width,
                           tp_size, has_bias, activation, reorder_input,
                           dtype, ssm_state_is_fp32):
    """Pure spec-decode batch: num_prefills == num_decodes == 0,
    num_spec_decodes sequences each contributing num_spec_tokens tokens.
    Token positions are shuffled in the global buffer via spec_token_indx
    so the kernel's gather/scatter is exercised."""
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(123)
    torch.manual_seed(123)
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

    assert head_k_dim == head_v_dim
    K = num_spec_tokens
    num_spec = K - 1  # num_speculative_tokens
    num_actual_tokens = num_spec_decodes * K
    cache_batch_size = 200

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn(num_actual_tokens,
                                        mixed_qkvz_size,
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn(num_actual_tokens,
                                      mixed_ba_size,
                                      dtype=dtype,
                                      device=device)
    # Sliding-window layout: state_len = (width-1) + num_spec rows per line.
    conv_state = torch.randn(cache_batch_size,
                             (width - 1) + num_spec,
                             mixed_qkv_size,
                             dtype=dtype,
                             device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(cache_batch_size,
                            num_v_heads // tp_size,
                            head_v_dim,
                            head_k_dim,
                            dtype=ssm_state_dtype,
                            device=device)
    ref_ssm_state = ssm_state.clone()
    conv_weights = torch.randn(mixed_qkv_size,
                               width,
                               dtype=dtype,
                               device=device)
    conv_bias = (torch.randn(mixed_qkv_size, dtype=dtype, device=device)
                 if has_bias else None)
    A_log = torch.randn(num_v_heads // tp_size,
                        dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn(num_v_heads // tp_size, dtype=dtype, device=device)

    # K slots per seq: conv uses only column 0, ssm uses all K columns.
    state_slots = random.sample(range(cache_batch_size), num_spec_decodes * K)
    spec_state_indices_tensor = torch.tensor(state_slots,
                                             dtype=torch.int32,
                                             device=device).reshape(
                                                 num_spec_decodes, K)
    # Cover acceptance 0..K (0 edge case and num_accepted > 1).
    num_accepted_tokens = torch.tensor(
        [n % (K + 1) for n in range(num_spec_decodes)],
        dtype=torch.int32,
        device=device)

    # Shuffle global token positions across the K-tokens-per-seq layout.
    perm = torch.randperm(num_actual_tokens, device=device).to(torch.int32)
    spec_token_indx = perm.contiguous()
    spec_query_start_loc = (torch.arange(
        num_spec_decodes + 1, dtype=torch.int32, device=device) * K)

    core_attn_out = torch.zeros(num_actual_tokens,
                                num_v_heads // tp_size,
                                head_v_dim,
                                dtype=dtype,
                                device=device)
    z = torch.zeros_like(core_attn_out)

    intermediates = torch.ops._xpu_C.causal_conv1d_spec(
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=num_spec_decodes,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input)

    torch.ops._xpu_C.gated_delta_rule_spec(
        core_attn_out,
        *intermediates,
        num_v_heads,
        head_v_dim,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=num_spec_decodes,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.zeros_like(core_attn_out)
    ref_gdn_attention_spec(
        ref_core_attn_out,
        ref_z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_spec_decodes=num_spec_decodes,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)

    # Conv state uses a single cache line (column 0); assert that slot.
    for n in range(num_spec_decodes):
        conv_slot = int(spec_state_indices_tensor[n, 0].item())
        torch.testing.assert_close(conv_state[conv_slot],
                                   ref_conv_state[conv_slot],
                                   atol=atol,
                                   rtol=rtol)
        # All K ssm-state slots are written per-step.
        for t in range(K):
            slot = int(spec_state_indices_tensor[n, t].item())
            torch.testing.assert_close(ssm_state[slot],
                                       ref_ssm_state[slot],
                                       atol=atol,
                                       rtol=rtol)


MIXED_NUM_SPEC_DECODES = [1, 4]
MIXED_NUM_SPEC_TOKENS = [2, 3]  # num_speculative_tokens + 1


@pytest.mark.parametrize("num_spec_decodes", MIXED_NUM_SPEC_DECODES)
@pytest.mark.parametrize("num_spec_tokens", MIXED_NUM_SPEC_TOKENS)
@pytest.mark.parametrize("non_spec_num_tokens", [1, 32, 257])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("activation", ["silu"])
@pytest.mark.parametrize("mode", ["prefill", "decode", "mix_mode"])
@pytest.mark.parametrize("reorder_input", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=format_tc)
@pytest.mark.parametrize("ssm_state_is_fp32", [False, True])
@torch.inference_mode()
def test_gdn_attention_mixed_spec_non_spec(num_spec_decodes, num_spec_tokens,
                                           non_spec_num_tokens, num_k_heads,
                                           head_k_dim, num_v_heads, head_v_dim,
                                           width, tp_size, has_bias, activation,
                                           mode, reorder_input, dtype,
                                           ssm_state_is_fp32):
    """Mixed batch exercising the fused gdn_attention with BOTH a non-spec
    (prefill + decode) group AND a spec-decode group in a single call.

    Layout in the global token buffer:
      - non-spec tokens occupy the leading segment [0, non_spec_token)
        contiguously (validated with ref_gdn_attention);
      - spec tokens occupy the trailing segment [non_spec_token,
        num_actual_tokens), addressed through spec_token_indx with the
        per-seq order shuffled to exercise gather/scatter (validated with
        ref_gdn_attention_spec).
    The two groups use disjoint conv/ssm cache slots so their execution order
    inside the fused op cannot interfere.
    """
    if (os.getenv("SKIP_ACC_ERROR_KERNEL") is not None
            and os.getenv("SKIP_ACC_ERROR_KERNEL") == "1"):
        pytest.skip("skip gdn attention kernels testing on PVC.")

    device = "xpu"
    random.seed(7)
    torch.manual_seed(7)
    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

    assert head_k_dim == head_v_dim

    # ---- non-spec segment sizing (prefill / decode split) ----
    non_spec_batch = min(32, non_spec_num_tokens)
    if mode == "prefill":
        num_prefills = non_spec_batch
    elif mode == "decode":
        num_prefills = 0
        # decode means one token per sequence; only valid when tokens == batch.
        if non_spec_batch != non_spec_num_tokens:
            pytest.skip("decode mode requires non_spec_num_tokens == batch")
    else:
        num_prefills = (random.randint(1, non_spec_batch - 1)
                        if non_spec_batch > 1 else 1)
    num_decodes = non_spec_batch - num_prefills

    K = num_spec_tokens
    spec_token = num_spec_decodes * K
    non_spec_token = non_spec_num_tokens
    num_actual_tokens = non_spec_token + spec_token
    cache_batch_size = 400

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype,
                                      device=device)

    # Shared conv cache: state_len=(width-1)+num_spec. non-spec uses only the
    # leading width-1 rows; spec uses the full sliding window in column 0.
    num_spec = K - 1
    conv_state = torch.randn(
        (cache_batch_size, (width - 1) + num_spec, mixed_qkv_size),
        dtype=dtype,
        device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=ssm_state_dtype,
        device=device)
    ref_ssm_state = ssm_state.clone()

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

    A_log = torch.randn((num_v_heads // tp_size),
                        dtype=torch.float32,
                        device=device)
    dt_bias = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)

    # ---- disjoint cache slots for the two groups ----
    all_slots = random.sample(range(cache_batch_size),
                              non_spec_batch + spec_token)
    non_spec_slots = all_slots[:non_spec_batch]
    spec_slots = all_slots[non_spec_batch:]

    # ---- non-spec indexing: leading segment [0, non_spec_token) ----
    prefill_batches = simple_random_distribute(non_spec_token - num_decodes,
                                               non_spec_batch - num_decodes)
    token_batches = torch.cat([torch.ones([num_decodes]),
                               prefill_batches]).to(device)
    perm = torch.randperm(token_batches.size(0)).to(device)
    shuffled_tensor = token_batches[perm]
    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)
    has_initial_state = perm >= num_decodes
    non_spec_state_indices_tensor = torch.tensor(non_spec_slots,
                                                 device=device,
                                                 dtype=torch.int32)
    # Non-spec tokens are laid out contiguously at the front; the identity
    # token index maps kernel writes to the same rows the reference uses.
    non_spec_token_indx = torch.arange(non_spec_token,
                                       dtype=torch.int32,
                                       device=device)

    # ---- spec indexing: trailing segment [non_spec_token, num_actual_tokens)
    spec_state_indices_tensor = torch.tensor(spec_slots,
                                             dtype=torch.int32,
                                             device=device).reshape(
                                                 num_spec_decodes, K)
    num_accepted_tokens = torch.tensor(
        [n % (K + 1) for n in range(num_spec_decodes)],
        dtype=torch.int32,
        device=device)
    # Shuffle spec token positions WITHIN the trailing segment.
    spec_perm = torch.randperm(spec_token, device=device)
    spec_token_indx = (non_spec_token +
                       spec_perm).to(torch.int32).contiguous()
    spec_query_start_loc = (torch.arange(
        num_spec_decodes + 1, dtype=torch.int32, device=device) * K)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype,
        device=device,
    )
    z = torch.zeros_like(core_attn_out)

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=num_spec_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=non_spec_token_indx,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input)

    # ---- reference: run non-spec ref on the leading segment and spec ref on
    # the trailing segment, into a shared reference buffer ----
    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.zeros_like(core_attn_out)

    # Non-spec reference consumes only the leading [0, non_spec_token) rows.
    ref_gdn_attention(
        ref_core_attn_out[:non_spec_token],
        ref_z[:non_spec_token],
        projected_states_qkvz[:non_spec_token].contiguous(),
        projected_states_ba[:non_spec_token].contiguous(),
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=non_spec_token,
        tp_size=tp_size,
        reorder_input=reorder_input,
    )

    # Spec reference scatters through spec_token_indx (trailing rows).
    ref_gdn_attention_spec(
        ref_core_attn_out,
        ref_z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        num_spec_decodes=num_spec_decodes,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=reorder_input,
    )

    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)

    # non-spec conv/ssm cache slots.
    for i in range(non_spec_batch):
        slot = non_spec_state_indices_tensor[i]
        torch.testing.assert_close(conv_state[slot],
                                   ref_conv_state[slot],
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(ssm_state[slot],
                                   ref_ssm_state[slot],
                                   atol=atol,
                                   rtol=rtol)

    # spec conv (single cache line, column 0) / ssm (all K per-step slots).
    for n in range(num_spec_decodes):
        conv_slot = int(spec_state_indices_tensor[n, 0].item())
        torch.testing.assert_close(conv_state[conv_slot],
                                   ref_conv_state[conv_slot],
                                   atol=atol,
                                   rtol=rtol)
        for t in range(K):
            slot = int(spec_state_indices_tensor[n, t].item())
            torch.testing.assert_close(ssm_state[slot],
                                       ref_ssm_state[slot],
                                       atol=atol,
                                       rtol=rtol)


# GQA ratio num_v_heads/num_k_heads == 3 regression. The SLM-tiled prefill
# conv1d (chunk_causal_conv1d_tiled_xe2) reorders the z gate one feat_chunk at a
# time: its 64 work-items cover feats_per_wg = wg_size * elems_per_item = 256 z
# features in a single pass. For ratio <= 2 with head_v_dim 128 the gate width
# z_dim = head_v_dim * num_v_heads / num_k_heads is <= 256 and fits, but ratio 3
# gives z_dim = 384, so the tail features 256..383 -- the 3rd v-head of every
# k-group -- were never written, corrupting z (and thus core_attn_out) for any
# prefill of >= conv1d_tile_size (8) tokens. Reproduces with a single short
# prefill; the existing tests only cover ratio 2 (num_v_heads = 32).
RATIO3_DTYPES = [torch.float16, torch.bfloat16]
RATIO3_REORDER = [True, False]


@pytest.mark.parametrize("dtype", RATIO3_DTYPES, ids=format_tc)
@pytest.mark.parametrize("reorder_input", RATIO3_REORDER)
@torch.inference_mode()
def test_gdn_attention_gqa_ratio3_prefill(dtype, reorder_input):
    device = "xpu"
    random.seed(0)
    torch.manual_seed(0)

    num_k_heads, head_k_dim = 2, 128
    num_v_heads, head_v_dim = 6, 128  # ratio 3 -> z_dim = head_v_dim * 3 = 384
    width, tp_size = 4, 1
    activation = "silu"
    num_actual_tokens = 16  # single prefill, >= conv1d_tile_size (8)
    num_prefills, num_decodes = 1, 0
    cache_batch_size = 4

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)
    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype, device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype, device=device)
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype, device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        dtype=dtype, device=device)
    ref_ssm_state = ssm_state.clone()
    conv_weights = torch.randn((mixed_qkv_size, width), dtype=dtype,
                               device=device)
    conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)
    A_log = torch.randn((num_v_heads), dtype=torch.float32, device=device)
    dt_bias = torch.randn((num_v_heads), dtype=dtype, device=device)

    non_spec_query_start_loc = torch.tensor([0, num_actual_tokens],
                                            dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True], dtype=torch.bool, device=device)
    non_spec_state_indices_tensor = torch.tensor([0], dtype=torch.int32,
                                                 device=device)

    core_attn_out = torch.zeros((num_actual_tokens, num_v_heads, head_v_dim),
                                dtype=dtype, device=device)
    z = torch.empty_like(core_attn_out)

    torch.ops._xpu_C.gdn_attention(
        core_attn_out, z, projected_states_qkvz, projected_states_ba,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=conv_state, ssm_state=ssm_state, conv_weights=conv_weights,
        conv_bias=conv_bias, activation=activation, A_log=A_log,
        dt_bias=dt_bias, num_prefills=num_prefills, num_decodes=num_decodes,
        num_spec_decodes=0, has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=None, spec_token_indx=None,
        spec_state_indices_tensor=None, num_accepted_tokens=None,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size,
        reorder_input=reorder_input)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)
    ref_gdn_attention(
        ref_core_attn_out, ref_z, projected_states_qkvz, projected_states_ba,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=ref_conv_state, ssm_state=ref_ssm_state,
        conv_weights=conv_weights, conv_bias=conv_bias, activation=activation,
        A_log=A_log, dt_bias=dt_bias, num_prefills=num_prefills,
        num_decodes=num_decodes, has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size,
        reorder_input=reorder_input)

    atol = rtol = 5e-2
    # z (the output gate) is reordered straight from projected_states_qkvz; with
    # ratio 3 the tiled kernel dropped the 3rd v-head of each k-group.
    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(core_attn_out, ref_core_attn_out, atol=atol,
                               rtol=rtol)


# chunk_prepare_kernel v_head_id guard coverage (this PR). chunk_prepare
# derives v_head_id = total_sg_id / (total_sg_range // num_v_heads); when
# total_sg_range is not a multiple of num_v_heads the top sub-group(s) reach
# v_head_id == num_v_heads, which the guard must bound. num_v_heads = 48
# (Qwen3.6-27B, GQA ratio 3 over num_k_heads = 16) exercises that path: on
# Battlemage sm_count = 32 sub-slices and sg_range = 32, so total_sg_range =
# 1024 is not a multiple of 48. The existing tests only cover ratio 2
# (num_v_heads = 32), which always divides total_sg_range.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=format_tc)
@torch.inference_mode()
def test_chunk_prepare_vhead_oob_guard(dtype):
    device = "xpu"
    random.seed(0)
    torch.manual_seed(0)

    num_k_heads, head_k_dim = 16, 128
    num_v_heads, head_v_dim = 48, 128  # GQA ratio 3, Qwen3.6-27B
    width, tp_size = 4, 1
    activation = "silu"
    num_actual_tokens = 64  # single prefill, spans multiple chunks
    num_prefills, num_decodes = 1, 0
    cache_batch_size = 4

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype, device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype, device=device)
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype, device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        dtype=dtype, device=device)
    ref_ssm_state = ssm_state.clone()
    conv_weights = torch.randn((mixed_qkv_size, width), dtype=dtype,
                               device=device)
    conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)
    A_log = torch.randn((num_v_heads), dtype=torch.float32, device=device)
    dt_bias = torch.randn((num_v_heads), dtype=dtype, device=device)

    non_spec_query_start_loc = torch.tensor([0, num_actual_tokens],
                                            dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True], dtype=torch.bool, device=device)
    non_spec_state_indices_tensor = torch.tensor([0], dtype=torch.int32,
                                                 device=device)

    core_attn_out = torch.zeros((num_actual_tokens, num_v_heads, head_v_dim),
                                dtype=dtype, device=device)
    z = torch.empty_like(core_attn_out)

    torch.ops._xpu_C.gdn_attention(
        core_attn_out, z, projected_states_qkvz, projected_states_ba,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=conv_state, ssm_state=ssm_state, conv_weights=conv_weights,
        conv_bias=conv_bias, activation=activation, A_log=A_log,
        dt_bias=dt_bias, num_prefills=num_prefills, num_decodes=num_decodes,
        num_spec_decodes=0, has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=None, spec_token_indx=None,
        spec_state_indices_tensor=None, num_accepted_tokens=None,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size,
        reorder_input=False)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)
    ref_gdn_attention(
        ref_core_attn_out, ref_z, projected_states_qkvz, projected_states_ba,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=ref_conv_state, ssm_state=ref_ssm_state,
        conv_weights=conv_weights, conv_bias=conv_bias, activation=activation,
        A_log=A_log, dt_bias=dt_bias, num_prefills=num_prefills,
        num_decodes=num_decodes, has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size,
        reorder_input=False)

    atol = rtol = 5e-2
    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(core_attn_out, ref_core_attn_out, atol=atol,
                               rtol=rtol)


@pytest.mark.parametrize("num_actual_tokens", [63, 64, 65, 104, 107, 162])
@torch.inference_mode()
def test_qwen38_gdn_fresh_state_replay_is_exact(num_actual_tokens):
    """Repeated Qwen3.8 prefills from fresh state must be bitwise stable.

    Qwen3.8-27B uses K16/V48/D128 GDN at TP1.  The XE2 prefill kernel has a
    64-token chunk, so 63/64 bracket the single-chunk path while 65/104/162
    exercise state hand-off between chunks.  Each invocation receives the
    same inputs and independently zeroed recurrent state, matching a new
    request with prefix caching disabled.
    """
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260829)

    num_k_heads, num_v_heads = 16, 48
    head_k_dim = head_v_dim = 128
    width, tp_size = 4, 1
    cache_batch_size, state_id = 4, 3
    repeats_per_pattern = 20
    poison_patterns = (0.0, 3.25, -1.75)

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = 2 * num_v_heads
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn(
        (num_actual_tokens, mixed_qkvz_size), dtype=dtype, device=device)
    projected_states_ba = torch.randn(
        (num_actual_tokens, mixed_ba_size), dtype=dtype, device=device)
    initial_conv_state = torch.zeros(
        (cache_batch_size, width - 1, mixed_qkv_size),
        dtype=dtype,
        device=device,
    )
    initial_ssm_state = torch.zeros(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        dtype=dtype,
        device=device,
    )
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=device)
    query_start_loc = torch.tensor(
        [0, num_actual_tokens], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False], dtype=torch.bool, device=device)
    state_indices = torch.tensor([state_id], dtype=torch.int32, device=device)

    def run_once(poison, allocator_perturbation=0):
        # Rotate a small temporary allocation between invocations so the
        # fused op's internal A/w/u workspaces are not guaranteed to recycle
        # the same addresses every time.
        if allocator_perturbation:
            pressure = torch.empty(
                (allocator_perturbation, 131072), dtype=dtype, device=device)
            pressure.fill_(allocator_perturbation)
            del pressure
        conv_state = initial_conv_state.clone()
        ssm_state = initial_ssm_state.clone()
        conv_state[state_id].fill_(poison)
        ssm_state[state_id].fill_(poison)
        core_attn_out = torch.zeros(
            (num_actual_tokens, num_v_heads, head_v_dim),
            dtype=dtype,
            device=device,
        )
        z = torch.empty_like(core_attn_out)
        torch.ops._xpu_C.gdn_attention(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_state=conv_state,
            ssm_state=ssm_state,
            conv_weights=conv_weights,
            conv_bias=None,
            activation="silu",
            A_log=A_log,
            dt_bias=dt_bias,
            num_prefills=1,
            num_decodes=0,
            num_spec_decodes=0,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=state_indices,
            spec_query_start_loc=None,
            spec_token_indx=None,
            spec_state_indices_tensor=None,
            num_accepted_tokens=None,
            num_actual_tokens=num_actual_tokens,
            tp_size=tp_size,
            reorder_input=True,
        )
        torch.xpu.synchronize()
        return {
            "core_attn_out": core_attn_out.cpu(),
            "z": z.cpu(),
            "conv_state": conv_state[state_id].cpu(),
            "ssm_state": ssm_state[state_id].cpu(),
        }

    def assert_snapshot_equal(expected, actual, context):
        for name, expected_tensor in expected.items():
            actual_tensor = actual[name]
            if torch.equal(actual_tensor, expected_tensor):
                continue
            mismatch = actual_tensor != expected_tensor
            max_abs_diff = (actual_tensor.float() -
                            expected_tensor.float()).abs().max().item()
            pytest.fail(
                f"{name} changed {context} at T={num_actual_tokens}: "
                f"mismatches={mismatch.sum().item()}, "
                f"max_abs_diff={max_abs_diff}"
            )

    zero_state_expected = None
    for poison in poison_patterns:
        expected = run_once(poison)
        for repeat in range(1, repeats_per_pattern):
            actual = run_once(poison, allocator_perturbation=1 + repeat % 8)
            assert_snapshot_equal(
                expected,
                actual,
                f"on same-poison repeat {repeat}/{repeats_per_pattern - 1} "
                f"(poison={poison})",
            )
        if zero_state_expected is None:
            zero_state_expected = expected
        else:
            assert_snapshot_equal(
                zero_state_expected,
                expected,
                f"when fresh state poison changed from 0.0 to {poison}",
            )

    if num_actual_tokens <= 64:
        return

    ref_core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads, head_v_dim),
        dtype=dtype,
        device=device,
    )
    ref_z = torch.empty_like(ref_core_attn_out)
    ref_conv_state = initial_conv_state.clone()
    ref_ssm_state = initial_ssm_state.clone()
    ref_gdn_attention(
        ref_core_attn_out,
        ref_z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=None,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=1,
        num_decodes=0,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=True,
    )
    atol = rtol = 5e-2
    torch.testing.assert_close(
        zero_state_expected["core_attn_out"],
        ref_core_attn_out.cpu(),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        zero_state_expected["z"], ref_z.cpu(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        zero_state_expected["conv_state"],
        ref_conv_state[state_id].cpu(),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        zero_state_expected["ssm_state"],
        ref_ssm_state[state_id].cpu(),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "sequence_lengths,projection_groups",
    [
        pytest.param(
            (162, 107, 104, 131),
            (0, 1, 2, 3),
            id="service-short-ragged",
        ),
        pytest.param(
            (127, 4095, 4095, 127),
            (0, 1, 1, 0),
            id="service-abba-127-4095",
        ),
    ],
)
@torch.inference_mode()
def test_qwen38_gdn_fresh_state_is_batch_separable(
        sequence_lengths, projection_groups):
    """One ragged B4 prefill must equal four independent B1 prefills.

    The sequence lengths and K16/V48/D128 geometry match the service isolation
    failure. Distinct, poisoned cache slots verify that fresh-state handling
    neither aliases requests nor reads a prior occupant. The ABBA case also
    reuses identical projections for each matching fixture pair.
    """
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260830)

    num_actual_tokens = sum(sequence_lengths)
    num_k_heads, num_v_heads = 16, 48
    head_k_dim = head_v_dim = 128
    width, tp_size = 4, 1
    cache_batch_size = 5
    state_indices_list = (4, 1, 3, 2)

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim
        + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = 2 * num_v_heads
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim
        + head_v_dim * num_v_heads // num_k_heads)

    def make_projected_states(feature_size):
        projected = torch.empty(
            (num_actual_tokens, feature_size), dtype=dtype, device=device)
        group_slices = {}
        start = 0
        for length, group_id in zip(
                sequence_lengths, projection_groups, strict=True):
            end = start + length
            if group_id in group_slices:
                source_start, source_end = group_slices[group_id]
                assert source_end - source_start == length
                projected[start:end].copy_(projected[source_start:source_end])
            else:
                projected[start:end].normal_()
                group_slices[group_id] = (start, end)
            start = end
        return projected

    projected_states_qkvz = make_projected_states(mixed_qkvz_size)
    projected_states_ba = make_projected_states(mixed_ba_size)
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=device)

    initial_conv_state = torch.empty(
        (cache_batch_size, width - 1, mixed_qkv_size),
        dtype=dtype,
        device=device,
    )
    initial_ssm_state = torch.empty(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        dtype=dtype,
        device=device,
    )
    for state_id in range(cache_batch_size):
        poison = float(state_id - 2)
        initial_conv_state[state_id].fill_(poison)
        initial_ssm_state[state_id].fill_(poison)

    def run_prefill(lengths, state_ids, qkvz, ba):
        total_tokens = sum(lengths)
        query_start_loc = torch.tensor(
            [0, *list(torch.tensor(lengths).cumsum(0).tolist())],
            dtype=torch.int32,
            device=device,
        )
        state_indices = torch.tensor(
            state_ids, dtype=torch.int32, device=device)
        has_initial_state = torch.zeros(
            len(lengths), dtype=torch.bool, device=device)
        conv_state = initial_conv_state.clone()
        ssm_state = initial_ssm_state.clone()
        core_attn_out = torch.zeros(
            (total_tokens, num_v_heads, head_v_dim),
            dtype=dtype,
            device=device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops._xpu_C.gdn_attention(
            core_attn_out,
            z,
            qkvz,
            ba,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_state=conv_state,
            ssm_state=ssm_state,
            conv_weights=conv_weights,
            conv_bias=None,
            activation="silu",
            A_log=A_log,
            dt_bias=dt_bias,
            num_prefills=len(lengths),
            num_decodes=0,
            num_spec_decodes=0,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=state_indices,
            spec_query_start_loc=None,
            spec_token_indx=None,
            spec_state_indices_tensor=None,
            num_accepted_tokens=None,
            num_actual_tokens=total_tokens,
            tp_size=tp_size,
            reorder_input=True,
        )
        torch.xpu.synchronize()
        return {
            "core_attn_out": core_attn_out.cpu(),
            "z": z.cpu(),
            "conv_states": conv_state[state_indices].cpu(),
            "ssm_states": ssm_state[state_indices].cpu(),
        }

    batched = run_prefill(
        sequence_lengths,
        state_indices_list,
        projected_states_qkvz,
        projected_states_ba,
    )
    independent = {
        "core_attn_out": [],
        "z": [],
        "conv_states": [],
        "ssm_states": [],
    }
    start = 0
    for length, state_id in zip(
            sequence_lengths, state_indices_list, strict=True):
        end = start + length
        snapshot = run_prefill(
            (length, ),
            (state_id, ),
            projected_states_qkvz[start:end],
            projected_states_ba[start:end],
        )
        for name, tensors in independent.items():
            tensors.append(snapshot[name])
        start = end

    for name, tensors in independent.items():
        expected = torch.cat(tensors)
        actual = batched[name]
        if torch.equal(actual, expected):
            continue
        mismatch = actual != expected
        max_abs_diff = (
            (actual.float() - expected.float()).abs().max().item())
        pytest.fail(
            f"GDN {name} is not batch-separable: "
            f"mismatches={mismatch.sum().item()}, "
            f"max_abs_diff={max_abs_diff}")


@torch.inference_mode()
def test_qwen38_gdn_state_carry_is_exact_on_canonical_boundaries():
    """A canonical 64-token split must reproduce a one-shot T=4095 prefill.

    The service's 8192-token budget can split two concurrent 4095-token
    prompts at 3970. GDN processes prefill in 64-token chunks, so the scheduler
    fix instead stops at absolute token 3968 and carries recurrent state into
    the final 127 tokens. The non-canonical 3970+125 result is diagnostic only.
    """
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260830)

    num_actual_tokens = 4095
    num_k_heads, num_v_heads = 16, 48
    head_k_dim = head_v_dim = 128
    width, tp_size = 4, 1
    cache_batch_size, state_id = 3, 2

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim
        + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = 2 * num_v_heads
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim
        + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn(
        (num_actual_tokens, mixed_qkvz_size), dtype=dtype, device=device)
    projected_states_ba = torch.randn(
        (num_actual_tokens, mixed_ba_size), dtype=dtype, device=device)
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=device)

    initial_conv_state = torch.full(
        (cache_batch_size, width - 1, mixed_qkv_size),
        7.0,
        dtype=dtype,
        device=device,
    )
    initial_ssm_state = torch.full(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        -3.0,
        dtype=torch.float32,
        device=device,
    )
    state_indices = torch.tensor(
        [state_id], dtype=torch.int32, device=device)

    def run_partitioned(chunk_lengths):
        assert sum(chunk_lengths) == num_actual_tokens
        conv_state = initial_conv_state.clone()
        ssm_state = initial_ssm_state.clone()
        core_attn_out = torch.empty(
            (num_actual_tokens, num_v_heads, head_v_dim),
            dtype=dtype,
            device=device,
        )
        z = torch.empty_like(core_attn_out)

        start = 0
        for chunk_index, chunk_length in enumerate(chunk_lengths):
            end = start + chunk_length
            query_start_loc = torch.tensor(
                [0, chunk_length], dtype=torch.int32, device=device)
            has_initial_state = torch.tensor(
                [chunk_index > 0], dtype=torch.bool, device=device)
            torch.ops._xpu_C.gdn_attention(
                core_attn_out[start:end],
                z[start:end],
                projected_states_qkvz[start:end],
                projected_states_ba[start:end],
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                conv_bias=None,
                activation="silu",
                A_log=A_log,
                dt_bias=dt_bias,
                num_prefills=1,
                num_decodes=0,
                num_spec_decodes=0,
                has_initial_state=has_initial_state,
                non_spec_query_start_loc=query_start_loc,
                non_spec_token_indx=None,
                non_spec_state_indices_tensor=state_indices,
                spec_query_start_loc=None,
                spec_token_indx=None,
                spec_state_indices_tensor=None,
                num_accepted_tokens=None,
                num_actual_tokens=chunk_length,
                tp_size=tp_size,
                reorder_input=True,
            )
            start = end

        torch.xpu.synchronize()
        return {
            "core_attn_out": core_attn_out.cpu(),
            "z": z.cpu(),
            "conv_state": conv_state[state_id].cpu(),
            "ssm_state": ssm_state[state_id].cpu(),
        }

    def assert_snapshot_equal(expected, actual, partition):
        for name, expected_tensor in expected.items():
            actual_tensor = actual[name]
            if torch.equal(actual_tensor, expected_tensor):
                continue
            mismatch = actual_tensor != expected_tensor
            max_abs_diff = (
                (actual_tensor.float() - expected_tensor.float())
                .abs()
                .max()
                .item()
            )
            pytest.fail(
                f"GDN {name} changed for partition {partition}: "
                f"mismatches={mismatch.sum().item()}, "
                f"max_abs_diff={max_abs_diff}")

    one_shot = run_partitioned((num_actual_tokens, ))
    canonical = run_partitioned((3968, 127))
    assert_snapshot_equal(one_shot, canonical, "3968+127")
    del canonical

    noncanonical = run_partitioned((3970, 125))
    for name, expected_tensor in one_shot.items():
        actual_tensor = noncanonical[name]
        mismatch_count = (actual_tensor != expected_tensor).sum().item()
        max_abs_diff = (
            (actual_tensor.float() - expected_tensor.float())
            .abs()
            .max()
            .item()
        )
        print(
            f"noncanonical 3970+125 {name}: mismatches={mismatch_count}, "
            f"max_abs_diff={max_abs_diff}")


@torch.inference_mode()
def test_qwen38_gdn_cached_decode_is_exact_in_mixed_prefill():
    """A cached decode must be independent of a fresh prefill peer.

    Production ``split_decodes_and_prefills`` packs the one-token decode first
    and the 127-token fresh prefill second. With no speculative requests,
    metadata keeps that order without token-index indirection. The mixed call
    selects the prefill-capable XPU route because ``num_prefills > 0``.
    """
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260830)

    num_k_heads, num_v_heads = 16, 48
    head_k_dim = head_v_dim = 128
    width, tp_size = 4, 1
    cache_batch_size = 5
    decode_state_id, prefill_state_id = 3, 1
    prefill_tokens = 127

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim
        + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = 2 * num_v_heads
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim
        + head_v_dim * num_v_heads // num_k_heads)

    num_mixed_tokens = 1 + prefill_tokens
    projected_states_qkvz = torch.randn(
        (num_mixed_tokens, mixed_qkvz_size), dtype=dtype, device=device)
    projected_states_ba = torch.randn(
        (num_mixed_tokens, mixed_ba_size), dtype=dtype, device=device)
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=device)

    initial_conv_state = torch.randn(
        (cache_batch_size, width - 1, mixed_qkv_size),
        dtype=dtype,
        device=device,
    )
    initial_ssm_state = torch.randn(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        dtype=torch.float32,
        device=device,
    )

    def run_non_spec(
            qkvz, ba, query_start_loc, state_ids, has_initial_state,
            num_prefills, num_decodes):
        num_actual_tokens = qkvz.shape[0]
        conv_state = initial_conv_state.clone()
        ssm_state = initial_ssm_state.clone()
        core_attn_out = torch.empty(
            (num_actual_tokens, num_v_heads, head_v_dim),
            dtype=dtype,
            device=device,
        )
        z = torch.empty_like(core_attn_out)
        query_start_loc_tensor = torch.tensor(
            query_start_loc, dtype=torch.int32, device=device)
        state_indices = torch.tensor(
            state_ids, dtype=torch.int32, device=device)
        has_initial_state_tensor = torch.tensor(
            has_initial_state, dtype=torch.bool, device=device)

        torch.ops._xpu_C.gdn_attention(
            core_attn_out,
            z,
            qkvz,
            ba,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_state=conv_state,
            ssm_state=ssm_state,
            conv_weights=conv_weights,
            conv_bias=None,
            activation="silu",
            A_log=A_log,
            dt_bias=dt_bias,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_spec_decodes=0,
            has_initial_state=has_initial_state_tensor,
            non_spec_query_start_loc=query_start_loc_tensor,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=state_indices,
            spec_query_start_loc=None,
            spec_token_indx=None,
            spec_state_indices_tensor=None,
            num_accepted_tokens=None,
            num_actual_tokens=num_actual_tokens,
            tp_size=tp_size,
            reorder_input=True,
        )
        torch.xpu.synchronize()
        return {
            "core_attn_out": core_attn_out[0].cpu(),
            "z": z[0].cpu(),
            "conv_state": conv_state[decode_state_id].cpu(),
            "ssm_state": ssm_state[decode_state_id].cpu(),
        }

    decode_alone = run_non_spec(
        projected_states_qkvz[:1],
        projected_states_ba[:1],
        (0, 1),
        (decode_state_id, ),
        (True, ),
        num_prefills=0,
        num_decodes=1,
    )
    decode_with_prefill = run_non_spec(
        projected_states_qkvz,
        projected_states_ba,
        (0, 1, num_mixed_tokens),
        (decode_state_id, prefill_state_id),
        (True, False),
        num_prefills=1,
        num_decodes=1,
    )

    for name, expected in decode_alone.items():
        actual = decode_with_prefill[name]
        if torch.equal(actual, expected):
            continue
        mismatch = actual != expected
        max_abs_diff = (
            (actual.float() - expected.float()).abs().max().item())
        pytest.fail(
            f"cached decode {name} changed in mixed prefill route: "
            f"mismatches={mismatch.sum().item()}, "
            f"max_abs_diff={max_abs_diff}")


@torch.inference_mode()
def test_qwen38_gdn_t107_async_chain_is_exact():
    """Queued T=107 fresh-state prefills must remain bitwise repeatable.

    Unlike the synchronized replay test above, this queues independent fused
    calls without a host barrier between them.  This catches temporary-storage
    or event-lifetime bugs that only appear while several service-shaped GDN
    prefills are outstanding on the XPU queue.
    """
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260829)

    num_actual_tokens = 107
    num_k_heads, num_v_heads = 16, 48
    head_k_dim = head_v_dim = 128
    width, tp_size = 4, 1
    cache_batch_size, state_id = 4, 3
    repeats = 24

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = 2 * num_v_heads
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)
    gdn_output_size = num_v_heads * head_v_dim

    hidden_size = 5120
    group_size = 128

    def packed_int4_weight(k, n):
        packed_bytes = torch.randint(
            -128,
            128,
            (k * n // 2,),
            dtype=torch.int8,
            device=device,
        )
        weight = packed_bytes.view(torch.int32).reshape(k // 8, n)
        return weight.transpose(0, 1).contiguous().transpose(0, 1)

    hidden_states = torch.randn(
        (num_actual_tokens, hidden_size), dtype=dtype, device=device)
    qkvz_weight = packed_int4_weight(hidden_size, mixed_qkvz_size)
    qkvz_scales = torch.rand(
        (hidden_size // group_size, mixed_qkvz_size),
        dtype=dtype,
        device=device,
    ).mul_(0.05)
    ba_weight = packed_int4_weight(hidden_size, mixed_ba_size)
    ba_scales = torch.rand(
        (hidden_size // group_size, mixed_ba_size),
        dtype=dtype,
        device=device,
    ).mul_(0.05)
    out_weight = packed_int4_weight(gdn_output_size, hidden_size)
    out_scales = torch.rand(
        (gdn_output_size // group_size, hidden_size),
        dtype=dtype,
        device=device,
    ).mul_(0.05)
    zero_points = torch.tensor([8], dtype=torch.int8, device=device)
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=device)
    query_start_loc = torch.tensor(
        [0, num_actual_tokens], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False], dtype=torch.bool, device=device)
    state_indices = torch.tensor([state_id], dtype=torch.int32, device=device)

    # Model execution may use a non-default stream. Finish creating the shared
    # immutable inputs before beginning the deliberately unsynchronized chain.
    torch.xpu.synchronize()
    gdn_stream = torch.xpu.Stream()
    runs = []
    allocation_anchors = []
    for repeat in range(repeats):
        # Keep differently sized allocations live across the chain so each
        # call gets distinct state/output placement.
        anchor = torch.empty(
            ((repeat + 1) * 131071,), dtype=torch.uint8, device=device)
        anchor.fill_(repeat + 1)
        allocation_anchors.append(anchor)

        with torch.xpu.stream(gdn_stream):
            projected_states_qkvz = torch.ops._xpu_C.int4_gemm_w4a16(
                hidden_states,
                qkvz_weight,
                None,
                qkvz_scales,
                zero_points,
                group_size,
                None,
            )
            projected_states_ba = torch.ops._xpu_C.int4_gemm_w4a16(
                hidden_states,
                ba_weight,
                None,
                ba_scales,
                zero_points,
                group_size,
                None,
            )
            conv_state = torch.zeros(
                (cache_batch_size, width - 1, mixed_qkv_size),
                dtype=dtype,
                device=device,
            )
            ssm_state = torch.zeros(
                (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
                dtype=dtype,
                device=device,
            )
            core_attn_out = torch.empty(
                (num_actual_tokens, num_v_heads, head_v_dim),
                dtype=dtype,
                device=device,
            )
            z = torch.empty_like(core_attn_out)

            torch.ops._xpu_C.gdn_attention(
                core_attn_out,
                z,
                projected_states_qkvz,
                projected_states_ba,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                conv_bias=None,
                activation="silu",
                A_log=A_log,
                dt_bias=dt_bias,
                num_prefills=1,
                num_decodes=0,
                num_spec_decodes=0,
                has_initial_state=has_initial_state,
                non_spec_query_start_loc=query_start_loc,
                non_spec_token_indx=None,
                non_spec_state_indices_tensor=state_indices,
                spec_query_start_loc=None,
                spec_token_indx=None,
                spec_state_indices_tensor=None,
                num_accepted_tokens=None,
                num_actual_tokens=num_actual_tokens,
                tp_size=tp_size,
                reorder_input=True,
            )
            gated_output = core_attn_out * F.silu(z)
            layer_output = torch.ops._xpu_C.int4_gemm_w4a16(
                gated_output.reshape(num_actual_tokens, gdn_output_size),
                out_weight,
                None,
                out_scales,
                zero_points,
                group_size,
                None,
            )

        # The fused wrapper releases q/k/v/b/a plus its A/w/u workspaces as it
        # returns, even though their kernels are asynchronous. Immediately
        # allocate and write every exact T=107 shape so the caching allocator
        # cannot avoid this test merely because generic pressure landed in a
        # different size bin. Keep the replacements live through the chain.
        padded_tokens = num_actual_tokens + 63
        released_shape_churn = [
            torch.empty(
                (padded_tokens, num_k_heads, head_k_dim),
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                (padded_tokens, num_k_heads, head_k_dim),
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                (padded_tokens, num_v_heads, head_v_dim),
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                (num_v_heads, padded_tokens),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty(
                (num_v_heads, padded_tokens),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty(
                (num_v_heads, padded_tokens, 64),
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                (num_v_heads, padded_tokens, head_k_dim),
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                (num_v_heads, padded_tokens, head_v_dim),
                dtype=dtype,
                device=device,
            ),
        ]
        for tensor in released_shape_churn:
            tensor.fill_(repeat + 17)
        allocation_anchors.extend(released_shape_churn)

        runs.append({
            "core_attn_out": core_attn_out,
            "z": z,
            "conv_state": conv_state[state_id],
            "ssm_state": ssm_state[state_id],
            "layer_output": layer_output,
        })

        trailing_pressure = torch.empty(
            ((repeat % 4 + 1) * 65537,), dtype=torch.uint8, device=device)
        trailing_pressure.fill_(repeats - repeat)
        del trailing_pressure

    # This is deliberately the only synchronization in the enqueue chain.
    torch.xpu.synchronize()

    expected = {name: tensor.cpu() for name, tensor in runs[0].items()}
    for repeat, run in enumerate(runs[1:], start=1):
        for name, expected_tensor in expected.items():
            actual_tensor = run[name].cpu()
            if torch.equal(actual_tensor, expected_tensor):
                continue
            mismatch = actual_tensor != expected_tensor
            max_abs_diff = (
                actual_tensor.float() - expected_tensor.float()
            ).abs().max().item()
            pytest.fail(
                f"{name} changed on queued T=107 repeat {repeat}/"
                f"{repeats - 1}: mismatches={mismatch.sum().item()}, "
                f"max_abs_diff={max_abs_diff}"
            )


# chunk_update_states_kernel conv_elems guard coverage. When total conv_elems
# is not a multiple of elems_per_group (1024), the last work-group is 
# over-provisioned and requires an upper bound check to prevent out-of-bounds
# memory writes. Qwen3.6-27B at TP=4 (local num_k_heads=4, num_v_heads=12) 
# forces conv_elems = 2560. 2560 % 1024 != 0, exercising this guard.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=format_tc)
@torch.inference_mode()
def test_causal_conv1d_conv_elems_oob_guard(dtype):
    device = "xpu"
    random.seed(0)
    torch.manual_seed(0)

    # Qwen-27B global heads: K=16, V=48. 
    # At TP=4, local heads are K=4 and V=12. 
    tp_size = 4
    num_k_heads = 16
    num_v_heads = 48
    # local heads will be K=4, V=12 internally
    head_k_dim = 128
    head_v_dim = 128 
    
    width = 4
    activation = "silu"
    num_actual_tokens = 64  # single prefill
    num_prefills, num_decodes = 1, 0
    cache_batch_size = 4

    local_num_k_heads = num_k_heads // tp_size
    local_num_v_heads = num_v_heads // tp_size
    
    mixed_qkvz_size = local_num_k_heads * (
        2 * head_k_dim + 2 * head_v_dim * \
        local_num_v_heads // local_num_k_heads)
    mixed_ba_size = local_num_k_heads * (
        2 * local_num_v_heads // local_num_k_heads)
    
    # conv_elems will equal mixed_qkv_size (2560 here)
    mixed_qkv_size = local_num_k_heads * (
        2 * head_k_dim + head_v_dim * local_num_v_heads // local_num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype, device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype, device=device)
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype, device=device)
    ref_conv_state = conv_state.clone()
    ssm_state = torch.randn(
        (cache_batch_size, local_num_v_heads, head_v_dim, head_k_dim),
        dtype=dtype, device=device)
    ref_ssm_state = ssm_state.clone()
        
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device)
    conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)
    A_log = torch.randn((local_num_v_heads), dtype=torch.float32, device=device)
    dt_bias = torch.randn((local_num_v_heads), dtype=dtype, device=device)

    non_spec_query_start_loc = torch.tensor([0, num_actual_tokens],
                                            dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True], dtype=torch.bool, device=device)
    non_spec_state_indices_tensor = torch.tensor(
        [0], dtype=torch.int32, device=device)

    core_attn_out = torch.zeros(
        (num_actual_tokens, local_num_v_heads, head_v_dim),
                                dtype=dtype, device=device)
    z = torch.empty_like(core_attn_out)

    torch.ops._xpu_C.gdn_attention(
        core_attn_out, z, projected_states_qkvz, projected_states_ba,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=conv_state, ssm_state=ssm_state, conv_weights=conv_weights,
        conv_bias=conv_bias, activation=activation, A_log=A_log,
        dt_bias=dt_bias, num_prefills=num_prefills, num_decodes=num_decodes,
        num_spec_decodes=0, has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=None, spec_token_indx=None,
        spec_state_indices_tensor=None, num_accepted_tokens=None,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size,
        reorder_input=False)

    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)
    ref_gdn_attention(
        ref_core_attn_out, ref_z, projected_states_qkvz, projected_states_ba,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=ref_conv_state, ssm_state=ref_ssm_state,
        conv_weights=conv_weights, conv_bias=conv_bias, activation=activation,
        A_log=A_log, dt_bias=dt_bias, num_prefills=num_prefills,
        num_decodes=num_decodes, has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size,
        reorder_input=False)

    atol = rtol = 5e-2
    torch.testing.assert_close(z, ref_z, atol=atol, rtol=rtol)
    torch.testing.assert_close(core_attn_out, ref_core_attn_out, atol=atol,
                               rtol=rtol)
