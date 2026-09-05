# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize("num_tokens", [162])
@torch.inference_mode()
def test_qwen38_fresh_prefill_replay_is_exact(num_tokens):
    """Fresh Qwen3.8 prefills must ignore recycled convolution state.

    The prompt length exceeds the disabled tiled kernel's eight-token
    threshold. Repeating the call with identical inputs and changing only the
    stale cache contents must leave every output and the final cache bitwise
    identical.
    """
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260829)

    num_k_heads, num_v_heads = 16, 48
    head_k_dim = head_v_dim = 128
    width, tp_size = 4, 1
    state_id = 1

    mixed_qkvz_size = num_k_heads * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads
    )
    mixed_ba_size = 2 * num_v_heads
    mixed_qkv_size = num_k_heads * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads
    )

    projected_states_qkvz = torch.randn(
        (num_tokens, mixed_qkvz_size), dtype=dtype, device=device
    )
    projected_states_ba = torch.randn(
        (num_tokens, mixed_ba_size), dtype=dtype, device=device
    )
    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device
    )
    query_start_loc = torch.tensor(
        [0, num_tokens], dtype=torch.int32, device=device
    )
    state_indices = torch.tensor([state_id], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False], dtype=torch.bool, device=device)

    def run_once(poison):
        conv_state = torch.full(
            (2, width - 1, mixed_qkv_size),
            poison,
            dtype=dtype,
            device=device,
        )
        z = torch.empty(
            (num_tokens, num_v_heads, head_v_dim),
            dtype=dtype,
            device=device,
        )
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
            conv_bias=None,
            activation="silu",
            num_prefills=1,
            num_decodes=0,
            num_spec_decodes=0,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=state_indices,
            num_actual_tokens=num_tokens,
            tp_size=tp_size,
            reorder_input=True,
        )
        torch.xpu.synchronize()
        return tuple(t.cpu() for t in intermediates) + (
            z.cpu(),
            conv_state[state_id].cpu(),
        )

    expected = run_once(0.0)
    for poison in (0.0, 3.25, -1.75):
        for _ in range(3):
            actual = run_once(poison)
            for expected_tensor, actual_tensor in zip(expected, actual):
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, atol=0, rtol=0
                )
