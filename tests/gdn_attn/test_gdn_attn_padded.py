# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for vllm-project/vllm-xpu-kernels#320.

The op was previously asserting
``core_attn_out.size(0) == num_actual_tokens`` (and the same for
``projected_states_qkvz`` / ``projected_states_ba``).  vLLM's PIECEWISE
cudagraph capture pads the leading dim to the captured graph size while
``attn_metadata.num_actual_tokens`` stays at the real (unpadded) count,
so any decode batch smaller than the captured size aborted the engine.

After the kernel-side fix the op should:

* accept ``size(0) >= num_actual_tokens``;
* write the active prefix [0:num_actual_tokens) into the caller's
  buffer;
* leave the padded tail (rows ``[num_actual_tokens:size(0))``) untouched
  at its zero-init.
"""

import random

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize("lengths", [(2,), (1,), (3, 2, 1)])
@pytest.mark.parametrize("accepted", [1, 3])
@pytest.mark.parametrize("reorder_input", [False, True])
@pytest.mark.parametrize("shape", [(4, 8, 32), (16, 48, 128)])
@torch.inference_mode()
def test_gdn_shortened_mtp_preserves_rollback(
    lengths, accepted, reorder_input, shape
):
    """Short/ragged steps equal full-step prefixes, including continuation.

    The state capacity stays three even when only one/two queries are run:
    the previous iteration may have accepted all three. Compare both stages
    against full-width execution, then continue from every accepted prefix.
    Unscheduled output rows, rollback slots and adjacent layers are guards.
    """
    torch.manual_seed(20260907)
    device, dtype = "xpu", torch.bfloat16
    kh, vh, dim = shape
    batch, capacity, width = len(lengths), 3, 4
    total = batch * capacity
    qkv_size = (2 * kh + vh) * dim
    state_len = width - 2 + capacity
    slots = (
        torch.randperm(total, device=device)
        .to(torch.int32)
        .reshape(batch, capacity)
    )
    conv_storage = torch.randn(
        total, 3, state_len, qkv_size, dtype=dtype, device=device
    )
    ref_conv_storage = conv_storage.clone()
    initial_conv = conv_storage.clone()
    conv, ref_conv = conv_storage[:, 1], ref_conv_storage[:, 1]
    ssm = torch.randn(total, vh, dim, dim, device=device)
    ref_ssm, initial_ssm = ssm.clone(), ssm.clone()
    weights = torch.randn(qkv_size, width, dtype=dtype, device=device)
    bias = torch.randn(qkv_size, dtype=dtype, device=device)
    a_log = torch.randn(vh, device=device)
    dt_bias = torch.randn(vh, dtype=dtype, device=device)

    def run(conv_state, ssm_state, query_lengths, naccepted, qkvz, ba):
        starts = [0]
        indices = []
        for n, length in enumerate(query_lengths):
            starts.append(starts[-1] + length)
            indices.extend(range(n * capacity, n * capacity + length))
        indices = torch.tensor(indices, dtype=torch.int32, device=device)
        out = torch.full((total + 2, vh, dim), 17, dtype=dtype, device=device)
        z = out.clone()
        torch.ops._xpu_C.gdn_attention(
            out,
            z,
            qkvz,
            ba,
            kh,
            vh,
            dim,
            dim,
            conv_state=conv_state,
            ssm_state=ssm_state,
            conv_weights=weights,
            conv_bias=bias,
            activation="silu",
            A_log=a_log,
            dt_bias=dt_bias,
            num_prefills=0,
            num_decodes=0,
            num_spec_decodes=batch,
            has_initial_state=None,
            non_spec_query_start_loc=None,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=None,
            spec_query_start_loc=torch.tensor(
                starts, dtype=torch.int32, device=device
            ),
            spec_token_indx=indices,
            spec_state_indices_tensor=slots,
            num_accepted_tokens=torch.tensor(
                naccepted, dtype=torch.int32, device=device
            ),
            num_actual_tokens=total,
            tp_size=1,
            reorder_input=reorder_input,
            split_mixed_non_spec=False,
        )
        return out, z, indices.long()

    def inputs():
        return (
            torch.randn(
                total, (2 * kh + 2 * vh) * dim, dtype=dtype, device=device
            ),
            torch.randn(total, 2 * vh, dtype=dtype, device=device),
        )

    qkvz, ba = inputs()
    expected, expected_z, _ = run(
        ref_conv, ref_ssm, [capacity] * batch, [accepted] * batch, qkvz, ba
    )
    actual, actual_z, active = run(
        conv, ssm, lengths, [accepted] * batch, qkvz, ba
    )
    torch.testing.assert_close(actual[active], expected[active], atol=0, rtol=0)
    torch.testing.assert_close(
        actual_z[active], expected_z[active], atol=0, rtol=0
    )
    inactive = torch.ones(total + 2, dtype=torch.bool, device=device)
    inactive[active] = False
    assert torch.all(actual[inactive] == 17)
    assert torch.all(actual_z[inactive] == 17)
    for n, length in enumerate(lengths):
        conv_slot = int(slots[n, 0])
        valid_rows = width - 2 + length
        torch.testing.assert_close(
            conv[conv_slot, :valid_rows],
            ref_conv[conv_slot, :valid_rows],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            conv[conv_slot, valid_rows:],
            initial_conv[conv_slot, 1, valid_rows:],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            ssm[slots[n, :length].long()],
            ref_ssm[slots[n, :length].long()],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            ssm[slots[n, length:].long()],
            initial_ssm[slots[n, length:].long()],
            atol=0,
            rtol=0,
        )
    torch.testing.assert_close(
        conv_storage[:, (0, 2)], initial_conv[:, (0, 2)], atol=0, rtol=0
    )

    # A shortened step must leave the next full step's history correct too.
    # Exercise every possible acceptance count, not only full acceptance.
    for accepted_next in range(1, max(lengths) + 1):
        counts = [min(accepted_next, length) for length in lengths]
        qkvz, ba = inputs()
        next_conv, next_ssm = conv.clone(), ssm.clone()
        next_ref_conv, next_ref_ssm = ref_conv.clone(), ref_ssm.clone()
        actual, actual_z, _ = run(
            next_conv, next_ssm, [capacity] * batch, counts, qkvz, ba
        )
        expected, expected_z, _ = run(
            next_ref_conv, next_ref_ssm, [capacity] * batch, counts, qkvz, ba
        )
        for result, reference in [
            (actual, expected),
            (actual_z, expected_z),
            (next_conv, next_ref_conv),
            (next_ssm, next_ref_ssm),
        ]:
            torch.testing.assert_close(result, reference, atol=0, rtol=0)


def _build_inputs(num_actual_tokens, padded_size, dtype, device):
    """Allocate a minimal decode-only call shape with a padded leading dim."""
    num_k_heads = 1
    num_v_heads = 1
    head_k_dim = 32
    head_v_dim = 32
    width = 2
    tp_size = 1
    cache_batch_size = 16

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    qkvz_padded = torch.randn(padded_size, mixed_qkvz_size,
                              dtype=dtype, device=device)
    ba_padded = torch.randn(padded_size, mixed_ba_size,
                            dtype=dtype, device=device)
    # Zero out the padded tail so the kernel can't accidentally pick up
    # uninitialized values from the dummy padding region.
    qkvz_padded[num_actual_tokens:].zero_()
    ba_padded[num_actual_tokens:].zero_()

    conv_state = torch.randn(cache_batch_size, width - 1, mixed_qkv_size,
                             dtype=dtype, device=device)
    ssm_state = torch.randn(cache_batch_size, num_v_heads // tp_size,
                            head_v_dim, head_k_dim,
                            dtype=dtype, device=device)
    conv_weights = torch.randn(mixed_qkv_size, width,
                               dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads // tp_size,
                        dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads // tp_size,
                          dtype=dtype, device=device)

    # Decode-only: one token per batch entry, num_prefills == 0.
    num_decodes = num_actual_tokens
    num_prefills = 0
    batch_size = num_decodes
    non_spec_query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32,
                                            device=device)
    has_initial_state = torch.ones(batch_size, dtype=torch.bool, device=device)
    non_spec_state_indices_tensor = torch.tensor(
        random.sample(range(cache_batch_size), batch_size),
        device=device, dtype=torch.int32)

    return dict(
        projected_states_qkvz=qkvz_padded,
        projected_states_ba=ba_padded,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=None,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=False,
        head_v_dim_for_alloc=head_v_dim,
        num_v_heads_for_alloc=num_v_heads,
    )


@pytest.mark.parametrize("num_actual_tokens,padded_size", [
    (1, 4),
    (3, 4),
    (5, 8),
])
@torch.inference_mode()
def test_gdn_attention_accepts_padded_leading_dim(num_actual_tokens,
                                                  padded_size):
    """Padded inputs should not crash, and the active prefix should match
    the result of the same call made with exact-size inputs."""
    device = "xpu"
    random.seed(0)
    torch.manual_seed(0)
    dtype = torch.bfloat16

    args = _build_inputs(num_actual_tokens, padded_size, dtype, device)
    head_v = args.pop("head_v_dim_for_alloc")
    nvh = args.pop("num_v_heads_for_alloc")

    core_attn_out_padded = torch.zeros(padded_size, nvh, head_v,
                                       dtype=dtype, device=device)
    z_padded = torch.empty_like(core_attn_out_padded)
    z_padded[num_actual_tokens:].zero_()

    # Clone storage for the reference run so the two calls are independent.
    args_padded = dict(args)
    args_padded.update(
        projected_states_qkvz=args["projected_states_qkvz"].clone(),
        projected_states_ba=args["projected_states_ba"].clone(),
        conv_state=args["conv_state"].clone(),
        ssm_state=args["ssm_state"].clone(),
    )

    intermediates_padded = torch.ops._xpu_C.causal_conv1d_non_spec(
        z_padded,
        args_padded["projected_states_qkvz"],
        args_padded["projected_states_ba"],
        args_padded["num_k_heads"],
        args_padded["num_v_heads"],
        args_padded["head_k_dim"],
        args_padded["head_v_dim"],
        conv_state=args_padded["conv_state"],
        conv_weights=args_padded["conv_weights"],
        conv_bias=args_padded["conv_bias"],
        activation=args_padded["activation"],
        num_prefills=args_padded["num_prefills"],
        num_decodes=args_padded["num_decodes"],
        num_spec_decodes=0,
        has_initial_state=args_padded["has_initial_state"],
        non_spec_query_start_loc=args_padded["non_spec_query_start_loc"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=args_padded[
            "non_spec_state_indices_tensor"],
        num_actual_tokens=args_padded["num_actual_tokens"],
        tp_size=args_padded["tp_size"],
        reorder_input=args_padded["reorder_input"],
    )

    torch.ops._xpu_C.gated_delta_rule_non_spec(
        core_attn_out_padded,
        *intermediates_padded,
        args_padded["num_v_heads"],
        args_padded["head_v_dim"],
        A_log=args_padded["A_log"],
        dt_bias=args_padded["dt_bias"],
        ssm_state=args_padded["ssm_state"],
        num_prefills=args_padded["num_prefills"],
        num_decodes=args_padded["num_decodes"],
        num_spec_decodes=0,
        has_initial_state=args_padded["has_initial_state"],
        non_spec_query_start_loc=args_padded["non_spec_query_start_loc"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=args_padded[
            "non_spec_state_indices_tensor"],
        num_actual_tokens=args_padded["num_actual_tokens"],
        tp_size=args_padded["tp_size"],
    )

    # Reference call with exact-size tensors. Note the kernel mutates
    # projected_states_qkvz / projected_states_ba internally (caller passes
    # them as const refs but inner kernels read them) — so we use the same
    # un-cloned exact-size slice we'd pass under the vLLM workaround in
    # PR #42383.
    core_attn_out_ref = torch.zeros(num_actual_tokens, nvh, head_v,
                                    dtype=dtype, device=device)
    z_ref = torch.empty_like(core_attn_out_ref)

    intermediates_ref = torch.ops._xpu_C.causal_conv1d_non_spec(
        z_ref,
        args["projected_states_qkvz"][:num_actual_tokens].contiguous(),
        args["projected_states_ba"][:num_actual_tokens].contiguous(),
        args["num_k_heads"],
        args["num_v_heads"],
        args["head_k_dim"],
        args["head_v_dim"],
        conv_state=args["conv_state"],
        conv_weights=args["conv_weights"],
        conv_bias=args["conv_bias"],
        activation=args["activation"],
        num_prefills=args["num_prefills"],
        num_decodes=args["num_decodes"],
        num_spec_decodes=0,
        has_initial_state=args["has_initial_state"],
        non_spec_query_start_loc=args["non_spec_query_start_loc"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=args["non_spec_state_indices_tensor"],
        num_actual_tokens=args["num_actual_tokens"],
        tp_size=args["tp_size"],
        reorder_input=args["reorder_input"],
    )

    torch.ops._xpu_C.gated_delta_rule_non_spec(
        core_attn_out_ref,
        *intermediates_ref,
        args["num_v_heads"],
        args["head_v_dim"],
        A_log=args["A_log"],
        dt_bias=args["dt_bias"],
        ssm_state=args["ssm_state"],
        num_prefills=args["num_prefills"],
        num_decodes=args["num_decodes"],
        num_spec_decodes=0,
        has_initial_state=args["has_initial_state"],
        non_spec_query_start_loc=args["non_spec_query_start_loc"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=args["non_spec_state_indices_tensor"],
        num_actual_tokens=args["num_actual_tokens"],
        tp_size=args["tp_size"],
    )

    torch.testing.assert_close(
        core_attn_out_padded[:num_actual_tokens],
        core_attn_out_ref,
        atol=5e-2, rtol=5e-2,
    )
    torch.testing.assert_close(
        z_padded[:num_actual_tokens],
        z_ref,
        atol=5e-2, rtol=5e-2,
    )

    # The padded tail must remain at its zero init.
    tail = core_attn_out_padded[num_actual_tokens:]
    assert torch.all(tail == 0), (
        f"padded tail of core_attn_out was written by the kernel: "
        f"max abs = {tail.abs().max().item():.3e}")
    z_tail = z_padded[num_actual_tokens:]
    assert torch.all(z_tail == 0), (
        f"padded tail of z was written by the kernel: "
        f"max abs = {z_tail.abs().max().item():.3e}")
