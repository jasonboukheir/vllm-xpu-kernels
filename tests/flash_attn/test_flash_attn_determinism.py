# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

_DTYPE = torch.bfloat16
_HEAD_DIM = 256
_NUM_QUERY_HEADS = 24
_NUM_KV_HEADS = 4
_PAGE_SIZE = 64
_REPEATS = 8


def _make_qkv(seq_len: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(seq_len,
                    _NUM_QUERY_HEADS,
                    _HEAD_DIM,
                    device="xpu",
                    dtype=_DTYPE),
        torch.randn(seq_len,
                    _NUM_KV_HEADS,
                    _HEAD_DIM,
                    device="xpu",
                    dtype=_DTYPE),
        torch.randn(seq_len,
                    _NUM_KV_HEADS,
                    _HEAD_DIM,
                    device="xpu",
                    dtype=_DTYPE),
    )


def _run_dense_prefill(q: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor) -> torch.Tensor:
    seq_len = q.shape[0]
    cu_seqlens = torch.tensor([0, seq_len],
                              device="xpu",
                              dtype=torch.int32)
    output = torch.empty_like(q)
    return flash_attn_varlen_func(
        q,
        k,
        v,
        seq_len,
        cu_seqlens,
        seq_len,
        cu_seqlens_k=cu_seqlens,
        softmax_scale=_HEAD_DIM**-0.5,
        causal=True,
        out=output,
    )


def _make_paged_kv(seq_len: int) -> tuple[torch.Tensor, ...]:
    num_pages = (seq_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    key_cache = torch.randn(num_pages,
                            _PAGE_SIZE,
                            _NUM_KV_HEADS,
                            _HEAD_DIM,
                            device="xpu",
                            dtype=_DTYPE)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_pages,
                               device="xpu",
                               dtype=torch.int32).unsqueeze(0)
    return key_cache, value_cache, block_table


def _run_paged_attention(q: torch.Tensor, key_cache: torch.Tensor,
                         value_cache: torch.Tensor,
                         block_table: torch.Tensor,
                         kv_len: int) -> torch.Tensor:
    query_len = q.shape[0]
    cu_seqlens_q = torch.tensor([0, query_len],
                                device="xpu",
                                dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device="xpu", dtype=torch.int32)
    output = torch.empty_like(q)
    return flash_attn_varlen_func(
        q,
        key_cache,
        value_cache,
        query_len,
        cu_seqlens_q,
        kv_len,
        seqused_k=seqused_k,
        softmax_scale=_HEAD_DIM**-0.5,
        causal=True,
        block_table=block_table,
        out=output,
    )


def _churn_allocator(repeat: int) -> torch.Tensor:
    # Keep the allocation live across the attention call so each repeat sees
    # a different surrounding allocation. The exact size is deliberately not
    # a multiple of an attention tensor's size.
    pressure = torch.empty(2 * 1024 * 1024 + repeat * 4099,
                           device="xpu",
                           dtype=torch.uint8)
    pressure.fill_(repeat)
    return pressure


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor,
                          context: str) -> None:
    actual_cpu = actual.cpu()
    if torch.equal(actual_cpu, expected):
        return
    different = actual_cpu != expected
    max_diff = (actual_cpu.float() - expected.float()).abs().max().item()
    pytest.fail(f"{context}: {different.sum().item()} values changed; "
                f"max_abs_diff={max_diff}")


def _assert_no_reference_fallback(caplog: pytest.LogCaptureFixture) -> None:
    fallback_records = [
        record for record in caplog.records
        if "not compiled" in record.getMessage()
        or "falling back" in record.getMessage()
    ]
    assert not fallback_records, (
        "determinism probe must exercise the native FA2 kernel, not the "
        "PyTorch reference fallback")


@pytest.mark.parametrize("seq_len", [104, 107, 162])
@torch.inference_mode()
def test_frozen_shape_dense_causal_prefill_is_bitwise_deterministic(
        seq_len: int, caplog: pytest.LogCaptureFixture) -> None:
    """Replay Kvarn's exact first-chunk FA2 shape after unrelated work."""
    caplog.set_level(logging.WARNING)
    torch.xpu.set_device("xpu:0")
    torch.manual_seed(0xFA2000 + seq_len)
    target = _make_qkv(seq_len)
    decoy = _make_qkv(seq_len)

    expected = _run_dense_prefill(*target).cpu()
    for repeat in range(_REPEATS):
        _run_dense_prefill(*decoy)
        pressure = _churn_allocator(repeat)
        actual = _run_dense_prefill(*target)
        _assert_bitwise_equal(actual, expected,
                              f"dense prefill L={seq_len}, repeat={repeat}")
        del actual, pressure
        torch.xpu.empty_cache()

    _assert_no_reference_fallback(caplog)


@torch.inference_mode()
def test_frozen_shape_paged_causal_prefill_is_bitwise_deterministic(
        caplog: pytest.LogCaptureFixture) -> None:
    """Replay the BF16 control's paged prefill path with exact model heads."""
    caplog.set_level(logging.WARNING)
    torch.xpu.set_device("xpu:0")
    seq_len = 162
    torch.manual_seed(0xFA2100 + seq_len)
    target_q = _make_qkv(seq_len)[0]
    target_cache = _make_paged_kv(seq_len)
    decoy_q = _make_qkv(seq_len)[0]
    decoy_cache = _make_paged_kv(seq_len)

    expected = _run_paged_attention(target_q, *target_cache, seq_len).cpu()
    for repeat in range(_REPEATS):
        _run_paged_attention(decoy_q, *decoy_cache, seq_len)
        pressure = _churn_allocator(repeat)
        actual = _run_paged_attention(target_q, *target_cache, seq_len)
        _assert_bitwise_equal(actual, expected,
                              f"paged prefill L={seq_len}, repeat={repeat}")
        del actual, pressure
        torch.xpu.empty_cache()

    _assert_no_reference_fallback(caplog)


@torch.inference_mode()
def test_frozen_shape_paged_decode_is_bitwise_deterministic(
        caplog: pytest.LogCaptureFixture) -> None:
    """Replay qlen=1 against fixed KV after same-shape pointer alternation."""
    caplog.set_level(logging.WARNING)
    torch.xpu.set_device("xpu:0")
    kv_len = 162
    torch.manual_seed(0xFA2200 + kv_len)
    target_q = torch.randn(1,
                           _NUM_QUERY_HEADS,
                           _HEAD_DIM,
                           device="xpu",
                           dtype=_DTYPE)
    target_cache = _make_paged_kv(kv_len)
    decoy_q = torch.randn_like(target_q)
    decoy_cache = _make_paged_kv(kv_len)

    expected = _run_paged_attention(target_q, *target_cache, kv_len).cpu()
    for repeat in range(_REPEATS):
        _run_paged_attention(decoy_q, *decoy_cache, kv_len)
        pressure = _churn_allocator(repeat)
        actual = _run_paged_attention(target_q, *target_cache, kv_len)
        _assert_bitwise_equal(actual, expected,
                              f"paged decode K={kv_len}, repeat={repeat}")
        del actual, pressure
        torch.xpu.empty_cache()

    _assert_no_reference_fallback(caplog)
