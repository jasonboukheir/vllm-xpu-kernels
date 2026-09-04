# SPDX-License-Identifier: Apache-2.0
"""Focused B70 runtime contracts for the isolated ID19 decoder.

This module intentionally does not add ID19 to the generic variant matrix.
Every ID19 call here uses the scratch ABI and passes the trailing
``last_producer_state_initialized`` flag explicitly, so a test cannot pass by
silently exercising ID18.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmark.check_kvarn_decode import make_random_cache  # noqa: E402
from benchmark.kvarn_utils import swizzle_record_dpas_k4v4  # noqa: E402


ID18_PREFETCH_RECORD_CURSOR = 18
ID19_B1_SHORT_LAST_PRODUCER = 19
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
PAGE_SIZE = 128
SHORT_CONTEXT_MAX = 8192


@pytest.fixture(scope="module", autouse=True)
def b70_runtime() -> None:
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if not library:
        pytest.skip("VLLM_XPU_KERNELS_LIBRARY is not set")
    if not torch.xpu.is_available():
        pytest.skip("an XPU is not available")
    if torch.xpu.get_device_name(0) != "Intel(R) Arc(TM) Pro B70 Graphics":
        pytest.skip("ID19 runtime contracts are gated to the Arc Pro B70")
    torch.ops.load_library(library)


def _tail_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.zeros(
        (1, PAGE_SIZE, KV_HEADS, HEAD_DIM), dtype=torch.float16, device="xpu"
    )
    return key, torch.zeros_like(key)


def _make_b1_dpas_case(
    seq_len: int,
) -> tuple[tuple[torch.Tensor, ...], int]:
    pages_per_row = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    canonical, layout = make_random_cache(pages_per_row)
    packed = canonical.clone()
    for block in range(pages_per_row):
        for kv_head in range(KV_HEADS):
            packed[block, kv_head] = swizzle_record_dpas_k4v4(
                canonical[block, kv_head], layout
            )

    query = torch.randn(
        (1, HEADS, HEAD_DIM),
        generator=torch.Generator().manual_seed(20260904 + seq_len),
        dtype=torch.float16,
    ).xpu()
    arguments = (
        query,
        packed.xpu(),
        torch.arange(
            pages_per_row, dtype=torch.int32, device="xpu"
        ).reshape(1, pages_per_row),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        torch.full((pages_per_row,), -1, dtype=torch.int32, device="xpu"),
        *_tail_tensors(),
    )
    return arguments, pages_per_row


def _new_scratch(
    splits: int, output_dtype: torch.dtype = torch.float16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            (1, HEADS * splits, HEAD_DIM),
            dtype=torch.float16,
            device="xpu",
        ),
        torch.empty(
            (1, HEADS, splits), dtype=torch.float32, device="xpu"
        ),
        # ID19 reserves the first four uint64 words of this zeroed tensor for
        # per-KV-head epoch/count completion state.
        torch.zeros((1, HEADS, splits), dtype=torch.float32, device="xpu"),
        torch.empty((1, HEADS, HEAD_DIM), dtype=output_dtype, device="xpu"),
    )


def _run_with_scratch(
    arguments: tuple[torch.Tensor, ...],
    seq_len: int,
    splits: int,
    kernel_variant: int,
    *,
    write_bf16_output: bool = False,
    last_producer_state_initialized: bool = False,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scratch is None:
        scratch = _new_scratch(
            splits,
            torch.bfloat16 if write_bf16_output else torch.float16,
        )
    temp_output, exp_sums, max_logits, output = scratch
    torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
        *arguments,
        temp_output,
        exp_sums,
        max_logits,
        output,
        seq_len,
        1.0 / 16.0,
        True,
        write_bf16_output,
        splits,
        kernel_variant,
        True,
        last_producer_state_initialized,
    )
    torch.xpu.synchronize()
    return output, max_logits


def _completion_words(max_logits: torch.Tensor) -> list[int]:
    words = max_logits.detach().cpu().contiguous().view(torch.int64).flatten()
    return [int(word) for word in words[:KV_HEADS].tolist()]


@pytest.mark.parametrize(
    ("seq_len", "splits"),
    [
        pytest.param(4095, 16, id="page-tail-before-4k"),
        pytest.param(4096, 16, id="4k-boundary"),
        pytest.param(4097, 16, id="page-tail-after-4k"),
        pytest.param(4608, 16, id="4k-plus-512"),
        pytest.param(8191, 32, id="short-bucket-tail"),
        pytest.param(8192, 32, id="short-bucket-boundary"),
    ],
)
def test_id19_b1_dpas_matches_exact_id18(
    seq_len: int, splits: int
) -> None:
    """ID19's persistent-scratch finalizer matches the exact ID18 result."""
    assert seq_len <= SHORT_CONTEXT_MAX
    arguments, _ = _make_b1_dpas_case(seq_len)
    id18, _ = _run_with_scratch(
        arguments, seq_len, splits, ID18_PREFETCH_RECORD_CURSOR
    )
    id19, id19_state = _run_with_scratch(
        arguments,
        seq_len,
        splits,
        ID19_B1_SHORT_LAST_PRODUCER,
        last_producer_state_initialized=True,
    )

    assert torch.isfinite(id19).all()
    torch.testing.assert_close(id19, id18, atol=0, rtol=0)
    assert _completion_words(id19_state) == [1 << 32] * KV_HEADS


def test_id19_reuses_persistent_scratch_and_advances_epochs() -> None:
    """Reuse scratch while advancing the completion epoch."""
    seq_len = 4608
    splits = 16
    arguments, _ = _make_b1_dpas_case(seq_len)
    scratch = _new_scratch(splits)

    for expected_epoch in (1, 2, 3):
        output, state = _run_with_scratch(
            arguments,
            seq_len,
            splits,
            ID19_B1_SHORT_LAST_PRODUCER,
            last_producer_state_initialized=True,
            scratch=scratch,
        )
        assert torch.isfinite(output).all()
        assert _completion_words(state) == [expected_epoch << 32] * KV_HEADS


@pytest.mark.parametrize(
    ("seq_len", "initialized"),
    [
        pytest.param(4096, False, id="trailing-init-false"),
        pytest.param(8193, True, id="above-8192-bucket"),
    ],
)
def test_id19_unsupported_contract_falls_back_to_exact_id18(
    seq_len: int, initialized: bool
) -> None:
    """The opt-in flag and bucket bound must both fail closed to ID18."""
    splits = 16 if seq_len <= 4608 else 32
    arguments, _ = _make_b1_dpas_case(seq_len)
    id18, _ = _run_with_scratch(
        arguments, seq_len, splits, ID18_PREFETCH_RECORD_CURSOR
    )
    fallback, fallback_state = _run_with_scratch(
        arguments,
        seq_len,
        splits,
        ID19_B1_SHORT_LAST_PRODUCER,
        last_producer_state_initialized=initialized,
    )

    torch.testing.assert_close(fallback, id18, atol=0, rtol=0)
    # Neither fail-closed case may execute the ID19 finalizer or advance its
    # completion epoch.
    assert _completion_words(fallback_state) == [0] * KV_HEADS


def test_id19_direct_bf16_output_matches_id18_fp16_rounding() -> None:
    """The native finalizer's direct BF16 store preserves the FP16 contract."""
    seq_len = 4096
    splits = 16
    arguments, _ = _make_b1_dpas_case(seq_len)
    id18_fp16, _ = _run_with_scratch(
        arguments, seq_len, splits, ID18_PREFETCH_RECORD_CURSOR
    )
    id19_bf16, state = _run_with_scratch(
        arguments,
        seq_len,
        splits,
        ID19_B1_SHORT_LAST_PRODUCER,
        write_bf16_output=True,
        last_producer_state_initialized=True,
    )

    torch.testing.assert_close(
        id19_bf16, id18_fp16.to(torch.bfloat16), atol=0, rtol=0
    )
    assert _completion_words(state) == [1 << 32] * KV_HEADS
