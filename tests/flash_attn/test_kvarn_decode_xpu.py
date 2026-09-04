# SPDX-License-Identifier: Apache-2.0
"""Parametrized runtime tests for the narrow native Xe2 KVarN decoder.

Set ``VLLM_XPU_KERNELS_LIBRARY`` to the freshly built ``_vllm_fa2_C`` shared
object.  Keeping the library explicit prevents this suite from accidentally
testing an older installed package during local source-override iteration.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmark.check_kvarn_decode import (  # noqa: E402
    _put_half,
    expected_value,
    make_cache,
    make_random_cache,
    reference_decode,
)
from benchmark.kvarn_utils import (  # noqa: E402
    KVarNLayout,
    _k_dpas_coord,
    _v_dpas_coord,
    dequant_record,
    swizzle_record_dpas_k4v4,
)


@pytest.fixture(scope="module", autouse=True)
def native_library() -> None:
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if not library:
        pytest.skip("VLLM_XPU_KERNELS_LIBRARY is not set")
    if not torch.xpu.is_available():
        pytest.skip("an XPU is not available")
    torch.ops.load_library(library)


def _tail_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.zeros((1, 128, 4, 256), dtype=torch.float16, device="xpu")
    return key, torch.zeros_like(key)


def _make_long_structured_cache(
    num_blocks: int,
) -> tuple[torch.Tensor, KVarNLayout, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a vectorized long-context cache with page-sensitive K/V rows."""
    layout = KVarNLayout(record_stride=35072)
    cache = torch.zeros(
        num_blocks, 4, layout.tile_bytes_aligned, dtype=torch.uint8
    )

    # Exercise the packed payload at high physical addresses as well as its
    # metadata. Page-varying uint4 K values make a wrong physical payload load
    # observable while keeping the independent expected value cheap.
    packed_k_values = torch.tensor([1, 2, 4], dtype=torch.uint8)[
        torch.arange(num_blocks) % 3
    ]
    packed_k_bytes = packed_k_values | (packed_k_values << 4)
    cache[:, :, : layout.k_packed_bytes] = packed_k_bytes[:, None, None]
    cache[
        :,
        :,
        layout.v_packed_offset : layout.v_packed_offset + layout.v_packed_bytes,
    ] = 0x11

    one_dim = torch.ones(layout.head_dim, dtype=torch.float16).view(torch.uint8)
    one_row = torch.ones(layout.group, dtype=torch.float16).view(torch.uint8)
    cache[
        :, :, layout.k_s_row_offset : layout.k_s_row_offset + one_row.numel()
    ] = one_row
    cache[
        :, :, layout.v_s_col_offset : layout.v_s_col_offset + one_dim.numel()
    ] = one_dim
    cache[
        :, :, layout.v_s_row_offset : layout.v_s_row_offset + one_row.numel()
    ] = one_row

    # Each batch query selects its own K dimension. The test marks one physical
    # final page per traversal with a high score, so dropping that page or
    # mishandling a partial tail changes the answer materially even at 262K.
    page_scores = torch.zeros(num_blocks, 4, dtype=torch.float16)
    key_column_scales = (
        (1.0 / packed_k_values.float())[:, None, None]
        .expand(-1, 4, layout.head_dim)
        .half()
        .contiguous()
    )
    cache[
        :,
        :,
        layout.k_s_col_offset : layout.k_s_col_offset + layout.head_dim * 2,
    ] = key_column_scales.view(torch.uint8)
    key_zero_points = torch.zeros(
        num_blocks, 4, layout.head_dim, dtype=torch.float16
    )
    key_zero_points[:, :, :4] = page_scores[:, None, :]
    cache[
        :, :, layout.k_zp_offset : layout.k_zp_offset + layout.head_dim * 2
    ] = key_zero_points.view(torch.uint8)

    page_values = ((torch.arange(num_blocks) * 73) % 257).float()
    page_values = (page_values - 128.0)[:, None] / 1024.0
    token_values = torch.arange(layout.group)[None, :] / 512.0
    value_rows = (page_values + token_values).half()
    value_zero_points = value_rows[:, None, :].expand(-1, 4, -1).contiguous()
    cache[:, :, layout.v_zp_offset : layout.v_zp_offset + layout.group * 2] = (
        value_zero_points.view(torch.uint8)
    )

    # Packed V contributes 0.25 through s_row. Per-head/per-dimension column
    # scales make GQA head or output-lane routing mistakes visible.
    value_row_scale = torch.full(
        (num_blocks, 4, layout.group), 0.25, dtype=torch.float16
    )
    cache[
        :, :, layout.v_s_row_offset : layout.v_s_row_offset + layout.group * 2
    ] = value_row_scale.view(torch.uint8)
    head_scale = 1.0 + torch.arange(4)[:, None] / 4.0
    dim_scale = 0.5 + ((torch.arange(layout.head_dim)[None, :] * 37) % 17) / 8.0
    value_column_scales = (head_scale * dim_scale).half()
    encoded_column_scales = (
        value_column_scales[None].expand(num_blocks, -1, -1).contiguous()
    )
    cache[
        :,
        :,
        layout.v_s_col_offset : layout.v_s_col_offset + layout.head_dim * 2,
    ] = encoded_column_scales.view(torch.uint8)
    return (
        cache,
        layout,
        page_scores.float(),
        value_rows.float(),
        value_column_scales.float(),
    )


def _long_structured_expected(
    pages: torch.Tensor,
    seq_len: int,
    page_scores: torch.Tensor,
    value_rows: torch.Tensor,
) -> float:
    full_pages, tail_tokens = divmod(seq_len, 128)
    physical = pages[: full_pages + int(tail_tokens > 0)]
    counts = torch.full((physical.numel(),), 128, dtype=torch.int64)
    if tail_tokens:
        counts[-1] = tail_tokens
    scores = page_scores[physical].double()
    weights = torch.exp(scores - scores.max())
    row_sums = torch.stack(
        [
            value_rows[block, :count].double().sum()
            for block, count in zip(physical.tolist(), counts.tolist())
        ]
    )
    packed_v_term = 0.25
    return float(
        (weights * (row_sums + packed_v_term * counts)).sum()
        / (weights * counts).sum()
    )


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("dpas_layout", [False, True])
def test_materialize_random_packed_cache_matches_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
    record_stride: int,
    dpas_layout: bool,
) -> None:
    canonical_cache, layout = make_random_cache(5, record_stride)
    packed_cache = canonical_cache.clone()
    if dpas_layout:
        monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
        for block in range(packed_cache.size(0)):
            for head in range(packed_cache.size(1)):
                packed_cache[block, head] = swizzle_record_dpas_k4v4(
                    canonical_cache[block, head], layout
                )
    else:
        monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)

    pages = [[4, 1], [3, 0]]
    lengths = [129, 194]
    total_tokens = sum(lengths)
    key_output = torch.empty(
        (total_tokens, 4, 256), dtype=torch.float16, device="xpu"
    )
    value_output = torch.empty_like(key_output)
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_materialize_packed_kv(
        packed_cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.tensor(
            [0, lengths[0], total_tokens], dtype=torch.int32, device="xpu"
        ),
        torch.full((5,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        key_output,
        value_output,
        max(lengths),
    )

    expected_key = []
    expected_value = []
    for request_pages, length in zip(pages, lengths, strict=True):
        records = [
            [
                dequant_record(canonical_cache[page, head], layout)
                for head in range(4)
            ]
            for page in request_pages
        ]
        expected_key.append(
            torch.stack(
                [
                    torch.cat([record[head][0] for record in records])[:length]
                    for head in range(4)
                ],
                dim=1,
            )
        )
        expected_value.append(
            torch.stack(
                [
                    torch.cat([record[head][1] for record in records])[:length]
                    for head in range(4)
                ],
                dim=1,
            )
        )

    torch.testing.assert_close(
        key_output.cpu(), torch.cat(expected_key).half(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        value_output.cpu(),
        torch.cat(expected_value).half(),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize(
    "seq_len", [1, 63, 64, 65, 127, 128, 129, 255, 256, 257]
)
def test_structured_permuted_pages(seq_len: int, record_stride: int) -> None:
    cache, _ = make_cache(3, record_stride)
    assert cache.stride(1) == record_stride
    pages = [2, 0, 1]
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        torch.full((3,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        output,
        seq_len,
        1.0 / 16.0,
    )
    expected = torch.full_like(output.cpu(), expected_value(pages, seq_len))
    torch.testing.assert_close(output.cpu(), expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("batch_size", [2, 3, 4, 12])
def test_ragged_batch_matches_independent_fp32_oracle(
    batch_size: int, record_stride: int
) -> None:
    cache, layout = make_random_cache(6, record_stride)
    assert cache.stride(1) == record_stride
    generator = torch.Generator().manual_seed(7)
    query = torch.randn((batch_size, 24, 256), generator=generator).to(
        torch.float16
    )
    pages = [[5, 1], [4, 0], [3, 2], [1, 5]] * 3
    lengths = [1, 63, 128, 129] * 3
    pages = pages[:batch_size]
    lengths = lengths[:batch_size]
    output = torch.empty_like(query, device="xpu")
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_decode(
        query.xpu(),
        cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.full((6,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        output,
        max(lengths),
        1.0 / 16.0,
    )
    references = [
        reference_decode(
            query[batch : batch + 1],
            cache,
            layout,
            [pages[batch]],
            lengths[batch],
            1.0 / 16.0,
        )
        for batch in range(batch_size)
    ]
    reference = torch.cat(references)
    torch.testing.assert_close(
        output.cpu().float(), reference, atol=2e-2, rtol=2e-2
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("dpas_layout", [False, True])
def test_nonuniform_kvarn_factors_across_page_boundary(
    monkeypatch: pytest.MonkeyPatch,
    record_stride: int,
    dpas_layout: bool,
) -> None:
    """Exercise separable K and V factors with no unit-scale shortcuts.

    The three rows end immediately before, on, and after a page boundary.
    Every physical record has different, strongly nonuniform token and
    dimension factors. This detects applying a factor to the wrong fragment
    axis or accidentally carrying page-specific metadata across records.
    """
    cache, layout = make_random_cache(6, record_stride)
    assert cache.stride(1) == record_stride
    token = torch.arange(layout.group, dtype=torch.float32)
    dim = torch.arange(layout.head_dim, dtype=torch.float32)
    for block in range(6):
        for kv_head in range(4):
            phase = (block * 19 + kv_head * 7) % layout.group
            row_scale = 0.125 + ((token + phase) % layout.group) / 80.0
            dim_phase = block * 23 + kv_head * 11
            dim_scale = 0.003 + ((dim + dim_phase) % 97) / 1600.0
            dim_zp = (((dim * 37 + dim_phase) % 101) - 50) / 125.0
            v_dim_scale = 0.005 + ((dim * 13 + dim_phase) % 89) / 1800.0
            v_row_scale = 0.02 + ((token * 17 + phase) % 83) / 700.0
            v_row_zp = (((token * 29 + phase) % 79) - 39) / 160.0
            _put_half(cache[block, kv_head], layout.k_s_col_offset, dim_scale)
            _put_half(cache[block, kv_head], layout.k_zp_offset, dim_zp)
            _put_half(cache[block, kv_head], layout.k_s_row_offset, row_scale)
            _put_half(cache[block, kv_head], layout.v_s_col_offset, v_dim_scale)
            _put_half(cache[block, kv_head], layout.v_s_row_offset, v_row_scale)
            _put_half(cache[block, kv_head], layout.v_zp_offset, v_row_zp)

    packed_cache = cache
    if dpas_layout:
        monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
        packed_cache = cache.clone()
        for block in range(packed_cache.size(0)):
            for kv_head in range(packed_cache.size(1)):
                packed_cache[block, kv_head] = swizzle_record_dpas_k4v4(
                    cache[block, kv_head], layout
                )
    else:
        monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)

    generator = torch.Generator().manual_seed(314159)
    query = torch.randn((3, 24, 256), generator=generator).to(torch.float16)
    pages = [[5, 0], [4, 1], [3, 2]]
    lengths = [127, 128, 129]
    output = torch.empty_like(query, device="xpu")
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_decode(
        query.xpu(),
        packed_cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.full((6,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        output,
        max(lengths),
        1.0 / 16.0,
    )
    reference = torch.cat(
        [
            reference_decode(
                query[batch : batch + 1],
                cache,
                layout,
                [pages[batch]],
                lengths[batch],
                1.0 / 16.0,
            )
            for batch in range(3)
        ]
    )
    torch.testing.assert_close(
        output.cpu().float(), reference, atol=3e-2, rtol=3e-2
    )
    assert torch.isfinite(output).all()


def test_kvarn_dpas_fragment_coordinate_tables_are_bijections() -> None:
    """Freeze the actual CUTE B-fragment ownership used by native decode."""
    anchor = torch.empty((), device="xpu")
    k_coords, v_coords = torch.ops._vllm_fa2_C.kvarn_fragment_coords(anchor)

    assert k_coords.device.type == "cpu"
    assert v_coords.device.type == "cpu"
    assert k_coords.dtype == torch.int32
    assert v_coords.dtype == torch.int32
    assert k_coords.shape == (16, 64, 2)
    assert v_coords.shape == (16, 32, 2)

    k_pairs = {tuple(coord) for coord in k_coords.reshape(-1, 2).tolist()}
    v_pairs = {tuple(coord) for coord in v_coords.reshape(-1, 2).tolist()}
    assert k_pairs == {(token, dim) for token in range(16) for dim in range(64)}
    assert v_pairs == {(dim, token) for dim in range(32) for token in range(16)}

    # Each logical coordinate is owned exactly once, not merely present.
    assert k_coords.reshape(-1, 2).unique(dim=0).shape[0] == 16 * 64
    assert v_coords.reshape(-1, 2).unique(dim=0).shape[0] == 32 * 16

    frozen_k = torch.tensor(
        [
            [_k_dpas_coord(lane, slot) for slot in range(64)]
            for lane in range(16)
        ],
        dtype=torch.int32,
    )
    frozen_v = torch.tensor(
        [
            [_v_dpas_coord(lane, slot) for slot in range(32)]
            for lane in range(16)
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(k_coords, frozen_k, rtol=0, atol=0)
    torch.testing.assert_close(v_coords, frozen_v, rtol=0, atol=0)


@pytest.mark.parametrize("dpas_layout", [False, True])
def test_k_column_scale_reaches_every_token_subgroup(
    monkeypatch: pytest.MonkeyPatch, dpas_layout: bool
) -> None:
    """Catch mixing MMA-A scale ownership with MMA-B cache fragments."""
    if dpas_layout:
        monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    else:
        monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "1")

    layout = KVarNLayout(record_stride=35072)
    cache = torch.zeros((2, 4, layout.tile_bytes_aligned), dtype=torch.uint8)
    cache[:, :, : layout.k_packed_bytes] = 0x11
    ones_d = torch.ones(layout.head_dim)
    ones_g = torch.ones(layout.group)
    for block in range(2):
        for head in range(4):
            record = cache[block, head]
            k_column_scale = ones_d.clone()
            if block == 1:
                k_column_scale[0] = 5.0
            _put_half(record, layout.k_s_col_offset, k_column_scale)
            _put_half(record, layout.k_s_row_offset, ones_g)
            _put_half(record, layout.v_s_col_offset, ones_d)
            _put_half(record, layout.v_s_row_offset, ones_g)
            _put_half(
                record,
                layout.v_zp_offset,
                torch.full((layout.group,), float(block)),
            )

    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    query[:, :, 0] = 16.0
    output = torch.full_like(query, float("nan"))
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache.xpu(),
        torch.tensor([[0, 1]], dtype=torch.int32, device="xpu"),
        torch.tensor([256], dtype=torch.int32, device="xpu"),
        torch.full((2,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        output,
        256,
        1.0 / 16.0,
    )

    # Page 0 has logit 1 and value 0; page 1 has logit 5 and value 1.
    # Each page contributes 128 tokens, so every output is sigmoid(4).
    expected = torch.full_like(output.cpu(), torch.sigmoid(torch.tensor(4.0)))
    torch.testing.assert_close(output.cpu(), expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(output).all()


def test_dpas_payload_decode_matches_canonical_ragged_and_hybrid() -> None:
    cache, layout = make_random_cache(6)
    swizzled = cache.clone()
    for block in range(6):
        for head in range(4):
            swizzled[block, head] = swizzle_record_dpas_k4v4(
                cache[block, head], layout
            )

    generator = torch.Generator().manual_seed(271828)
    query = torch.randn((3, 24, 256), generator=generator).half().xpu()
    pages = torch.tensor(
        [[5, 0, 1], [4, 1, 2], [3, 2, 0]], dtype=torch.int32, device="xpu"
    )
    lengths = torch.tensor([127, 129, 257], dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((6,), -1, dtype=torch.int32, device="xpu")
    block_to_slot[5] = 0
    tail_key = torch.randn(
        (1, 128, 4, 256), generator=generator, dtype=torch.float16
    ).xpu()
    tail_value = torch.randn(
        (1, 128, 4, 256), generator=generator, dtype=torch.float16
    ).xpu()
    canonical_output = torch.empty_like(query)
    swizzled_output = torch.empty_like(query)

    old_flag = os.environ.pop("KVARN_NATIVE_XPU_DPAS_LAYOUT", None)
    try:
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            cache.xpu(),
            pages,
            lengths,
            block_to_slot,
            tail_key,
            tail_value,
            canonical_output,
            257,
            1.0 / 16.0,
        )
        os.environ["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "1"
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            swizzled.xpu(),
            pages,
            lengths,
            block_to_slot,
            tail_key,
            tail_value,
            swizzled_output,
            257,
            1.0 / 16.0,
        )
    finally:
        if old_flag is None:
            os.environ.pop("KVARN_NATIVE_XPU_DPAS_LAYOUT", None)
        else:
            os.environ["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = old_flag
    torch.testing.assert_close(
        swizzled_output, canonical_output, rtol=0, atol=0
    )


def test_full_precision_tail_and_packed_history_share_softmax() -> None:
    cache, _ = make_cache(3)
    pages = [2, 0]
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    tail_key, tail_value = _tail_tensors()
    tail_value.fill_(0.75)
    block_to_slot = torch.full((3,), -1, dtype=torch.int32, device="xpu")
    block_to_slot[2] = 0
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([129], dtype=torch.int32, device="xpu"),
        block_to_slot,
        tail_key,
        tail_value,
        output,
        129,
        1.0 / 16.0,
    )
    expected = torch.full_like(output.cpu(), (128 * 0.75) / 129)
    torch.testing.assert_close(output.cpu(), expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("seq_len", [128, 129, 257])
def test_multiple_full_precision_tail_pages_share_one_softmax(
    seq_len: int,
) -> None:
    """Decode must not assume there is only one pool-resident page.

    Hybrid KVarN keeps both the sink and current partial page in fp16.  The
    first decode append therefore changes a one-tail-page request into a
    two-tail-page request, which is the exact transition exercised here.
    """
    torch.xpu.synchronize()
    cache, _ = make_cache(6)
    pages = [2, 4, 1]
    block_to_slot = torch.full((6,), -1, dtype=torch.int32, device="xpu")
    for slot, physical in enumerate(pages):
        block_to_slot[physical] = slot
    tail_key = torch.zeros((3, 128, 4, 256), dtype=torch.float16, device="xpu")
    tail_value = torch.empty_like(tail_key)
    values = (0.25, 0.75, -0.5)
    for slot, value in enumerate(values):
        tail_value[slot].fill_(value)
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        block_to_slot,
        tail_key,
        tail_value,
        output,
        seq_len,
        1.0 / 16.0,
    )
    counts = (
        min(seq_len, 128),
        min(max(seq_len - 128, 0), 128),
        max(seq_len - 256, 0),
    )
    expected_value_ = (
        sum(n * value for n, value in zip(counts, values)) / seq_len
    )
    torch.testing.assert_close(
        output.cpu(),
        torch.full_like(output.cpu(), expected_value_),
        atol=2e-3,
        rtol=2e-3,
    )


def test_two_random_tail_pages_match_fp32_attention_at_first_append() -> None:
    seq_len = 129
    cache, _ = make_cache(4)
    pages = [3, 1] + [0] * 62
    lookup = torch.full((4,), -1, dtype=torch.int32, device="xpu")
    lookup[3], lookup[1] = 0, 1
    generator = torch.Generator().manual_seed(12946613)
    tail_key_cpu = torch.randn((2, 128, 4, 256), generator=generator).half()
    tail_value_cpu = torch.randn((2, 128, 4, 256), generator=generator).half()
    query_cpu = torch.randn((1, 24, 256), generator=generator).half()
    output = torch.empty_like(query_cpu, device="xpu")
    torch.ops._vllm_fa2_C.kvarn_decode(
        query_cpu.xpu(),
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        lookup,
        tail_key_cpu.xpu(),
        tail_value_cpu.xpu(),
        output,
        8192,
        1.0 / 16.0,
    )
    keys = torch.cat((tail_key_cpu[0], tail_key_cpu[1, :1]))
    values = torch.cat((tail_value_cpu[0], tail_value_cpu[1, :1]))
    kv_heads = torch.arange(24) // 6
    selected_k = keys[:, kv_heads].float().permute(1, 0, 2)
    selected_v = values[:, kv_heads].float().permute(1, 0, 2)
    logits = torch.einsum("hd,htd->ht", query_cpu[0].float(), selected_k) / 16.0
    reference = torch.einsum(
        "ht,htd->hd", torch.softmax(logits, -1), selected_v
    )
    torch.testing.assert_close(
        output.cpu()[0].float(), reference, atol=3e-3, rtol=3e-3
    )


def test_shared_prefix_is_deterministic() -> None:
    cache, _ = make_random_cache(5)
    generator = torch.Generator().manual_seed(19)
    query_row = torch.randn((1, 24, 256), generator=generator).to(torch.float16)
    query = query_row.expand(4, -1, -1).contiguous().xpu()
    # All rows alias the same physical prefix. Their second pages differ, but
    # seq_len=128 makes those entries padding and therefore deliberately
    # invalid inputs to the device loader.
    pages = [[3, 0], [3, 1], [3, 2], [3, 4]]
    seq_lens = torch.full((4,), 128, dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((5,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    output = torch.empty_like(query)
    repeat = torch.empty_like(query)
    arguments = (
        query,
        cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )
    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, output, 128, 1.0 / 16.0)
    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, repeat, 128, 1.0 / 16.0)
    torch.testing.assert_close(output, repeat, atol=0, rtol=0)
    torch.testing.assert_close(
        output, output[0:1].expand_as(output), atol=0, rtol=0
    )


def test_decode_with_scratch_matches_legacy_and_reuses_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.xpu.synchronize()
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "16")
    cache, _ = make_random_cache(6)
    generator = torch.Generator().manual_seed(20260808)
    query = torch.randn((4, 24, 256), generator=generator).half().xpu()
    pages = torch.tensor(
        [[5, 0], [4, 1], [3, 2], [1, 5]], dtype=torch.int32, device="xpu"
    )
    seq_lens = torch.tensor([1, 127, 128, 129], dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((6,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )
    legacy = torch.empty_like(query)
    actual = torch.empty_like(query)
    temp_output = torch.empty(
        (4, 16 * 24, 256), dtype=torch.float16, device="xpu"
    )
    exp_sums = torch.empty((4, 24, 16), dtype=torch.float32, device="xpu")
    max_logits = torch.empty_like(exp_sums)
    pointers = tuple(
        tensor.data_ptr() for tensor in (temp_output, exp_sums, max_logits)
    )

    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, legacy, 129, 1.0 / 16.0)
    for _ in range(3):
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            *arguments,
            temp_output,
            exp_sums,
            max_logits,
            actual,
            129,
            1.0 / 16.0,
        )
        assert pointers == tuple(
            tensor.data_ptr() for tensor in (temp_output, exp_sums, max_logits)
        )
        torch.testing.assert_close(actual, legacy, atol=0, rtol=0)


@pytest.mark.parametrize("oversized", ["temp_output", "statistics"])
def test_multisplit_scratch_requires_exact_split_extent(
    monkeypatch: pytest.MonkeyPatch, oversized: str
) -> None:
    """Raw packed reducer strides cannot address oversized split extents."""
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "2")
    cache, _ = make_random_cache(16)
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    arguments = (
        query,
        cache.xpu(),
        torch.arange(16, dtype=torch.int32, device="xpu").reshape(1, 16),
        torch.tensor([2048], dtype=torch.int32, device="xpu"),
        torch.full((16,), -1, dtype=torch.int32, device="xpu"),
        *_tail_tensors(),
    )
    temp_splits = 3 if oversized == "temp_output" else 2
    stats_splits = 3 if oversized == "statistics" else 2
    temp_output = torch.empty(
        (1, 24 * temp_splits, 256), dtype=torch.float16, device="xpu"
    )
    exp_sums = torch.empty(
        (1, 24, stats_splits), dtype=torch.float32, device="xpu"
    )
    max_logits = torch.empty_like(exp_sums)
    expected_error = (
        "temp_output must" if oversized == "temp_output" else "exp_sums and"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            *arguments,
            temp_output,
            exp_sums,
            max_logits,
            torch.empty_like(query),
            2048,
            1.0 / 16.0,
        )


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("splits", [16, 32])
def test_split_local_scratch_survives_allocator_pressure(
    monkeypatch: pytest.MonkeyPatch, record_stride: int, splits: int
) -> None:
    """The asynchronous reducer must retain legacy wrapper scratch storage."""
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    cache, _ = make_random_cache(188, record_stride)
    generator = torch.Generator().manual_seed(16000 + record_stride)
    query = torch.randn((4, 24, 256), generator=generator).half().xpu()
    pages = torch.arange(188, dtype=torch.int32, device="xpu").reshape(4, 47)
    seq_lens = torch.tensor(
        [6000, 5889, 5761, 5633], dtype=torch.int32, device="xpu"
    )
    block_to_slot = torch.full((188,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    expected = torch.empty_like(query)
    actual = torch.empty_like(query)
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )

    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, expected, 6000, 1.0 / 16.0)
    torch.xpu.synchronize()
    for _ in range(16):
        torch.ops._vllm_fa2_C.kvarn_decode(*arguments, actual, 6000, 1.0 / 16.0)
        # Match all three local scratch allocation classes before consuming
        # the result, maximizing the chance of premature allocator reuse.
        torch.empty(
            (4, 24 * splits, 256), dtype=torch.float16, device="xpu"
        ).fill_(float("nan"))
        torch.empty((4, 24, splits), dtype=torch.float32, device="xpu").fill_(
            float("nan")
        )
        torch.empty((4, 24, splits), dtype=torch.float32, device="xpu").fill_(
            float("nan")
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("seq_len", [128, 129, 6000])
@pytest.mark.parametrize("splits", [16, 32])
def test_split_matches_split1(
    monkeypatch: pytest.MonkeyPatch,
    record_stride: int,
    seq_len: int,
    splits: int,
) -> None:
    pages_per_row = (seq_len + 127) // 128
    blocks = 4 * pages_per_row
    cache, _ = make_random_cache(blocks, record_stride)
    generator = torch.Generator().manual_seed(seq_len + record_stride)
    query = torch.randn((4, 24, 256), generator=generator).half().xpu()
    pages = torch.arange(blocks, dtype=torch.int32, device="xpu").reshape(
        4, pages_per_row
    )
    seq_lens = torch.full((4,), seq_len, dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((blocks,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )
    split1 = torch.empty_like(query)
    split_output = torch.empty_like(query)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "1")
    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, split1, seq_len, 1.0 / 16.0)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    torch.ops._vllm_fa2_C.kvarn_decode(
        *arguments, split_output, seq_len, 1.0 / 16.0
    )
    torch.testing.assert_close(split_output, split1, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("splits", [16, 32])
@pytest.mark.parametrize("fused_hadamard", [False, True])
def test_ragged_b4_ignores_poisoned_globally_inactive_splits(
    monkeypatch: pytest.MonkeyPatch, splits: int, fused_hadamard: bool
) -> None:
    """Reducers must ignore split lanes the producer does not schedule."""
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    max_seq_len = 3073
    seq_lengths = (3070, 3071, 3072, max_seq_len)
    pages_per_row = (max_seq_len + 127) // 128
    cache, _ = make_cache(pages_per_row)
    pages_cpu = list(range(pages_per_row))
    pages = torch.tensor([pages_cpu] * 4, dtype=torch.int32, device="xpu")
    query = torch.zeros((4, 24, 256), dtype=torch.float16, device="xpu")
    seq_lens = torch.tensor(seq_lengths, dtype=torch.int32, device="xpu")
    block_to_slot = torch.full(
        (pages_per_row,), -1, dtype=torch.int32, device="xpu"
    )
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        *_tail_tensors(),
    )
    expected = torch.stack(
        [
            torch.full(
                (24, 256),
                expected_value(pages_cpu, seq_len),
                dtype=torch.float16,
                device="xpu",
            )
            for seq_len in seq_lengths
        ]
    )
    if fused_hadamard:
        transformed = torch.empty_like(expected)
        torch.ops._vllm_fa2_C.kvarn_hadamard(
            expected.view(-1, 256), transformed.view(-1, 256)
        )
        expected = transformed

    temp_output = torch.empty(
        (4, 24 * splits, 256), dtype=torch.float16, device="xpu"
    )
    exp_sums = torch.empty((4, 24, splits), dtype=torch.float32, device="xpu")
    max_logits = torch.empty_like(exp_sums)
    output = torch.empty_like(query)
    max_kv_tiles = (max_seq_len + 63) // 64
    tiles_per_split = (max_kv_tiles + splits - 1) // splits
    globally_active_splits = (
        max_kv_tiles + tiles_per_split - 1
    ) // tiles_per_split

    for _ in range(3):
        temp_output.fill_(float("nan"))
        exp_sums.fill_(float("nan"))
        max_logits.fill_(float("nan"))
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            *arguments,
            temp_output,
            exp_sums,
            max_logits,
            output,
            max_seq_len,
            1.0 / 16.0,
            fused_hadamard,
        )
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)

        softmax_lse_cpu = exp_sums.cpu()
        legacy_max_logits_cpu = max_logits.cpu()
        invalid_lse = torch.finfo(torch.float32).min
        for row, seq_len in enumerate(seq_lengths):
            kv_tiles = (seq_len + 63) // 64
            active_splits = (kv_tiles + tiles_per_split - 1) // tiles_per_split
            # The producer launches the batch-global split surface. A split
            # empty only for this row must publish its validity sentinel.
            torch.testing.assert_close(
                softmax_lse_cpu[row, :, active_splits:globally_active_splits],
                torch.full(
                    (24, globally_active_splits - active_splits),
                    invalid_lse,
                ),
                atol=0,
                rtol=0,
            )
        # The second public scratch tensor is retained for ABI compatibility,
        # but upstream's natural-LSE producer deliberately leaves it alone.
        assert torch.isnan(legacy_max_logits_cpu).all()
        # The producer publishes the same validity sentinel for splits outside
        # the batch-global active surface, but deliberately leaves their much
        # larger partial-output rows untouched.  Keeping those rows poisoned
        # proves the specialized reducers never consume an inactive partial.
        torch.testing.assert_close(
            softmax_lse_cpu[:, :, globally_active_splits:],
            torch.full((4, 24, splits - globally_active_splits), invalid_lse),
            atol=0,
            rtol=0,
        )
        inactive_partials = temp_output.view(4, splits, 24, 256)[
            :, globally_active_splits:
        ]
        assert torch.isnan(inactive_partials).all()


@pytest.mark.parametrize("splits", [2, 4, 8, 16, 17, 24, 32])
def test_fused_output_hadamard_matches_separate_transform(
    monkeypatch: pytest.MonkeyPatch, splits: int
) -> None:
    """The reducer fusion preserves the established fp16/H256 boundary."""
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    max_seq_len = 4096
    pages_per_row = max_seq_len // 128
    blocks = 4 * pages_per_row
    cache, _ = make_random_cache(blocks)
    generator = torch.Generator().manual_seed(8675309 + splits)
    query = torch.randn((4, 24, 256), generator=generator).half().xpu()
    pages = torch.arange(blocks, dtype=torch.int32, device="xpu").reshape(
        4, pages_per_row
    )
    seq_lens = torch.tensor(
        [max_seq_len, 2048, 1024, 128], dtype=torch.int32, device="xpu"
    )
    block_to_slot = torch.full((blocks,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )

    rotated = torch.empty_like(query)
    expected = torch.empty_like(query)
    fused = torch.empty_like(query)
    torch.ops._vllm_fa2_C.kvarn_decode(
        *arguments, rotated, max_seq_len, 1.0 / 16.0
    )
    torch.ops._vllm_fa2_C.kvarn_hadamard(
        rotated.view(-1, 256), expected.view(-1, 256)
    )
    torch.ops._vllm_fa2_C.kvarn_decode(
        *arguments, fused, max_seq_len, 1.0 / 16.0, True
    )

    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused, expected, atol=2e-2, rtol=2e-2)


def test_fused_output_hadamard_rejects_single_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "1")
    cache, _ = make_random_cache(4)
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="requires a multi-split"):
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            cache.xpu(),
            torch.arange(4, dtype=torch.int32, device="xpu").reshape(1, 4),
            torch.tensor([128], dtype=torch.int32, device="xpu"),
            torch.full((4,), -1, dtype=torch.int32, device="xpu"),
            *_tail_tensors(),
            torch.empty_like(query),
            128,
            1.0 / 16.0,
            True,
        )


@pytest.mark.parametrize("with_scratch", [False, True])
@pytest.mark.parametrize("splits", [16, 24])
def test_fused_bf16_output_matches_fp16_copy_contract(
    monkeypatch: pytest.MonkeyPatch, with_scratch: bool, splits: int
) -> None:
    """Direct bf16 output must preserve both historical fp16 roundings."""
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    seq_len = 6000
    pages_per_row = (seq_len + 127) // 128
    cache, _ = make_random_cache(pages_per_row)
    generator = torch.Generator().manual_seed(20260904 + splits)
    query = torch.randn((1, 24, 256), generator=generator).half().xpu()
    arguments = (
        query,
        cache.xpu(),
        torch.arange(pages_per_row, dtype=torch.int32, device="xpu").reshape(
            1, pages_per_row
        ),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        torch.full((pages_per_row,), -1, dtype=torch.int32, device="xpu"),
        *_tail_tensors(),
    )
    historical = torch.empty_like(query)
    direct = torch.empty_like(query, dtype=torch.bfloat16)

    if with_scratch:
        temp_output = torch.empty(
            (1, 24 * splits, 256), dtype=torch.float16, device="xpu"
        )
        exp_sums = torch.empty(
            (1, 24, splits), dtype=torch.float32, device="xpu"
        )
        max_logits = torch.empty_like(exp_sums)
        scratch = (temp_output, exp_sums, max_logits)
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            *arguments,
            *scratch,
            historical,
            seq_len,
            1.0 / 16.0,
            True,
        )
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            *arguments,
            *scratch,
            direct,
            seq_len,
            1.0 / 16.0,
            True,
            True,
        )
    else:
        torch.ops._vllm_fa2_C.kvarn_decode(
            *arguments, historical, seq_len, 1.0 / 16.0, True
        )
        torch.ops._vllm_fa2_C.kvarn_decode(
            *arguments, direct, seq_len, 1.0 / 16.0, True, True
        )

    torch.testing.assert_close(
        direct, historical.to(torch.bfloat16), atol=0, rtol=0
    )


def test_bf16_output_requires_fused_unrotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "16")
    cache, _ = make_random_cache(4)
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="requires unrotate_output"):
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            cache.xpu(),
            torch.arange(4, dtype=torch.int32, device="xpu").reshape(1, 4),
            torch.tensor([128], dtype=torch.int32, device="xpu"),
            torch.full((4,), -1, dtype=torch.int32, device="xpu"),
            *_tail_tensors(),
            torch.empty_like(query, dtype=torch.bfloat16),
            128,
            1.0 / 16.0,
            False,
            True,
        )


_LONG_CONTEXT_LAYOUT_SPLITS = [
    pytest.param(False, splits, id=f"natural-split{splits}")
    for splits in (1, 2, 4, 8, 16, 17, 24, 32)
] + [
    pytest.param(True, splits, id=f"dpas-split{splits}")
    for splits in (1, 16, 24, 32)
]


@pytest.mark.parametrize(
    ("dpas_layout", "splits"), _LONG_CONTEXT_LAYOUT_SPLITS
)
def test_long_context_ragged_b4_matches_structured_oracle(
    monkeypatch: pytest.MonkeyPatch, dpas_layout: bool, splits: int
) -> None:
    """Exercise packed and hybrid traversal through the 262K boundary."""
    if dpas_layout:
        monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    else:
        monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    num_blocks = 2048
    cache, layout, page_scores, value_rows, column_scales = (
        _make_long_structured_cache(num_blocks)
    )
    base = torch.arange(num_blocks, dtype=torch.int64)
    page_rows = torch.stack(
        (
            base,
            (base * 5 + 17) % num_blocks,
            base.flip(0),
            torch.cat((torch.tensor([2047, 1023]), base[2:])),
        )
    )
    seq_lengths = (262144, 131071, 65536, 192)
    target_pages = [
        int(page_rows[row, (seq_len - 1) // 128])
        for row, seq_len in enumerate(seq_lengths)
    ]
    target_values = (0.75, 0.25, -0.25, -0.75)
    for row, (target_page, target_value) in enumerate(
        zip(target_pages, target_values)
    ):
        page_scores[target_page, row] = 8.0
        value_rows[target_page] = target_value + (torch.arange(128) / 512.0)

    # Apply the late score sentinels through q_k * k_s_col rather than K zero
    # points. The packed K payload is therefore required for the oracle.
    packed_k_values = torch.tensor([1.0, 2.0, 4.0])[
        torch.arange(num_blocks) % 3
    ]
    key_column_scales = (
        (1.0 / packed_k_values)[:, None, None]
        .expand(-1, 4, layout.head_dim)
        .clone()
    )
    key_column_scales[:, :, :4] = (
        (1.0 + page_scores) / packed_k_values[:, None]
    )[:, None, :]
    cache[
        :,
        :,
        layout.k_s_col_offset : layout.k_s_col_offset + layout.head_dim * 2,
    ] = key_column_scales.half().view(torch.uint8)
    value_zero_points = (
        value_rows.half()[:, None, :].expand(-1, 4, -1).contiguous()
    )
    cache[:, :, layout.v_zp_offset : layout.v_zp_offset + layout.group * 2] = (
        value_zero_points.view(torch.uint8)
    )

    query = torch.zeros((4, 24, 256), dtype=torch.float16)
    for row in range(4):
        query[row, :, row] = 16.0
    output = torch.full_like(query, float("nan"), device="xpu")
    repeat = torch.full_like(output, float("nan"))

    # Put the highest physical block in the fp16 tail while leaving every
    # other block packed.  It is active in all four differently shaped rows,
    # including the final page of the 262K row, so both packed/tail softmax
    # composition and high-address traversal are covered in one B4 case.
    # The structured packed payload uses a constant nibble per record, making
    # its natural and DPAS byte layouts identical.  The independent random
    # payload tests above retain responsibility for validating the swizzle.
    hybrid_physical = num_blocks - 1
    tail_key_cpu = torch.empty((1, 128, 4, 256), dtype=torch.float16)
    tail_value_cpu = torch.empty_like(tail_key_cpu)
    for kv_head in range(4):
        key_page, value_page = dequant_record(
            cache[hybrid_physical, kv_head], layout
        )
        tail_key_cpu[0, :, kv_head] = key_page
        tail_value_cpu[0, :, kv_head] = value_page
    block_to_slot = torch.full(
        (num_blocks,), -1, dtype=torch.int32, device="xpu"
    )
    block_to_slot[hybrid_physical] = 0
    arguments = (
        query.xpu(),
        cache.xpu(),
        page_rows.to(dtype=torch.int32, device="xpu"),
        torch.tensor(seq_lengths, dtype=torch.int32, device="xpu"),
        block_to_slot,
        tail_key_cpu.xpu(),
        tail_value_cpu.xpu(),
    )

    temp_output = torch.full(
        (4, 24 * splits, 256),
        float("nan"),
        dtype=torch.float16,
        device="xpu",
    )
    exp_sums = torch.full(
        (4, 24, splits), float("nan"), dtype=torch.float32, device="xpu"
    )
    max_logits = torch.full_like(exp_sums, float("nan"))
    torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
        *arguments,
        temp_output,
        exp_sums,
        max_logits,
        output,
        max(seq_lengths),
        1.0 / 16.0,
    )
    torch.ops._vllm_fa2_C.kvarn_decode(
        *arguments, repeat, max(seq_lengths), 1.0 / 16.0
    )
    if splits > 1:
        partials_cpu = temp_output.cpu().view(4, splits, 24, 256)
        softmax_lse_cpu = exp_sums.cpu()
        legacy_max_logits_cpu = max_logits.cpu()
        invalid_lse = torch.finfo(torch.float32).min
        max_kv_tiles = (max(seq_lengths) + 63) // 64
        tiles_per_split = (max_kv_tiles + splits - 1) // splits
        globally_active_splits = min(
            splits,
            (max_kv_tiles + tiles_per_split - 1) // tiles_per_split,
        )
        for row, seq_len in enumerate(seq_lengths):
            kv_tiles = (seq_len + 63) // 64
            active_splits = min(
                splits, (kv_tiles + tiles_per_split - 1) // tiles_per_split
            )
            assert torch.isfinite(partials_cpu[row, :active_splits]).all()
            assert torch.isfinite(softmax_lse_cpu[row, :, :active_splits]).all()
            torch.testing.assert_close(
                softmax_lse_cpu[row, :, active_splits:globally_active_splits],
                torch.full(
                    (24, globally_active_splits - active_splits),
                    invalid_lse,
                ),
                atol=0,
                rtol=0,
            )
        assert torch.isnan(softmax_lse_cpu[:, :, globally_active_splits:]).all()
        assert torch.isnan(legacy_max_logits_cpu).all()

        kv_tiles = max(seq_lengths) // 64
        tiles_per_split = (kv_tiles + splits - 1) // splits
        target_split = (kv_tiles - 2) // tiles_per_split
        split_start = target_split * tiles_per_split
        split_end = min(split_start + tiles_per_split, kv_tiles)
        expected_exp_sum = 128 + (
            (split_end - split_start) * 64 - 128
        ) * math.exp(-8.0)
        expected_lse = 9.0 + math.log(expected_exp_sum)
        torch.testing.assert_close(
            exp_sums[0, :, target_split].cpu(),
            torch.full((24,), expected_lse),
            atol=2e-4,
            rtol=2e-4,
        )
    expected_rows = [
        _long_structured_expected(
            page_rows[row], seq_len, page_scores[:, row], value_rows
        )
        for row, seq_len in enumerate(seq_lengths)
    ]
    # Mutation checks keep this oracle from becoming insensitive again.
    assert (
        min(
            abs(left - right)
            for index, left in enumerate(expected_rows)
            for right in expected_rows[index + 1 :]
        )
        > 0.2
    )
    omitted_last = [
        _long_structured_expected(
            page_rows[row],
            ((seq_len - 1) // 128) * 128,
            page_scores[:, row],
            value_rows,
        )
        for row, seq_len in enumerate(seq_lengths)
    ]
    assert all(
        abs(actual - truncated) > 0.05
        for actual, truncated in zip(expected_rows, omitted_last)
    )
    rounded_tail = _long_structured_expected(
        page_rows[3], 256, page_scores[:, 3], value_rows
    )
    assert abs(expected_rows[3] - rounded_tail) > 0.05

    packed_k_mutant = cache[target_pages[0], 0].clone()
    packed_k_mutant[: layout.k_packed_bytes].zero_()
    mutant_key, _ = dequant_record(packed_k_mutant, layout)
    assert mutant_key[0, 0] == 0

    expected = torch.empty_like(query)
    for row, base_value in enumerate(expected_rows):
        for query_head in range(24):
            expected[row, query_head] = (
                base_value * column_scales[query_head // 6]
            )
    for row in range(4):
        adjacent_gaps = (expected[row, :, 1:] - expected[row, :, :-1]).abs()
        assert adjacent_gaps.min() > 0.05

    assert torch.isfinite(output).all()
    assert torch.isfinite(repeat).all()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(output, repeat, atol=0, rtol=0)
