"""CPU contracts for compact and padded KVarN K4V4 records.

These tests deliberately use the independent benchmark oracle.  They freeze
the storage contract needed by serving before production allocation and XPU
reader/writer code switches to the compact stride.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmark.kvarn_utils import (  # noqa: E402
    KVarNLayout,
    dequant_record,
    pack_dpas_k4v4,
    swizzle_record_dpas_k4v4,
    unpack_dpas_k4v4,
)


COMPACT_STRIDE = 35072
PADDED_STRIDE = 65536


def _put_half(record: torch.Tensor, offset: int, values: torch.Tensor) -> None:
    raw = values.half().contiguous().view(torch.uint8)
    record[offset : offset + raw.numel()].copy_(raw)


def _canonical_record(layout: KVarNLayout) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260809)
    qk = torch.randint(
        0, 16, (256, 128), generator=generator, dtype=torch.uint8
    )
    qv = torch.randint(
        0, 16, (128, 256), generator=generator, dtype=torch.uint8
    )
    record = torch.zeros(layout.tile_bytes_aligned, dtype=torch.uint8)
    record[: layout.k_packed_bytes] = (qk[:, 0::2] | qk[:, 1::2] << 4).flatten()
    record[
        layout.v_packed_offset : layout.v_packed_offset + layout.v_packed_bytes
    ] = (qv[:, 0::2] | qv[:, 1::2] << 4).flatten()
    _put_half(
        record,
        layout.k_s_col_offset,
        torch.rand(256, generator=generator) + 0.1,
    )
    _put_half(
        record, layout.k_zp_offset, torch.rand(256, generator=generator) - 0.5
    )
    _put_half(
        record,
        layout.k_s_row_offset,
        torch.rand(128, generator=generator) + 0.1,
    )
    _put_half(
        record,
        layout.v_s_col_offset,
        torch.rand(256, generator=generator) + 0.1,
    )
    _put_half(
        record,
        layout.v_s_row_offset,
        torch.rand(128, generator=generator) + 0.1,
    )
    _put_half(
        record, layout.v_zp_offset, torch.rand(128, generator=generator) - 0.5
    )
    return record


@pytest.mark.parametrize("stride", [COMPACT_STRIDE, PADDED_STRIDE])
def test_stride_selection_preserves_all_semantic_offsets(stride: int) -> None:
    selected = KVarNLayout(record_stride=stride)
    baseline = KVarNLayout(record_stride=PADDED_STRIDE)
    assert selected.tile_bytes == COMPACT_STRIDE
    assert selected.tile_bytes_aligned == stride
    for field in (
        "k_s_col_offset",
        "k_zp_offset",
        "k_s_row_offset",
        "v_packed_offset",
        "v_s_col_offset",
        "v_s_row_offset",
        "v_zp_offset",
    ):
        assert getattr(selected, field) == getattr(baseline, field)


def test_stride_rejects_truncating_the_active_record() -> None:
    layout = KVarNLayout(record_stride=COMPACT_STRIDE - 1)
    with pytest.raises(ValueError, match="contain the active KVarN record"):
        _ = layout.tile_bytes_aligned


@pytest.mark.parametrize("stride", [COMPACT_STRIDE, PADDED_STRIDE])
def test_cache_shape_and_allocation_math_use_selected_stride(
    stride: int,
) -> None:
    layout = KVarNLayout(record_stride=stride)
    num_blocks, block_size, heads = 13, 256, 4
    tiles_per_block = block_size // layout.group
    shape = (num_blocks * tiles_per_block, heads, layout.tile_bytes_aligned)
    cache = torch.empty(shape, dtype=torch.uint8)
    assert shape == (26, 4, stride)
    assert cache.numel() == 26 * 4 * stride
    assert cache.untyped_storage().nbytes() == 26 * 4 * stride


def test_compact_memory_ratio_is_exact() -> None:
    assert Fraction(COMPACT_STRIDE, PADDED_STRIDE) == Fraction(137, 256)
    assert Fraction(PADDED_STRIDE - COMPACT_STRIDE, PADDED_STRIDE) == Fraction(
        119, 256
    )
    assert Fraction(PADDED_STRIDE, COMPACT_STRIDE) == Fraction(256, 137)


@pytest.mark.parametrize("stride", [COMPACT_STRIDE, PADDED_STRIDE])
@pytest.mark.parametrize("physical_layout", ["canonical", "dpas"])
def test_pack_dequant_roundtrip_is_stride_independent(
    stride: int, physical_layout: str
) -> None:
    layout = KVarNLayout(record_stride=stride)
    canonical = _canonical_record(layout)
    expected_k, expected_v = dequant_record(canonical, layout)
    if physical_layout == "canonical":
        logical = canonical
    else:
        physical = swizzle_record_dpas_k4v4(canonical, layout)
        k_payload = physical[: layout.k_packed_bytes]
        v_payload = physical[
            layout.v_packed_offset : layout.v_packed_offset
            + layout.v_packed_bytes
        ]
        qk, qv = unpack_dpas_k4v4(k_payload, v_payload)
        logical = physical.clone()
        logical[: layout.k_packed_bytes] = (
            qk[:, 0::2] | qk[:, 1::2] << 4
        ).flatten()
        logical[
            layout.v_packed_offset : layout.v_packed_offset
            + layout.v_packed_bytes
        ] = (qv[:, 0::2] | qv[:, 1::2] << 4).flatten()
        # Independently prove the physical payload contains every logical q.
        dpas_k, dpas_v = pack_dpas_k4v4(qk, qv)
        torch.testing.assert_close(dpas_k, k_payload, rtol=0, atol=0)
        torch.testing.assert_close(dpas_v, v_payload, rtol=0, atol=0)
    actual_k, actual_v = dequant_record(logical, layout)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)


@pytest.mark.parametrize("stride", [COMPACT_STRIDE, PADDED_STRIDE])
@pytest.mark.parametrize("physical_layout", ["canonical", "dpas"])
def test_adjacent_head_and_block_records_are_isolated(
    stride: int, physical_layout: str
) -> None:
    layout = KVarNLayout(record_stride=stride)
    blocks, heads = 3, 4
    sentinel = 0xA7
    cache = torch.full((blocks, heads, stride), sentinel, dtype=torch.uint8)
    target = _canonical_record(layout)
    if physical_layout == "dpas":
        target = swizzle_record_dpas_k4v4(target, layout)
    cache[1, 2].copy_(target)

    assert torch.equal(cache[1, 2], target)
    untouched = torch.ones((blocks, heads), dtype=torch.bool)
    untouched[1, 2] = False
    assert bool((cache[untouched] == sentinel).all())
    flat = cache.flatten()
    start = (1 * heads + 2) * stride
    assert bool((flat[:start] == sentinel).all())
    assert bool((flat[start + stride :] == sentinel).all())
