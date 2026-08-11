"""Independent KVarN K4V4 packed-layout tests for native XPU decode."""

import struct
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmark.kvarn_utils import KVarNLayout, dequant_record  # noqa: E402


HEAD_DIM = 256
GROUP = 128
K_PACKED_BYTES = HEAD_DIM * GROUP // 2
V_PACKED_BYTES = GROUP * HEAD_DIM // 2
K_S_COL_OFFSET = K_PACKED_BYTES
K_ZP_OFFSET = K_S_COL_OFFSET + HEAD_DIM * 2
K_S_ROW_OFFSET = K_ZP_OFFSET + HEAD_DIM * 2
V_PACKED_OFFSET = K_S_ROW_OFFSET + GROUP * 2
V_S_COL_OFFSET = V_PACKED_OFFSET + V_PACKED_BYTES
V_S_ROW_OFFSET = V_S_COL_OFFSET + HEAD_DIM * 2
V_ZP_OFFSET = V_S_ROW_OFFSET + GROUP * 2
TILE_BYTES = V_ZP_OFFSET + GROUP * 2
TILE_BYTES_ALIGNED = 65536


def _unpack_nibbles(data: bytes, count: int) -> list[int]:
    values: list[int] = []
    for value in data:
        values.extend((value & 0xF, value >> 4))
    return values[:count]


def _half_array(record: bytes, offset: int, count: int) -> tuple[float, ...]:
    return struct.unpack_from(f"<{count}e", record, offset)


def test_qwen36_k4v4_record_offsets_and_alignment() -> None:
    assert K_PACKED_BYTES == 16384
    assert K_S_COL_OFFSET == 16384
    assert K_ZP_OFFSET == 16896
    assert K_S_ROW_OFFSET == 17408
    assert V_PACKED_OFFSET == 17664
    assert V_S_COL_OFFSET == 34048
    assert V_S_ROW_OFFSET == 34560
    assert V_ZP_OFFSET == 34816
    assert TILE_BYTES == 35072
    # D256 uses a power-of-two per-token slot: 35072/128 -> 512 bytes.
    assert TILE_BYTES_ALIGNED == 65536


@pytest.mark.parametrize("low,high", [(0, 15), (3, 12), (7, 8)])
def test_low_nibble_precedes_high_nibble(low: int, high: int) -> None:
    packed = bytes([low | (high << 4)])
    assert _unpack_nibbles(packed, 2) == [low, high]


def test_scale_fields_are_little_endian_fp16() -> None:
    record = bytearray(TILE_BYTES_ALIGNED)
    struct.pack_into("<3e", record, K_S_COL_OFFSET, 0.5, -1.0, 2.0)
    assert _half_array(record, K_S_COL_OFFSET, 3) == (0.5, -1.0, 2.0)


def _set_half(record: torch.Tensor, offset: int, values: torch.Tensor) -> None:
    encoded = values.to(torch.float16).contiguous().view(torch.uint8)
    record[offset : offset + encoded.numel()].copy_(encoded)


def test_independent_dequant_oracle_applies_k_and_v_axes() -> None:
    layout = KVarNLayout(head_dim=4, group=4)
    record = torch.zeros(layout.tile_bytes_aligned, dtype=torch.uint8)
    # K and V use every representable nibble value exactly once.
    quantized = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
    packed = quantized[:, 0::2] | (quantized[:, 1::2] << 4)
    record[: layout.k_packed_bytes].copy_(packed.flatten())
    record[
        layout.v_packed_offset : layout.v_packed_offset + layout.v_packed_bytes
    ].copy_(packed.flatten())

    _set_half(record, layout.k_s_col_offset, torch.tensor([1, 2, 3, 4]))
    _set_half(record, layout.k_zp_offset, torch.tensor([0, 1, 2, 3]))
    _set_half(record, layout.k_s_row_offset, torch.tensor([1, 2, 3, 4]))
    _set_half(record, layout.v_s_col_offset, torch.tensor([1, 2, 3, 4]))
    _set_half(record, layout.v_s_row_offset, torch.tensor([1, 2, 3, 4]))
    _set_half(record, layout.v_zp_offset, torch.tensor([0, 1, 2, 3]))

    key, value = dequant_record(record, layout)

    expected_key = (
        quantized.float() * torch.tensor([1, 2, 3, 4])[:, None]
        + torch.tensor([0, 1, 2, 3])[:, None]
    ) * torch.tensor([1, 2, 3, 4])[None, :]
    expected_value = (
        quantized.float() * torch.tensor([1, 2, 3, 4])[:, None]
        + torch.tensor([0, 1, 2, 3])[:, None]
    ) * torch.tensor([1, 2, 3, 4])[None, :]
    torch.testing.assert_close(key, expected_key.T)
    torch.testing.assert_close(value, expected_value)
