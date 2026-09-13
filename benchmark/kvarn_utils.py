# SPDX-License-Identifier: Apache-2.0
"""Independent KVarN packed-record helpers for tests and microbenchmarks.

This module intentionally does not import vLLM's KVarN implementation. It is
the correctness oracle for native XPU kernels that consume the same record.
"""

from __future__ import annotations

import dataclasses
import math

import torch


@dataclasses.dataclass(frozen=True)
class KVarNLayout:
    head_dim: int = 256
    group: int = 128
    key_bits: int = 4
    value_bits: int = 4
    # Optional physical record stride.  Tests use this to exercise both the
    # historical per-token-power-of-two padding and the compact active record
    # without changing any semantic field offsets.
    record_stride: int | None = None

    @property
    def k_packed_bytes(self) -> int:
        return math.ceil(self.head_dim * self.group * self.key_bits / 8)

    @property
    def v_packed_bytes(self) -> int:
        return math.ceil(self.group * self.head_dim * self.value_bits / 8)

    @property
    def k_s_col_offset(self) -> int:
        return self.k_packed_bytes

    @property
    def k_zp_offset(self) -> int:
        return self.k_s_col_offset + self.head_dim * 2

    @property
    def k_s_row_offset(self) -> int:
        return self.k_zp_offset + self.head_dim * 2

    @property
    def v_packed_offset(self) -> int:
        return self.k_s_row_offset + self.group * 2

    @property
    def v_s_col_offset(self) -> int:
        return self.v_packed_offset + self.v_packed_bytes

    @property
    def v_s_row_offset(self) -> int:
        return self.v_s_col_offset + self.head_dim * 2

    @property
    def v_zp_offset(self) -> int:
        return self.v_s_row_offset + self.group * 2

    @property
    def tile_bytes(self) -> int:
        return self.v_zp_offset + self.group * 2

    @property
    def tile_bytes_aligned(self) -> int:
        if self.record_stride is not None:
            if self.record_stride < self.tile_bytes:
                raise ValueError(
                    "record stride must contain the active KVarN record"
                )
            return self.record_stride
        if self.head_dim >= 256:
            slot_bytes = math.ceil(self.tile_bytes / self.group)
            return (1 << (slot_bytes - 1).bit_length()) * self.group
        return (self.tile_bytes + 7) // 8 * 8


def unpack_lowbit(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    """Unpack low-to-high bit fields along the final byte dimension."""
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    fields = [
        torch.bitwise_and(torch.bitwise_right_shift(packed, i * bits), mask)
        for i in range(values_per_byte)
    ]
    return torch.stack(fields, dim=-1).flatten(-2)[..., :count]


def _k_dpas_coord(lane: int, slot: int) -> tuple[int, int]:
    """Frozen Xe2 DPAS K-fragment coordinate for one 16-token SG tile."""
    return lane // 2 + 8 * (slot % 2), 2 * (slot // 2) + lane % 2


def _v_dpas_coord(lane: int, slot: int) -> tuple[int, int]:
    """Frozen Xe2 DPAS V-fragment coordinate for one 32x16 SG tile."""
    inner = slot % 16
    return (
        lane // 2 + 8 * (inner % 2) + 16 * (slot // 16),
        2 * (inner // 2) + lane % 2,
    )


# Cache the lane/slot ownership tables once. Broadcasting these small CPU
# tensors across half/tile/subgroup dimensions avoids millions of scalar
# Tensor.__setitem__ calls when long-context fixtures pack many records.
_K_LANE = torch.arange(16, dtype=torch.int64)[:, None]
_K_SLOT = torch.arange(64, dtype=torch.int64)[None, :]
_K_TOKEN = _K_LANE // 2 + 8 * (_K_SLOT % 2)
_K_DIM = 2 * (_K_SLOT // 2) + _K_LANE % 2
_V_LANE = torch.arange(16, dtype=torch.int64)[:, None]
_V_SLOT = torch.arange(32, dtype=torch.int64)[None, :]
_V_INNER = _V_SLOT % 16
_V_DIM = _V_LANE // 2 + 8 * (_V_INNER % 2) + 16 * (_V_SLOT // 16)
_V_TOKEN = 2 * (_V_INNER // 2) + _V_LANE % 2


def pack_dpas_kv(
    q_k: torch.Tensor, q_v: torch.Tensor, value_bits: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack K4 and V2/V4 in Xe2 B-fragment order, low bits first."""
    if q_k.shape != (256, 128) or q_v.shape != (128, 256):
        raise ValueError("DPAS prototype requires K[256,128] and V[128,256]")
    if q_k.dtype != torch.uint8 or q_v.dtype != torch.uint8:
        raise ValueError("DPAS prototype inputs must have dtype uint8")
    if value_bits not in (2, 4):
        raise ValueError("DPAS values must use two or four bits")
    if bool((q_k > 15).any()) or bool((q_v >= 1 << value_bits).any()):
        raise ValueError("DPAS inputs exceed their declared bit width")

    half = torch.arange(2, dtype=torch.int64)[:, None, None, None, None]
    subgroup = torch.arange(4, dtype=torch.int64)[None, None, :, None, None]
    k_tile = torch.arange(4, dtype=torch.int64)[None, :, None, None, None]
    v_tile = torch.arange(8, dtype=torch.int64)[None, :, None, None, None]
    k_slots = q_k[
        k_tile * 64 + _K_DIM[None, None, None, :, :],
        half * 64 + subgroup * 16 + _K_TOKEN[None, None, None, :, :],
    ]
    v_slots = q_v[
        half * 64 + subgroup * 16 + _V_TOKEN[None, None, None, :, :],
        v_tile * 32 + _V_DIM[None, None, None, :, :],
    ]
    k_bytes = k_slots[..., 0::2] | (k_slots[..., 1::2] << 4)
    fields = 8 // value_bits
    v_bytes = torch.zeros_like(v_slots[..., ::fields])
    for field in range(fields):
        v_bytes |= v_slots[..., field::fields] << (field * value_bits)
    return k_bytes.flatten(), v_bytes.flatten()


def unpack_dpas_kv(
    k_packed: torch.Tensor, v_packed: torch.Tensor, value_bits: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert DPAS payloads into logical K[D,G]/V[G,D] integer codes."""
    if value_bits not in (2, 4):
        raise ValueError("DPAS values must use two or four bits")
    if (
        k_packed.dtype != torch.uint8
        or v_packed.dtype != torch.uint8
        or k_packed.numel() != 256 * 128 // 2
        or v_packed.numel() != 128 * 256 * value_bits // 8
    ):
        raise ValueError(
            "DPAS payload sizes must match D256/G128 and the bit widths"
        )
    k_bytes = k_packed.reshape(2, 4, 4, 16, 32)
    v_bytes = v_packed.reshape(2, 8, 4, 16, 4 * value_bits)
    k_slots = torch.stack((k_bytes & 15, k_bytes >> 4), dim=-1).flatten(-2)
    v_slots = unpack_lowbit(v_bytes, 32, value_bits)
    q_k = torch.empty((256, 128), dtype=torch.uint8)
    q_v = torch.empty((128, 256), dtype=torch.uint8)
    for half in range(2):
        for tile in range(4):
            for subgroup in range(4):
                for lane in range(16):
                    for slot in range(64):
                        token, dim = _k_dpas_coord(lane, slot)
                        q_k[
                            tile * 64 + dim, half * 64 + subgroup * 16 + token
                        ] = k_slots[half, tile, subgroup, lane, slot]
    for half in range(2):
        for tile in range(8):
            for subgroup in range(4):
                for lane in range(16):
                    for slot in range(32):
                        dim, token = _v_dpas_coord(lane, slot)
                        q_v[
                            half * 64 + subgroup * 16 + token, tile * 32 + dim
                        ] = v_slots[half, tile, subgroup, lane, slot]
    return q_k, q_v


def swizzle_record_dpas_kv(
    record: torch.Tensor, layout: KVarNLayout
) -> torch.Tensor:
    """Copy a canonical record and replace only its K/V payload orientation."""
    if record.dtype != torch.uint8 or record.ndim != 1:
        raise ValueError("record must be a one-dimensional uint8 tensor")
    if (layout.head_dim, layout.group, layout.key_bits) != (256, 128, 4) or (
        layout.value_bits not in (2, 4)
    ):
        raise ValueError("DPAS layout requires D256/G128/K4 and V2/V4")
    result = record.clone()
    canonical_k = record[: layout.k_packed_bytes].reshape(256, 64)
    canonical_v = record[
        layout.v_packed_offset : layout.v_packed_offset + layout.v_packed_bytes
    ].reshape(128, 32 * layout.value_bits)
    if not bool(canonical_k.any()) and not bool(canonical_v.any()):
        return result
    q_k = unpack_lowbit(canonical_k, 128, 4)
    q_v = unpack_lowbit(canonical_v, 256, layout.value_bits)
    k_packed, v_packed = pack_dpas_kv(q_k, q_v, layout.value_bits)
    result[: layout.k_packed_bytes] = k_packed
    result[
        layout.v_packed_offset : layout.v_packed_offset + layout.v_packed_bytes
    ] = v_packed
    return result


def pack_dpas_k4v4(
    q_k: torch.Tensor, q_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return pack_dpas_kv(q_k, q_v, 4)


def unpack_dpas_k4v4(
    k_packed: torch.Tensor, v_packed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return unpack_dpas_kv(k_packed, v_packed, 4)


def swizzle_record_dpas_k4v4(
    record: torch.Tensor, layout: KVarNLayout
) -> torch.Tensor:
    if layout.value_bits != 4:
        raise ValueError("K4V4 swizzle requires four-bit values")
    return swizzle_record_dpas_kv(record, layout)


def _half_field(record: torch.Tensor, offset: int, count: int) -> torch.Tensor:
    raw = record[offset : offset + count * 2].contiguous()
    return raw.view(torch.float16).float()


def dequant_record(
    record: torch.Tensor, layout: KVarNLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rotated K [G,D] and V [G,D] from one uint8 record."""
    if record.dtype != torch.uint8 or record.ndim != 1:
        raise ValueError("record must be a one-dimensional uint8 tensor")
    if record.numel() < layout.tile_bytes:
        raise ValueError("record is shorter than the declared KVarN layout")

    k_bytes_per_row = math.ceil(layout.group * layout.key_bits / 8)
    k_packed = record[: layout.k_packed_bytes].reshape(
        layout.head_dim, k_bytes_per_row
    )
    q_k = unpack_lowbit(k_packed, layout.group, layout.key_bits).float()
    k_s_col = _half_field(record, layout.k_s_col_offset, layout.head_dim)
    k_zp = _half_field(record, layout.k_zp_offset, layout.head_dim)
    k_s_row = _half_field(record, layout.k_s_row_offset, layout.group)
    k = (q_k * k_s_col[:, None] + k_zp[:, None]) * k_s_row[None, :]

    v_bytes_per_row = math.ceil(layout.head_dim * layout.value_bits / 8)
    v_packed = record[
        layout.v_packed_offset : layout.v_packed_offset + layout.v_packed_bytes
    ].reshape(layout.group, v_bytes_per_row)
    q_v = unpack_lowbit(v_packed, layout.head_dim, layout.value_bits).float()
    v_s_col = _half_field(record, layout.v_s_col_offset, layout.head_dim)
    v_s_row = _half_field(record, layout.v_s_row_offset, layout.group)
    v_zp = _half_field(record, layout.v_zp_offset, layout.group)
    v = (q_v * v_s_row[:, None] + v_zp[:, None]) * v_s_col[None, :]
    return k.transpose(0, 1).contiguous(), v
