# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from benchmark.kvarn_utils import (KVarNLayout, _k_dpas_coord,  # noqa: E402
                                   _v_dpas_coord, dequant_record,
                                   pack_dpas_k4v4, unpack_dpas_k4v4)


def _canonical_record(q_k: torch.Tensor, q_v: torch.Tensor) -> torch.Tensor:
    layout = KVarNLayout()
    record = torch.zeros(layout.tile_bytes_aligned, dtype=torch.uint8)
    record[: layout.k_packed_bytes] = (
        q_k[:, 0::2] | (q_k[:, 1::2] << 4)
    ).flatten()
    record[
        layout.v_packed_offset : layout.v_packed_offset + layout.v_packed_bytes
    ] = (q_v[:, 0::2] | (q_v[:, 1::2] << 4)).flatten()

    def put_half(offset: int, values: torch.Tensor) -> None:
        raw = values.half().contiguous().view(torch.uint8)
        record[offset : offset + raw.numel()] = raw

    generator = torch.Generator().manual_seed(20260808)
    put_half(layout.k_s_col_offset, torch.rand(256, generator=generator) + 0.1)
    put_half(layout.k_zp_offset, torch.rand(256, generator=generator) - 0.5)
    put_half(layout.k_s_row_offset, torch.rand(128, generator=generator) + 0.1)
    put_half(layout.v_s_col_offset, torch.rand(256, generator=generator) + 0.1)
    put_half(layout.v_s_row_offset, torch.rand(128, generator=generator) + 0.1)
    put_half(layout.v_zp_offset, torch.rand(128, generator=generator) - 0.5)
    return record


def test_frozen_dpas_coordinate_formulas_are_bijections() -> None:
    k_coords = {
        _k_dpas_coord(lane, slot) for lane in range(16) for slot in range(64)
    }
    v_coords = {
        _v_dpas_coord(lane, slot) for lane in range(16) for slot in range(32)
    }
    assert k_coords == {
        (token, dim) for token in range(16) for dim in range(64)
    }
    assert v_coords == {
        (dim, token) for dim in range(32) for token in range(16)
    }


def test_dpas_pack_roundtrip_random_and_all_nibbles() -> None:
    generator = torch.Generator().manual_seed(314159)
    q_k = torch.randint(
        0, 16, (256, 128), dtype=torch.uint8, generator=generator
    )
    q_v = torch.randint(
        0, 16, (128, 256), dtype=torch.uint8, generator=generator
    )
    q_k[:, :16] = torch.arange(16, dtype=torch.uint8).repeat(256, 1)
    q_v[:16, :] = torch.arange(16, dtype=torch.uint8)[:, None]

    k_packed, v_packed = pack_dpas_k4v4(q_k, q_v)
    assert k_packed.numel() == 256 * 128 // 2
    assert v_packed.numel() == 128 * 256 // 2
    actual_k, actual_v = unpack_dpas_k4v4(k_packed, v_packed)
    torch.testing.assert_close(actual_k, q_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v, q_v, rtol=0, atol=0)


def test_dpas_physical_offsets_cover_every_boundary_axis() -> None:
    q_k = torch.zeros((256, 128), dtype=torch.uint8)
    q_v = torch.zeros((128, 256), dtype=torch.uint8)
    cases = []
    for half in (0, 1):
        for tile in (0, 3):
            for subgroup in (0, 3):
                for lane in (0, 15):
                    for slot in (0, 63):
                        token, dim = _k_dpas_coord(lane, slot)
                        value = 1 + len(cases) % 15
                        q_k[
                            tile * 64 + dim, half * 64 + subgroup * 16 + token
                        ] = value
                        cases.append((half, tile, subgroup, lane, slot, value))
    v_cases = []
    for half in (0, 1):
        for tile in (0, 7):
            for subgroup in (0, 3):
                for lane in (0, 15):
                    for slot in (0, 31):
                        dim, token = _v_dpas_coord(lane, slot)
                        value = 1 + len(v_cases) % 15
                        q_v[
                            half * 64 + subgroup * 16 + token, tile * 32 + dim
                        ] = value
                        v_cases.append(
                            (half, tile, subgroup, lane, slot, value)
                        )
    k_packed, v_packed = pack_dpas_k4v4(q_k, q_v)
    for half, tile, subgroup, lane, slot, value in cases:
        byte = (((half * 4 + tile) * 4 + subgroup) * 16 + lane) * 32 + slot // 2
        assert int((k_packed[byte] >> (4 * (slot % 2))) & 15) == value
    for half, tile, subgroup, lane, slot, value in v_cases:
        byte = (((half * 8 + tile) * 4 + subgroup) * 16 + lane) * 16 + slot // 2
        assert int((v_packed[byte] >> (4 * (slot % 2))) & 15) == value


def test_swizzled_and_canonical_dequant_are_equal() -> None:
    q_k = (
        torch.arange(256 * 128, dtype=torch.int64).reshape(256, 128).byte() & 15
    )
    q_v = (
        torch.arange(128 * 256, dtype=torch.int64).reshape(128, 256).byte() * 7
    ) & 15
    k_packed, v_packed = pack_dpas_k4v4(q_k, q_v)
    roundtrip_k, roundtrip_v = unpack_dpas_k4v4(k_packed, v_packed)
    canonical = _canonical_record(q_k, q_v)
    reconstructed = _canonical_record(roundtrip_k, roundtrip_v)
    expected_k, expected_v = dequant_record(canonical, KVarNLayout())
    actual_k, actual_v = dequant_record(reconstructed, KVarNLayout())
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
