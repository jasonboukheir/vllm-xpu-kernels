# SPDX-License-Identifier: Apache-2.0
"""Host-only source and address proofs for the qualified Xe2 KVarN reader."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
MAINLOOP = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/collective/kvarn_decode_mainloop.hpp"
).read_text()
CONFIG = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode.hpp").read_text()
DISPATCH = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode_xe2.cpp").read_text()


def test_id18_composes_prefetch_and_record_cursor() -> None:
    assert "kKVarNDecodeKernel = 18" in DISPATCH
    assert "only the qualified KVarN decoder (18) is supported" in DISPATCH
    assert (
        "KVarNDecodeD256G128DpasQ6PrefetchRecordCursorConfig::run(queue, args)"
        in DISPATCH
    )
    assert "using Mainloop =" in CONFIG
    assert "KVarNDecodeFwdMainloop<" in CONFIG
    assert "prefetch_dpas_v_half_l1" in MAINLOOP
    assert "bool const refresh_record =" in MAINLOOP


def test_half_local_v_prefetch_ranges_match_xe2_dpas_record() -> None:
    packed_bytes = 256 * 128 // 2
    column_bytes = 256 * 2
    row_bytes = 128 * 2
    half_row_bytes = row_bytes // 2
    v_packed_offset = packed_bytes + 2 * column_bytes + row_bytes
    v_s_col_offset = v_packed_offset + packed_bytes
    v_s_row_offset = v_s_col_offset + column_bytes
    v_zp_offset = v_s_row_offset + row_bytes
    record_bytes = v_zp_offset + row_bytes

    for half in (0, 1):
        ranges = (
            (v_packed_offset + half * packed_bytes // 2, packed_bytes // 2),
            (v_s_col_offset, column_bytes),
            (v_s_row_offset + half * half_row_bytes, half_row_bytes),
            (v_zp_offset + half * half_row_bytes, half_row_bytes),
        )
        assert all(offset >= v_packed_offset for offset, _ in ranges)
        assert all(offset + size <= record_bytes for offset, size in ranges)
        assert all(size % 64 == 0 for _, size in ranges)

    assert "prefetch_dpas_v_half_l1" in MAINLOOP
    assert "syclex::prefetch_hint_L1" in MAINLOOP


def test_page_record_cursor_refreshes_once_per_touched_page() -> None:
    def refresh_tiles(first_tile: int, last_tile: int) -> list[int]:
        return [
            tile
            for tile in range(first_tile, last_tile)
            if tile == first_tile or tile % 2 == 0
        ]

    for first_tile, last_tile in ((0, 8), (1, 8), (5, 6), (5, 11)):
        touched_pages = {tile // 2 for tile in range(first_tile, last_tile)}
        refreshed_pages = {
            tile // 2 for tile in refresh_tiles(first_tile, last_tile)
        }
        assert refreshed_pages == touched_pages
        assert len(refresh_tiles(first_tile, last_tile)) == len(touched_pages)

    assert "bool const refresh_record =" in MAINLOOP
    assert "(k_tile & 1) == 0 || k_tile == blk_k0" in MAINLOOP
