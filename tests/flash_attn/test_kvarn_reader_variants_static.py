# SPDX-License-Identifier: Apache-2.0
"""Host-only source and address proofs for Xe2 KVarN reader variants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
MAINLOOP = (
    REPO_ROOT
    / "csrc/xpu/attn/xe_2/collective/kvarn_decode_mainloop.hpp"
).read_text()
CONFIG = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode.hpp").read_text()
DISPATCH = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode_xe2.cpp").read_text()


def test_id13_remains_the_unmodified_reader_control() -> None:
    assert "kQ6NextPagePrefetchSplitReducer = 13" in DISPATCH
    assert (
        "!KVarNDecodeD256G128DpasQ6NextPagePrefetchSplitReducerConfig::\n"
        "                  Mainloop::CurrentHalfVPrefetch"
    ) in CONFIG
    assert (
        "!KVarNDecodeD256G128DpasQ6NextPagePrefetchSplitReducerConfig::\n"
        "                  Mainloop::ReusePageRecordCursor"
    ) in CONFIG


def test_id16_selects_only_current_half_v_prefetch_over_id13() -> None:
    assert "kQ6CurrentHalfVPrefetch = 16" in DISPATCH
    assert "use_q6_current_half_v_prefetch" in DISPATCH
    assert (
        "KVarNDecodeD256G128DpasQ6CurrentHalfVPrefetchConfig::Mainloop::\n"
        "                  CurrentHalfVPrefetch"
    ) in CONFIG
    assert (
        "!KVarNDecodeD256G128DpasQ6CurrentHalfVPrefetchConfig::Mainloop::\n"
        "                  ReusePageRecordCursor"
    ) in CONFIG


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
