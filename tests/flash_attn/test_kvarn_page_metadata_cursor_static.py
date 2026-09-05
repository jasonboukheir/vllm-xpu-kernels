# SPDX-License-Identifier: Apache-2.0
"""Host-only contract proofs for the isolated ID20 metadata cursor."""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
MAINLOOP = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/collective/kvarn_decode_mainloop.hpp"
).read_text()
CONFIG = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode.hpp").read_text()
DISPATCH = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode_xe2.cpp"
).read_text()


def _metadata_refresh_tiles(first_tile: int, last_tile: int) -> list[int]:
    """Model the workgroup-uniform refresh predicate in the K64 loop."""
    return [
        tile
        for tile in range(first_tile, last_tile)
        if tile == first_tile or tile % 2 == 0
    ]


def test_id20_is_an_independent_runtime_selection() -> None:
    assert "kQ6PageMetadataCursor = 20" in DISPATCH
    assert "use_q6_page_metadata_cursor" in DISPATCH
    assert (
        "KVarNDecodeD256G128DpasQ6PageMetadataCursorConfig::run" in DISPATCH
    )
    assert "q6_page_metadata_cursor" in DISPATCH


def test_id20_adds_only_metadata_reuse_to_the_id18_reader() -> None:
    assert (
        "!KVarNDecodeD256G128DpasQ6PrefetchRecordCursorConfig::Mainloop::\n"
        "                  ReusePageMetadataCursor"
        in CONFIG
    )
    for flag in (
        "NextPagePrefetch",
        "CurrentHalfVPrefetch",
        "ReusePageRecordCursor",
        "ReusePageMetadataCursor",
    ):
        assert (
            "KVarNDecodeD256G128DpasQ6PageMetadataCursorConfig::Mainloop::\n"
            f"                  {flag}"
            in CONFIG
        )
    assert "B1ShortLastProducer" not in CONFIG


def test_page_metadata_refreshes_once_per_owned_physical_page() -> None:
    assert "bool const refresh_page_metadata =" in MAINLOOP
    assert "(k_tile & 1) == 0 || k_tile == blk_k0" in MAINLOOP

    for first_tile, last_tile in ((0, 8), (1, 8), (5, 6), (5, 11)):
        touched_pages = {tile // 2 for tile in range(first_tile, last_tile)}
        refreshed_pages = {
            tile // 2
            for tile in _metadata_refresh_tiles(first_tile, last_tile)
        }
        assert refreshed_pages == touched_pages
        assert len(_metadata_refresh_tiles(first_tile, last_tile)) == len(
            touched_pages
        )


def test_cursor_keeps_page_wide_metadata_but_not_a_second_score() -> None:
    assert "KDimMetadataFragment page_k_dim_scale[4]" in MAINLOOP
    assert "float page_k_zp_bias[BiasRows]" in MAINLOOP
    assert "VDimMetadataFragment page_v_dim_scale[VTiles]" in MAINLOOP
    assert "if (refresh_page_metadata)" in MAINLOOP
    assert "k_zp_bias[query_row] = page_k_zp_bias[query_row]" in MAINLOOP
    assert "v_dim_scale(i) = page_v_dim_scale[vv](i)" in MAINLOOP

    # The only extra score fragment belongs to the old PagePair experiment.
    # ID20 stays on the ordinary K64 scheduler and consumes each half before
    # advancing the loop.
    assert MAINLOOP.count("auto tSrSSecond") == 1
    assert (
        "static_assert(!ReusePageMetadataCursor || !PagePair)" in MAINLOOP
    )


def test_second_half_prefetch_omits_only_reused_column_scale() -> None:
    assert "template <bool IncludeColumnScale = true>" in MAINLOOP
    assert "prefetch_dpas_v_half_l1<false>" in MAINLOOP
    assert "prefetch_dpas_v_half_l1<true>" in MAINLOOP
    assert "rec + layout.v_s_row_offset + half * kHalfRowBytes" in MAINLOOP
    assert "rec + layout.v_zp_offset + half * kHalfRowBytes" in MAINLOOP


def test_metadata_cursor_does_not_change_cache_or_launch_abi() -> None:
    for abi_type in ("Params", "Arguments", "SharedStorage"):
        assert re.search(
            r"sizeof\(\s*KVarNDecodeD256G128DpasQ6PageMetadataCursorConfig::"
            rf"Mainloop::\s*{abi_type}\)\s*==\s*sizeof\(\s*"
            r"KVarNDecodeD256G128DpasQ6PrefetchRecordCursorConfig::"
            rf"Mainloop::\s*{abi_type}\)",
            CONFIG,
        )
    assert "static_assert(kActiveRecordBytes == 35072)" in MAINLOOP
    assert "KVarNK4V4Layout layout;" in MAINLOOP
    assert "KVarNHybridTailLayout tail;" in MAINLOOP
