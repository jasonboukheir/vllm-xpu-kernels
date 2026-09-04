# SPDX-License-Identifier: Apache-2.0
"""Host-only contracts for the isolated Xe2 Q6 paired-nibble candidate."""

from __future__ import annotations

import struct
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
MAINLOOP = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/collective/kvarn_decode_mainloop.hpp"
).read_text()
CONFIG = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode.hpp").read_text()
DISPATCH = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode_xe2.cpp").read_text()


def _half_bits(value: int) -> int:
    return int.from_bytes(struct.pack("<e", float(value)), "little")


def _paired_half2_bits(packed: int) -> int:
    return _half_bits(packed & 0x0F) | (_half_bits(packed >> 4) << 16)


def test_all_256_packed_bytes_expand_to_exact_unsigned_nibbles() -> None:
    expected_half_bits = (
        0x0000,
        0x3C00,
        0x4000,
        0x4200,
        0x4400,
        0x4500,
        0x4600,
        0x4700,
        0x4800,
        0x4880,
        0x4900,
        0x4980,
        0x4A00,
        0x4A80,
        0x4B00,
        0x4B80,
    )
    assert tuple(_half_bits(value) for value in range(16)) == expected_half_bits
    for packed in range(256):
        pair = _paired_half2_bits(packed)
        assert pair & 0xFFFF == expected_half_bits[packed & 0x0F]
        assert pair >> 16 == expected_half_bits[packed >> 4]

    assert "q6_paired_nibble_half2_lut_is_exact" in CONFIG
    assert "static_assert(q6_paired_nibble_half2_lut_is_exact());" in CONFIG


def test_id20_and_id21_are_independently_runtime_selectable() -> None:
    assert "kQ6PageMetadataCursor = 20" in DISPATCH
    assert "kQ6PairedNibbleHalf2 = 21" in DISPATCH
    assert "bool const use_q6_page_metadata_cursor" in DISPATCH
    assert "bool const use_q6_paired_nibble_half2" in DISPATCH
    assert (
        "KVarNDecodeD256G128DpasQ6PairedNibbleHalf2Config::run(queue, args)"
        in DISPATCH
    )


def test_id21_composes_with_id18_without_changing_cache_abi() -> None:
    assert "KVarNDecodeD256G128DpasQ6PairedNibbleHalf2Config" in CONFIG
    for contract in (
        "PairedNibbleHalf2",
        "NextPagePrefetch",
        "CurrentHalfVPrefetch",
        "ReusePageRecordCursor",
        "UsesSpecializedSplitReducer",
    ):
        assert (
            "KVarNDecodeD256G128DpasQ6PairedNibbleHalf2Config" in CONFIG
            and contract in CONFIG
        )
    assert "!SimdPackedUnpack && !PagePair" in CONFIG
    assert "KVarNK4V4Layout" in MAINLOOP
    assert "paired_nibble_half2_lut[256]" in MAINLOOP
    assert "256 * sizeof(std::uint32_t)" in CONFIG
    assert "BaseStorage base;" in MAINLOOP


def test_hot_loop_performs_one_lookup_per_packed_byte() -> None:
    start = MAINLOOP.index("fill_paired_nibble_half2_fragment")
    end = MAINLOOP.index("fill_packed_lane_fragment", start)
    paired_path = MAINLOOP[start:end]
    assert "sycl::vec<std::uint8_t, kBytesPerChunk>" in paired_path
    assert "paired_nibble_half2_lut[bytes[byte]]" in paired_path
    assert "sycl::bit_cast<Half2>" in paired_path
    assert "unpack_nibble" not in paired_path
    assert ".template convert<sycl::half>()" not in paired_path


def test_established_id18_does_not_enable_paired_lookup() -> None:
    id18_start = CONFIG.index(
        "using KVarNDecodeD256G128DpasQ6PrefetchRecordCursorConfig"
    )
    id20_start = CONFIG.index(
        "using KVarNDecodeD256G128DpasQ6PageMetadataCursorConfig", id18_start
    )
    id18_alias = CONFIG[id18_start:id20_start]
    # ID18 ends at ReusePageRecordCursor=true; both later experiment flags use
    # their defaults, so its producer and reducer remain untouched.
    assert id18_alias.rstrip().endswith("true>;")
    assert "PairedNibbleHalf2" not in id18_alias
    assert "CUTLASS_DEVICE static void\n  fill_packed_lane_fragment" in MAINLOOP
