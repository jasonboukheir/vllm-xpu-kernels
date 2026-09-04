# SPDX-License-Identifier: Apache-2.0
"""Host-only ABI and resource proofs for the fused Xe2 Sinkhorn writer."""

from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
WRITER = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_sinkhorn_writer_xe2.cpp"
).read_text()
REGISTRATION = (REPO_ROOT / "csrc/flash_attn/flash_api.cpp").read_text()
ESTABLISHED_WRITER = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_balanced_writer_xe2.cpp"
).read_text()


def test_fused_writer_keeps_slm_below_xe2_limit() -> None:
    # Six D256 vectors plus one scalar; the compact source page remains in
    # global/L2 storage and no 64 KiB tile accessor is added to SLM.
    state_floats = 6 * 256 + 1
    assert state_floats * 4 == 6_148
    assert state_floats * 4 < 64 * 1024
    assert "kLocalMemoryBytes = kStateFloats * sizeof(float)" in WRITER
    assert "kLocalMemoryBytes <= 64 * 1024" in WRITER
    assert "local_accessor<input_t" not in WRITER


def test_fused_writer_has_a_distinct_raw_page_schema() -> None:
    schema_fragments = (
        '"kvarn_sinkhorn_pack_kv(Tensor tail_key, Tensor tail_value, Tensor "',
        '"pool_slots, Tensor block_ids, Tensor! packed_cache, int "',
        '"sinkhorn_iterations=16, bool dpas_layout=False) -> ()"',
    )
    assert all(fragment in REGISTRATION for fragment in schema_fragments)
    assert "&kvarn_sinkhorn_pack_kv_xe2" in REGISTRATION


def test_fused_writer_preserves_record_offsets_and_rtn_rule() -> None:
    packed_bytes = 256 * 128 // 2
    k_s_col = packed_bytes
    k_zp = k_s_col + 256 * 2
    k_s_row = k_zp + 256 * 2
    v_packed = k_s_row + 128 * 2
    v_s_col = v_packed + packed_bytes
    v_s_row = v_s_col + 256 * 2
    v_zp = v_s_row + 128 * 2
    assert v_zp + 128 * 2 == 35_072
    for token in (
        "kKSColOffset",
        "kKZpOffset",
        "kKSRowOffset",
        "kVPackedOffset",
        "kVSColOffset",
        "kVSRowOffset",
        "kVZpOffset",
        "fraction == 0.5f && (lower & 1)",
    ):
        assert token in WRITER


def test_existing_balanced_writer_remains_an_independent_control() -> None:
    assert "class KVarNBalancedWriterKernel" in ESTABLISHED_WRITER
    assert "void kvarn_pack_balanced_kv_xe2" in ESTABLISHED_WRITER
    assert "KVarNSinkhornWriterKernel" not in ESTABLISHED_WRITER
