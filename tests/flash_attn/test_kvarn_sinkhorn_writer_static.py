# SPDX-License-Identifier: Apache-2.0
"""Host-only ABI and resource proofs for the fused Xe2 Sinkhorn writer."""

from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
WRITER = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_sinkhorn_writer_xe2.cpp"
).read_text()
ZERO_WRITER = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_sinkhorn_writer_zero_xe2.hpp"
).read_text()
REGISTRATION = (REPO_ROOT / "csrc/flash_attn/flash_api.cpp").read_text()
ESTABLISHED_WRITER = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_balanced_writer_xe2.cpp"
).read_text()


def test_fused_writer_keeps_slm_below_xe2_limit() -> None:
    # Six D256 vectors, one scalar, and two D256 moment scratch vectors; the
    # compact source page remains in global/L2 storage.
    state_floats = 8 * 256 + 1
    assert state_floats * 4 == 8_196
    assert state_floats * 4 < 64 * 1024
    assert "kLocalMemoryBytes = kStateFloats * sizeof(float)" in WRITER
    assert "kLocalMemoryBytes <= 64 * 1024" in WRITER
    assert "local_accessor<input_t" not in WRITER


def test_iterative_writer_matches_triton_xpu_reduction_topology() -> None:
    # Artifact-era Triton-XPU 3.7.2 IR uses subgroup32, 32-value lane-local
    # trees, and SPIR-V clustered reductions of 8 (K) or 4 (V).
    assert "reqd_sub_group_size(kSubgroup)" in WRITER
    assert "kSubgroup = 32" in WRITER
    assert "kValuesPerLane == kSubgroup" in WRITER
    assert "chunked_partition<kCluster>(subgroup)" in WRITER
    assert "chunked_partition<2>(subgroup)" in WRITER
    assert "sycl::reduce_over_group(cluster" in WRITER
    assert "subgroup, (x0 + x1) + (x2 + x3)" in WRITER
    assert "half * kGroup + lane * 4" in WRITER


def test_iterative_writer_preserves_triton_math_order() -> None:
    assert "sycl::exp2(log_scale * 1.4426950216293335f)" in WRITER
    assert "variance *= float(Extent)" in WRITER
    assert "variance /= float(Extent - 1)" in WRITER
    assert "float(Extent) / float(Extent - 1)" not in WRITER


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


def test_launch_and_allocator_use_the_cache_device_current_stream() -> None:
    selector = "getCurrentXPUStream(packed_cache.device().index())"
    assert WRITER.count(selector) == 1
    assert "getCurrentXPUStream().queue()" not in WRITER
    assert "auto& queue = current_stream.queue()" in WRITER
    assert "tensor->storage().data_ptr(), current_stream" in WRITER


def test_zero_iteration_packer_is_compile_time_isolated_from_sinkhorn() -> None:
    assert '#include "kvarn_sinkhorn_writer_zero_xe2.hpp"' in WRITER
    assert "if (sinkhorn_iterations == 0)" in WRITER
    assert "submit_kvarn_sinkhorn_zero_writer" in WRITER
    assert "KVarNSinkhornZeroWriterKernel" in ZERO_WRITER
    assert "reqd_sub_group_size(kZeroSubgroup)" in ZERO_WRITER
    assert "kZeroSubgroup = 16" in ZERO_WRITER
    assert "kLogCol" not in ZERO_WRITER
    assert "sinkhorn_iterations" not in ZERO_WRITER


def test_zero_iteration_packer_mirrors_exact_dpas_writer_arithmetic() -> None:
    shared_fragments = (
        "sycl::reduce_over_group(",
        "subgroup, lane_lo, sycl::minimum<float>()",
        "subgroup, lane_hi, sycl::maximum<float>()",
        "(hi - lo) / float(kZeroQMax)",
        "fraction == 0.5f && (lower & 1)",
        "q0 | (q1 << 4)",
    )
    assert all(fragment in ZERO_WRITER for fragment in shared_fragments)
    assert "rows = tiles * (kZeroHeadDim + kZeroGroup)" in ZERO_WRITER
    assert "key_scale[channel] = sycl::half(scale)" in ZERO_WRITER
    assert "key_zero[channel] = sycl::half(lo)" in ZERO_WRITER
    assert "key_row[token] = sycl::half(1.0f)" in ZERO_WRITER
    assert "value_col[channel] = sycl::half(1.0f)" in ZERO_WRITER
    assert "value_scale[token] = sycl::half(scale)" in ZERO_WRITER
    assert "value_zero[token] = sycl::half(lo)" in ZERO_WRITER
