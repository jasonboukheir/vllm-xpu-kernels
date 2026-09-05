# SPDX-License-Identifier: Apache-2.0
"""Host-only registration and stream-lifetime proofs for QKV scatter."""

from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SOURCE = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_hadamard_scatter_xe2.cpp"
).read_text()
REGISTRATION = (REPO_ROOT / "csrc/flash_attn/flash_api.cpp").read_text()


def _function_body(name: str) -> str:
    start = SOURCE.index(f"void {name}(")
    next_function = SOURCE.find("\nvoid ", start + 1)
    return (
        SOURCE[start:] if next_function == -1 else SOURCE[start:next_function]
    )


def test_current_stream_qkv_scatter_has_a_distinct_schema() -> None:
    assert (
        '"kvarn_hadamard_qkv_scatter_current_stream(Tensor query, Tensor key, "'
        in REGISTRATION
    )
    assert "&kvarn_hadamard_qkv_scatter_current_stream_xe2" in REGISTRATION


def test_qkv_scatter_ops_keep_separate_lifetime_policies() -> None:
    conservative = _function_body("kvarn_hadamard_qkv_scatter_xe2")
    current_stream = _function_body(
        "kvarn_hadamard_qkv_scatter_current_stream_xe2"
    )
    assert "dpas_layout,\n      true);" in conservative
    assert "dpas_layout,\n      false);" in current_stream
    assert "if (record_streams)" in SOURCE
    assert "XPUCachingAllocator::recordStream" in SOURCE


def test_current_stream_variant_reuses_the_identical_kernel_submission() -> (
    None
):
    assert SOURCE.count("KVarNHadamardQKVScatterKernel<input_t>") == 1
    assert SOURCE.count("kvarn_hadamard_qkv_scatter_impl_xe2(") == 3
