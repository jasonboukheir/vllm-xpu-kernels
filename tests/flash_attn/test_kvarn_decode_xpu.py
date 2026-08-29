# SPDX-License-Identifier: Apache-2.0
"""Parametrized runtime tests for the narrow native Xe2 KVarN decoder.

Set ``VLLM_XPU_KERNELS_LIBRARY`` to the freshly built ``_vllm_fa2_C`` shared
object.  Keeping the library explicit prevents this suite from accidentally
testing an older installed package during local source-override iteration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmark.check_kvarn_decode import (_put_half,  # noqa: E402
                                          expected_value, make_cache,
                                          make_random_cache, reference_decode)
from benchmark.kvarn_utils import (_k_dpas_coord, _v_dpas_coord,  # noqa: E402
                                   dequant_record,
                                   swizzle_record_dpas_k4v4)


@pytest.fixture(scope="module", autouse=True)
def native_library() -> None:
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if not library:
        pytest.skip("VLLM_XPU_KERNELS_LIBRARY is not set")
    if not torch.xpu.is_available():
        pytest.skip("an XPU is not available")
    torch.ops.load_library(library)


def _tail_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.zeros((1, 128, 4, 256), dtype=torch.float16, device="xpu")
    return key, torch.zeros_like(key)


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("dpas_layout", [False, True])
def test_materialize_random_packed_cache_matches_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
    record_stride: int,
    dpas_layout: bool,
) -> None:
    canonical_cache, layout = make_random_cache(5, record_stride)
    packed_cache = canonical_cache.clone()
    if dpas_layout:
        monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
        for block in range(packed_cache.size(0)):
            for head in range(packed_cache.size(1)):
                packed_cache[block, head] = swizzle_record_dpas_k4v4(
                    canonical_cache[block, head], layout
                )
    else:
        monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)

    pages = [[4, 1], [3, 0]]
    lengths = [129, 194]
    total_tokens = sum(lengths)
    key_output = torch.empty(
        (total_tokens, 4, 256), dtype=torch.float16, device="xpu"
    )
    value_output = torch.empty_like(key_output)
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_materialize_packed_kv(
        packed_cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.tensor([0, lengths[0], total_tokens],
                     dtype=torch.int32,
                     device="xpu"),
        torch.full((5,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        key_output,
        value_output,
        max(lengths),
    )

    expected_key = []
    expected_value = []
    for request_pages, length in zip(pages, lengths, strict=True):
        records = [
            [
                dequant_record(canonical_cache[page, head], layout)
                for head in range(4)
            ]
            for page in request_pages
        ]
        expected_key.append(
            torch.stack(
                [
                    torch.cat([record[head][0] for record in records])[:length]
                    for head in range(4)
                ],
                dim=1,
            )
        )
        expected_value.append(
            torch.stack(
                [
                    torch.cat([record[head][1] for record in records])[:length]
                    for head in range(4)
                ],
                dim=1,
            )
        )

    torch.testing.assert_close(
        key_output.cpu(), torch.cat(expected_key).half(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        value_output.cpu(),
        torch.cat(expected_value).half(),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize(
    "seq_len", [1, 63, 64, 65, 127, 128, 129, 255, 256, 257]
)
def test_structured_permuted_pages(seq_len: int, record_stride: int) -> None:
    cache, _ = make_cache(3, record_stride)
    assert cache.stride(1) == record_stride
    pages = [2, 0, 1]
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        torch.full((3,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        output,
        seq_len,
        1.0 / 16.0,
    )
    expected = torch.full_like(output.cpu(), expected_value(pages, seq_len))
    torch.testing.assert_close(output.cpu(), expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("batch_size", [2, 3, 4, 12])
def test_ragged_batch_matches_independent_fp32_oracle(
    batch_size: int, record_stride: int
) -> None:
    cache, layout = make_random_cache(6, record_stride)
    assert cache.stride(1) == record_stride
    generator = torch.Generator().manual_seed(7)
    query = torch.randn((batch_size, 24, 256), generator=generator).to(
        torch.float16
    )
    pages = [[5, 1], [4, 0], [3, 2], [1, 5]] * 3
    lengths = [1, 63, 128, 129] * 3
    pages = pages[:batch_size]
    lengths = lengths[:batch_size]
    output = torch.empty_like(query, device="xpu")
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_decode(
        query.xpu(),
        cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.full((6,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        output,
        max(lengths),
        1.0 / 16.0,
    )
    references = [
        reference_decode(
            query[batch : batch + 1],
            cache,
            layout,
            [pages[batch]],
            lengths[batch],
            1.0 / 16.0,
        )
        for batch in range(batch_size)
    ]
    reference = torch.cat(references)
    torch.testing.assert_close(
        output.cpu().float(), reference, atol=2e-2, rtol=2e-2
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("record_stride", [35072, 65536])
def test_nonuniform_kvarn_factors_across_page_boundary(
    record_stride: int,
) -> None:
    """Exercise separable K and V factors with no unit-scale shortcuts.

    The three rows end immediately before, on, and after a page boundary.
    Every physical record has different, strongly nonuniform token and
    dimension factors. This detects applying a factor to the wrong fragment
    axis or accidentally carrying page-specific metadata across records.
    """
    cache, layout = make_random_cache(6, record_stride)
    assert cache.stride(1) == record_stride
    token = torch.arange(layout.group, dtype=torch.float32)
    dim = torch.arange(layout.head_dim, dtype=torch.float32)
    for block in range(6):
        for kv_head in range(4):
            phase = (block * 19 + kv_head * 7) % layout.group
            row_scale = 0.125 + ((token + phase) % layout.group) / 80.0
            dim_phase = block * 23 + kv_head * 11
            dim_scale = 0.003 + ((dim + dim_phase) % 97) / 1600.0
            dim_zp = (((dim * 37 + dim_phase) % 101) - 50) / 125.0
            v_dim_scale = 0.005 + ((dim * 13 + dim_phase) % 89) / 1800.0
            v_row_scale = 0.02 + ((token * 17 + phase) % 83) / 700.0
            v_row_zp = (((token * 29 + phase) % 79) - 39) / 160.0
            _put_half(cache[block, kv_head], layout.k_s_col_offset, dim_scale)
            _put_half(cache[block, kv_head], layout.k_zp_offset, dim_zp)
            _put_half(cache[block, kv_head], layout.k_s_row_offset, row_scale)
            _put_half(cache[block, kv_head], layout.v_s_col_offset, v_dim_scale)
            _put_half(cache[block, kv_head], layout.v_s_row_offset, v_row_scale)
            _put_half(cache[block, kv_head], layout.v_zp_offset, v_row_zp)

    generator = torch.Generator().manual_seed(314159)
    query = torch.randn((3, 24, 256), generator=generator).to(torch.float16)
    pages = [[5, 0], [4, 1], [3, 2]]
    lengths = [127, 128, 129]
    output = torch.empty_like(query, device="xpu")
    tail_key, tail_value = _tail_tensors()
    torch.ops._vllm_fa2_C.kvarn_decode(
        query.xpu(),
        cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.full((6,), -1, dtype=torch.int32, device="xpu"),
        tail_key,
        tail_value,
        output,
        max(lengths),
        1.0 / 16.0,
    )
    reference = torch.cat(
        [
            reference_decode(
                query[batch : batch + 1],
                cache,
                layout,
                [pages[batch]],
                lengths[batch],
                1.0 / 16.0,
            )
            for batch in range(3)
        ]
    )
    torch.testing.assert_close(
        output.cpu().float(), reference, atol=3e-2, rtol=3e-2
    )
    assert torch.isfinite(output).all()


def test_kvarn_dpas_fragment_coordinate_tables_are_bijections() -> None:
    """Freeze the actual CUTE B-fragment ownership used by native decode."""
    anchor = torch.empty((), device="xpu")
    k_coords, v_coords = torch.ops._vllm_fa2_C.kvarn_fragment_coords(anchor)

    assert k_coords.device.type == "cpu"
    assert v_coords.device.type == "cpu"
    assert k_coords.dtype == torch.int32
    assert v_coords.dtype == torch.int32
    assert k_coords.shape == (16, 64, 2)
    assert v_coords.shape == (16, 32, 2)

    k_pairs = {tuple(coord) for coord in k_coords.reshape(-1, 2).tolist()}
    v_pairs = {tuple(coord) for coord in v_coords.reshape(-1, 2).tolist()}
    assert k_pairs == {(token, dim) for token in range(16) for dim in range(64)}
    assert v_pairs == {(dim, token) for dim in range(32) for token in range(16)}

    # Each logical coordinate is owned exactly once, not merely present.
    assert k_coords.reshape(-1, 2).unique(dim=0).shape[0] == 16 * 64
    assert v_coords.reshape(-1, 2).unique(dim=0).shape[0] == 32 * 16

    frozen_k = torch.tensor(
        [
            [_k_dpas_coord(lane, slot) for slot in range(64)]
            for lane in range(16)
        ],
        dtype=torch.int32,
    )
    frozen_v = torch.tensor(
        [
            [_v_dpas_coord(lane, slot) for slot in range(32)]
            for lane in range(16)
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(k_coords, frozen_k, rtol=0, atol=0)
    torch.testing.assert_close(v_coords, frozen_v, rtol=0, atol=0)


def test_dpas_payload_decode_matches_canonical_ragged_and_hybrid() -> None:
    cache, layout = make_random_cache(6)
    swizzled = cache.clone()
    for block in range(6):
        for head in range(4):
            swizzled[block, head] = swizzle_record_dpas_k4v4(
                cache[block, head], layout
            )

    generator = torch.Generator().manual_seed(271828)
    query = torch.randn((3, 24, 256), generator=generator).half().xpu()
    pages = torch.tensor(
        [[5, 0, 1], [4, 1, 2], [3, 2, 0]], dtype=torch.int32, device="xpu"
    )
    lengths = torch.tensor([127, 129, 257], dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((6,), -1, dtype=torch.int32, device="xpu")
    block_to_slot[5] = 0
    tail_key = torch.randn(
        (1, 128, 4, 256), generator=generator, dtype=torch.float16
    ).xpu()
    tail_value = torch.randn(
        (1, 128, 4, 256), generator=generator, dtype=torch.float16
    ).xpu()
    canonical_output = torch.empty_like(query)
    swizzled_output = torch.empty_like(query)

    old_flag = os.environ.pop("KVARN_NATIVE_XPU_DPAS_LAYOUT", None)
    try:
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            cache.xpu(),
            pages,
            lengths,
            block_to_slot,
            tail_key,
            tail_value,
            canonical_output,
            257,
            1.0 / 16.0,
        )
        os.environ["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "1"
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            swizzled.xpu(),
            pages,
            lengths,
            block_to_slot,
            tail_key,
            tail_value,
            swizzled_output,
            257,
            1.0 / 16.0,
        )
    finally:
        if old_flag is None:
            os.environ.pop("KVARN_NATIVE_XPU_DPAS_LAYOUT", None)
        else:
            os.environ["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = old_flag
    torch.testing.assert_close(
        swizzled_output, canonical_output, rtol=0, atol=0
    )


def test_full_precision_tail_and_packed_history_share_softmax() -> None:
    cache, _ = make_cache(3)
    pages = [2, 0]
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    tail_key, tail_value = _tail_tensors()
    tail_value.fill_(0.75)
    block_to_slot = torch.full((3,), -1, dtype=torch.int32, device="xpu")
    block_to_slot[2] = 0
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([129], dtype=torch.int32, device="xpu"),
        block_to_slot,
        tail_key,
        tail_value,
        output,
        129,
        1.0 / 16.0,
    )
    expected = torch.full_like(output.cpu(), (128 * 0.75) / 129)
    torch.testing.assert_close(output.cpu(), expected, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("seq_len", [128, 129, 257])
def test_multiple_full_precision_tail_pages_share_one_softmax(
    seq_len: int,
) -> None:
    """Decode must not assume there is only one pool-resident page.

    Hybrid KVarN keeps both the sink and current partial page in fp16.  The
    first decode append therefore changes a one-tail-page request into a
    two-tail-page request, which is the exact transition exercised here.
    """
    torch.xpu.synchronize()
    cache, _ = make_cache(6)
    pages = [2, 4, 1]
    block_to_slot = torch.full((6,), -1, dtype=torch.int32, device="xpu")
    for slot, physical in enumerate(pages):
        block_to_slot[physical] = slot
    tail_key = torch.zeros((3, 128, 4, 256), dtype=torch.float16, device="xpu")
    tail_value = torch.empty_like(tail_key)
    values = (0.25, 0.75, -0.5)
    for slot, value in enumerate(values):
        tail_value[slot].fill_(value)
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        block_to_slot,
        tail_key,
        tail_value,
        output,
        seq_len,
        1.0 / 16.0,
    )
    counts = (
        min(seq_len, 128),
        min(max(seq_len - 128, 0), 128),
        max(seq_len - 256, 0),
    )
    expected_value_ = (
        sum(n * value for n, value in zip(counts, values)) / seq_len
    )
    torch.testing.assert_close(
        output.cpu(),
        torch.full_like(output.cpu(), expected_value_),
        atol=2e-3,
        rtol=2e-3,
    )


def test_two_random_tail_pages_match_fp32_attention_at_first_append() -> None:
    seq_len = 129
    cache, _ = make_cache(4)
    pages = [3, 1] + [0] * 62
    lookup = torch.full((4,), -1, dtype=torch.int32, device="xpu")
    lookup[3], lookup[1] = 0, 1
    generator = torch.Generator().manual_seed(12946613)
    tail_key_cpu = torch.randn((2, 128, 4, 256), generator=generator).half()
    tail_value_cpu = torch.randn((2, 128, 4, 256), generator=generator).half()
    query_cpu = torch.randn((1, 24, 256), generator=generator).half()
    output = torch.empty_like(query_cpu, device="xpu")
    torch.ops._vllm_fa2_C.kvarn_decode(
        query_cpu.xpu(),
        cache.xpu(),
        torch.tensor([pages], dtype=torch.int32, device="xpu"),
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        lookup,
        tail_key_cpu.xpu(),
        tail_value_cpu.xpu(),
        output,
        8192,
        1.0 / 16.0,
    )
    keys = torch.cat((tail_key_cpu[0], tail_key_cpu[1, :1]))
    values = torch.cat((tail_value_cpu[0], tail_value_cpu[1, :1]))
    kv_heads = torch.arange(24) // 6
    selected_k = keys[:, kv_heads].float().permute(1, 0, 2)
    selected_v = values[:, kv_heads].float().permute(1, 0, 2)
    logits = torch.einsum("hd,htd->ht", query_cpu[0].float(), selected_k) / 16.0
    reference = torch.einsum(
        "ht,htd->hd", torch.softmax(logits, -1), selected_v
    )
    torch.testing.assert_close(
        output.cpu()[0].float(), reference, atol=3e-3, rtol=3e-3
    )


def test_shared_prefix_is_deterministic() -> None:
    cache, _ = make_random_cache(5)
    generator = torch.Generator().manual_seed(19)
    query_row = torch.randn((1, 24, 256), generator=generator).to(torch.float16)
    query = query_row.expand(4, -1, -1).contiguous().xpu()
    # All rows alias the same physical prefix. Their second pages differ, but
    # seq_len=128 makes those entries padding and therefore deliberately
    # invalid inputs to the device loader.
    pages = [[3, 0], [3, 1], [3, 2], [3, 4]]
    seq_lens = torch.full((4,), 128, dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((5,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    output = torch.empty_like(query)
    repeat = torch.empty_like(query)
    arguments = (
        query,
        cache.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )
    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, output, 128, 1.0 / 16.0)
    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, repeat, 128, 1.0 / 16.0)
    torch.testing.assert_close(output, repeat, atol=0, rtol=0)
    torch.testing.assert_close(
        output, output[0:1].expand_as(output), atol=0, rtol=0
    )


def test_decode_with_scratch_matches_legacy_and_reuses_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.xpu.synchronize()
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "16")
    cache, _ = make_random_cache(6)
    generator = torch.Generator().manual_seed(20260808)
    query = torch.randn((4, 24, 256), generator=generator).half().xpu()
    pages = torch.tensor(
        [[5, 0], [4, 1], [3, 2], [1, 5]], dtype=torch.int32, device="xpu"
    )
    seq_lens = torch.tensor([1, 127, 128, 129], dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((6,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )
    legacy = torch.empty_like(query)
    actual = torch.empty_like(query)
    temp_output = torch.empty(
        (4, 16 * 24, 256), dtype=torch.float16, device="xpu"
    )
    exp_sums = torch.empty((4, 24, 16), dtype=torch.float32, device="xpu")
    max_logits = torch.empty_like(exp_sums)
    pointers = tuple(
        tensor.data_ptr() for tensor in (temp_output, exp_sums, max_logits)
    )

    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, legacy, 129, 1.0 / 16.0)
    for _ in range(3):
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            *arguments,
            temp_output,
            exp_sums,
            max_logits,
            actual,
            129,
            1.0 / 16.0,
        )
        assert pointers == tuple(
            tensor.data_ptr() for tensor in (temp_output, exp_sums, max_logits)
        )
        torch.testing.assert_close(actual, legacy, atol=0, rtol=0)


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("splits", [16, 32])
def test_split_local_scratch_survives_allocator_pressure(
    monkeypatch: pytest.MonkeyPatch, record_stride: int, splits: int
) -> None:
    """The asynchronous reducer must retain legacy wrapper scratch storage."""
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    cache, _ = make_random_cache(188, record_stride)
    generator = torch.Generator().manual_seed(16000 + record_stride)
    query = torch.randn((4, 24, 256), generator=generator).half().xpu()
    pages = torch.arange(188, dtype=torch.int32, device="xpu").reshape(4, 47)
    seq_lens = torch.tensor(
        [6000, 5889, 5761, 5633], dtype=torch.int32, device="xpu"
    )
    block_to_slot = torch.full((188,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    expected = torch.empty_like(query)
    actual = torch.empty_like(query)
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )

    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, expected, 6000, 1.0 / 16.0)
    torch.xpu.synchronize()
    for _ in range(16):
        torch.ops._vllm_fa2_C.kvarn_decode(*arguments, actual, 6000, 1.0 / 16.0)
        # Match all three local scratch allocation classes before consuming
        # the result, maximizing the chance of premature allocator reuse.
        torch.empty(
            (4, 24 * splits, 256), dtype=torch.float16, device="xpu"
        ).fill_(float("nan"))
        torch.empty((4, 24, splits), dtype=torch.float32, device="xpu").fill_(
            float("nan")
        )
        torch.empty((4, 24, splits), dtype=torch.float32, device="xpu").fill_(
            float("nan")
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("record_stride", [35072, 65536])
@pytest.mark.parametrize("seq_len", [128, 129, 6000])
@pytest.mark.parametrize("splits", [16, 32])
def test_split_matches_split1(
    monkeypatch: pytest.MonkeyPatch,
    record_stride: int,
    seq_len: int,
    splits: int,
) -> None:
    pages_per_row = (seq_len + 127) // 128
    blocks = 4 * pages_per_row
    cache, _ = make_random_cache(blocks, record_stride)
    generator = torch.Generator().manual_seed(seq_len + record_stride)
    query = torch.randn((4, 24, 256), generator=generator).half().xpu()
    pages = torch.arange(blocks, dtype=torch.int32, device="xpu").reshape(
        4, pages_per_row
    )
    seq_lens = torch.full((4,), seq_len, dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((blocks,), -1, dtype=torch.int32, device="xpu")
    tail_key, tail_value = _tail_tensors()
    arguments = (
        query,
        cache.xpu(),
        pages,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
    )
    split1 = torch.empty_like(query)
    split_output = torch.empty_like(query)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "1")
    torch.ops._vllm_fa2_C.kvarn_decode(*arguments, split1, seq_len, 1.0 / 16.0)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", str(splits))
    torch.ops._vllm_fa2_C.kvarn_decode(
        *arguments, split_output, seq_len, 1.0 / 16.0
    )
    torch.testing.assert_close(split_output, split1, atol=2e-2, rtol=2e-2)
