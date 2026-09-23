# SPDX-License-Identifier: Apache-2.0
"""K4V2 native writer/readers against independent CPU format and attention."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from benchmark.kvarn_utils import (KVarNLayout, dequant_record,  # noqa: E402
                                   swizzle_record_dpas_kv)
from tests.flash_attn.test_kvarn_hadamard_xpu import _hadamard_256


@pytest.fixture(scope="module", autouse=True)
def native_library():
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if not library or not torch.xpu.is_available():
        pytest.skip("explicit candidate library and XPU required")
    torch.ops.load_library(library)


def _put(record, offset, value):
    raw = value.half().contiguous().view(torch.uint8).flatten()
    record[offset : offset + raw.numel()] = raw


def _pack(q, bits):
    q = q.to(torch.uint8)
    n = 8 // bits
    packed = torch.zeros_like(q[..., ::n])
    for field in range(n):
        packed |= q[..., field::n] << (field * bits)
    return packed.flatten()


def _rtn(x, bits):
    low = x.amin(-1)
    scale = ((x.amax(-1) - low) / ((1 << bits) - 1)).clamp_min(1e-10)
    codes = ((x - low[..., None]) / scale[..., None]).round()
    return codes.clamp(0, (1 << bits) - 1).to(torch.uint8), scale, low


def _fixture(bits, stride, pages=5):
    """CPU-only input generation; no production writer or reader is called."""
    layout = KVarNLayout(value_bits=bits, record_stride=stride)
    gen = torch.Generator().manual_seed(429120 + bits)
    n = pages * 4
    k = torch.randn(n, 256, 128, generator=gen) / 2
    v = torch.randn(n, 128, 256, generator=gen) / 2
    # Constant rows and exact half-way cases exercise degenerate ranges and
    # ties-to-even, including two token rows sharing a two-bit packed byte.
    v[:, 0] = 0.125
    pattern = torch.tensor(
        [0, 0.5, 1.5, 2.5, (1 << bits) - 1], dtype=torch.float32
    )
    v[:, 2] = pattern[torch.arange(256) % pattern.numel()]
    k[:, 0] = -0.25
    kc = torch.rand(n, 128, generator=gen) / 2 + 0.5
    kr = torch.rand(n, 256, generator=gen) / 2 + 0.5
    vc = torch.rand(n, 256, generator=gen) / 2 + 0.5
    vr = torch.rand(n, 128, generator=gen) / 2 + 0.5
    qk, sk, zk = _rtn(k, 4)
    qv, sv, zv = _rtn(v, bits)
    natural = torch.full((pages, 4, stride), 0xA5, dtype=torch.uint8)
    packed = natural.clone()
    for i in range(n):
        record = natural.view(n, stride)[i]
        record[:16384] = _pack(qk[i], 4)
        record[layout.v_packed_offset : layout.v_s_col_offset] = _pack(
            qv[i], bits
        )
        for offset, values in (
            (layout.k_s_col_offset, kr[i] * sk[i]),
            (layout.k_zp_offset, kr[i] * zk[i]),
            (layout.k_s_row_offset, kc[i]),
            (layout.v_s_col_offset, vc[i]),
            (layout.v_s_row_offset, vr[i] * sv[i]),
            (layout.v_zp_offset, vr[i] * zv[i]),
        ):
            _put(record, offset, values)
        packed.view(n, stride)[i] = swizzle_record_dpas_kv(record, layout)
    return layout, (k, kc, kr, v, vc, vr), natural, packed


@pytest.mark.parametrize("bits,stride", [(2, 26880), (2, 32768), (4, 35072)])
def test_balanced_writer_matches_independent_bytes_and_preserves_other_pages(
    bits, stride
):
    layout, balanced, _, expected = _fixture(bits, stride, pages=2)
    cache = torch.full((4, 4, stride), 0xA5, dtype=torch.uint8, device="xpu")
    gpu_inputs = tuple(t.xpu() for t in balanced)
    blocks = torch.tensor([3, 1], dtype=torch.int64, device="xpu")
    for _ in range(3):
        torch.ops._vllm_fa2_C.kvarn_pack_balanced_kv(
            *gpu_inputs, blocks, cache, True, bits
        )
        actual = cache.cpu()
        assert torch.equal(actual[0], torch.full_like(actual[0], 0xA5))
        assert torch.equal(actual[2], torch.full_like(actual[2], 0xA5))
        torch.testing.assert_close(
            actual[[3, 1], :, : layout.tile_bytes],
            expected[:, :, : layout.tile_bytes],
            rtol=0,
            atol=0,
        )
        # Padding initialization is part of the existing writer's contract.
        if stride > layout.tile_bytes:
            assert not actual[[3, 1], :, layout.tile_bytes :].any()


@pytest.mark.parametrize("bits,stride", [(2, 26880), (2, 32768), (4, 35072)])
@pytest.mark.parametrize("dpas", [False, True])
def test_dequant_and_materialize_match_independent_cpu_records(
    bits, stride, dpas
):
    layout, _, natural, packed = _fixture(bits, stride)
    cache = (packed if dpas else natural).xpu()
    key = torch.empty(5, 4, 256, 128, dtype=torch.float16, device="xpu")
    value = torch.empty(5, 4, 128, 256, dtype=torch.float16, device="xpu")
    torch.ops._vllm_fa2_C.kvarn_dequant(cache, key, value, dpas, bits)
    expected_k, expected_v = [], []
    for page in range(5):
        records = [dequant_record(natural[page, h], layout) for h in range(4)]
        expected_k.append(torch.stack([r[0].T for r in records]))
        expected_v.append(torch.stack([r[1] for r in records]))
    expected_k = torch.stack(expected_k).half()
    expected_v = torch.stack(expected_v).half()
    torch.testing.assert_close(key.cpu(), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(value.cpu(), expected_v, rtol=0, atol=0)
    pages, lengths = [[4, 1], [0, 3]], [129, 255]
    out_k = torch.empty(sum(lengths), 4, 256, dtype=torch.float16, device="xpu")
    out_v = torch.empty_like(out_k)
    tail = torch.zeros(1, 128, 4, 256, dtype=torch.float16, device="xpu")
    torch.ops._vllm_fa2_C.kvarn_materialize_packed_kv(
        cache,
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.tensor([0, 129, 384], dtype=torch.int32, device="xpu"),
        torch.full((5,), -1, dtype=torch.int32, device="xpu"),
        tail,
        tail,
        out_k,
        out_v,
        255,
        dpas,
        bits,
    )
    ek = torch.cat(
        [
            torch.cat([expected_k[p].permute(2, 0, 1) for p in row])[:n]
            for row, n in zip(pages, lengths)
        ]
    )
    ev = torch.cat(
        [
            torch.cat([expected_v[p].permute(1, 0, 2) for p in row])[:n]
            for row, n in zip(pages, lengths)
        ]
    )
    torch.testing.assert_close(out_k.cpu(), ek, rtol=0, atol=0)
    torch.testing.assert_close(out_v.cpu(), ev, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("batch,splits", [(1, 1), (4, 1), (4, 4), (12, 4)])
def test_ragged_decode_and_fp16_tail_match_independent_attention(
    bits, batch, splits
):
    layout, _, natural, packed = _fixture(bits, 18688 + 4096 * bits)
    gen = torch.Generator().manual_seed(98001)
    query = torch.randn(batch, 24, 256, generator=gen).half()
    tail_k = torch.randn(1, 128, 4, 256, generator=gen).half() / 2
    tail_v = torch.randn(1, 128, 4, 256, generator=gen).half() / 2
    pages = [[4, 1, 3], [0, 4, 3], [1, 0, 3], [4, 0, 3]] * 3
    lengths = ([129, 255, 256, 257] * 3)[:batch]
    pages = pages[:batch]
    lookup = torch.tensor([-1, -1, -1, 0, -1], dtype=torch.int32, device="xpu")
    actual = torch.empty_like(query, device="xpu")
    args = (
        query.xpu(),
        packed.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        lookup,
        tail_k.xpu(),
        tail_v.xpu(),
    )
    scratch = (
        torch.empty(batch, 24 * splits, 256, dtype=torch.float16, device="xpu"),
        torch.full((batch, 24, splits), float("nan"), device="xpu"),
        torch.full((batch, 24, splits), float("nan"), device="xpu"),
    )
    torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
        *args,
        *scratch,
        actual,
        max(lengths),
        1 / 16,
        False,
        False,
        splits,
        18,
        True,
        bits,
    )
    expected = torch.empty_like(query)
    records = [
        [dequant_record(natural[p, h], layout) for h in range(4)]
        for p in range(5)
    ]
    for b in range(batch):
        for h in range(24):
            hk = h // 6
            k = torch.cat(
                [
                    tail_k[0, :, hk] if p == 3 else records[p][hk][0].half()
                    for p in pages[b]
                ]
            )[: lengths[b]].float()
            v = torch.cat(
                [
                    tail_v[0, :, hk] if p == 3 else records[p][hk][1].half()
                    for p in pages[b]
                ]
            )[: lengths[b]].float()
            expected[b, h] = (
                (query[b, h].float() @ k.T / 16).softmax(-1) @ v
            ).half()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.006, atol=0.002)
    legacy = torch.empty_like(actual)
    torch.ops._vllm_fa2_C.kvarn_decode(
        *args,
        legacy,
        max(lengths),
        1 / 16,
        False,
        False,
        splits,
        18,
        True,
        bits,
    )
    torch.testing.assert_close(legacy, actual, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize(
    "splits,unrotate,dtype",
    [(1, False, torch.float16), (24, False, torch.float16),
     (32, False, torch.float16), (16, True, torch.float16),
     (16, True, torch.bfloat16)],
)
def test_resident_pages_and_offset_tail_views_match_independent_attention(
    bits, splits, unrotate, dtype
):
    """Cover both page halves, head/dimension tiles and scalar fallbacks."""
    layout, _, natural, packed = _fixture(bits, 18688 + 4096 * bits)
    gen = torch.Generator().manual_seed(98002)
    query = (torch.randn(4, 24, 256, generator=gen) / 2).half()
    tail_k = (torch.randn(3, 128, 4, 256, generator=gen) / 2).half()
    tail_v = (torch.randn(3, 128, 4, 256, generator=gen) / 2).half()
    pages = [[(3 * p + b) % 5 for p in range(33)] for b in range(4)]
    lengths = [127, 257, 4001, 4223]
    # Deliberately permute physical blocks and resident slot ownership.
    slots = [2, -1, 0, 1, -1]
    device_args = (
        query.xpu(),
        packed.xpu(),
        torch.tensor(pages, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.tensor(slots, dtype=torch.int32, device="xpu"),
    )
    records = [
        [dequant_record(natural[p, h], layout) for h in range(4)]
        for p in range(5)
    ]
    expected = torch.empty_like(query)
    for b in range(4):
        for h in range(24):
            hk = h // 6
            key = torch.cat(
                [
                    tail_k[slots[p], :, hk]
                    if slots[p] >= 0
                    else records[p][hk][0].half()
                    for p in pages[b]
                ]
            )[: lengths[b]].float()
            value = torch.cat(
                [
                    tail_v[slots[p], :, hk]
                    if slots[p] >= 0
                    else records[p][hk][1].half()
                    for p in pages[b]
                ]
            )[: lengths[b]].float()
            expected[b, h] = (
                (query[b, h].float() @ key.T / 16).softmax(-1) @ value
            ).half()

    if unrotate:
        expected = (expected.float() @ _hadamard_256()).half().to(dtype)

    reference = None
    # Offset32 is still64-byte aligned; offset1 forces scalar loading.
    # Mixed pairs prove K and V choose their paths independently.
    for key_offset, value_offset in [(0, 0), (32, 32), (1, 0), (0, 1), (1, 1)]:
        buffers = []
        views = []
        for cpu, offset in [(tail_k, key_offset), (tail_v, value_offset)]:
            backing = torch.full(
                (cpu.numel() + 64,), 23.0, dtype=torch.float16, device="xpu"
            )
            view = backing[offset : offset + cpu.numel()].view_as(cpu)
            assert view.is_contiguous()
            assert view.data_ptr() % 64 == (offset * 2) % 64
            view.copy_(cpu)
            buffers.append((backing, backing.clone()))
            views.append(view)
        scratch = (
            torch.full(
                (4, 24 * splits, 256),
                float("nan"),
                dtype=torch.float16,
                device="xpu",
            ),
            torch.full((4, 24, splits), float("nan"), device="xpu"),
            torch.full((4, 24, splits), float("nan"), device="xpu"),
        )
        actual = torch.full_like(query, float("nan"), device="xpu", dtype=dtype)
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            *device_args,
            *views,
            *scratch,
            actual,
            max(lengths),
            1 / 16,
            unrotate,
            dtype == torch.bfloat16,
            splits,
            18,
            True,
            bits,
        )
        output = actual.cpu()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=0.006, atol=0.002)
        if reference is None:
            reference = output
        else:
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        for backing, before in buffers:
            torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bits,stride", [(3, 35072), (2, 26876), (2, 26882), (4, 26880)]
)
def test_mismatched_format_is_rejected_before_launch(bits, stride):
    cache = torch.empty(1, 4, stride, dtype=torch.uint8, device="xpu")
    key = torch.empty(1, 4, 256, 128, dtype=torch.float16, device="xpu")
    value = torch.empty(1, 4, 128, 256, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError):
        torch.ops._vllm_fa2_C.kvarn_dequant(cache, key, value, True, bits)
