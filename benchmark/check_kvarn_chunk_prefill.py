# SPDX-License-Identifier: Apache-2.0
"""Deterministic oracle for native compact cached/chunked prefill."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmark.check_kvarn_decode import make_cache, make_random_cache
from benchmark.kvarn_utils import (
    KVarNLayout,
    dequant_record,
    swizzle_record_dpas_k4v4,
)


def maybe_swizzle(cache: torch.Tensor, layout: KVarNLayout, dpas: bool) -> torch.Tensor:
    if not dpas:
        return cache
    result = cache.clone()
    for block in range(result.shape[0]):
        for head in range(result.shape[1]):
            result[block, head] = swizzle_record_dpas_k4v4(
                result[block, head], layout
            )
    return result


def expected_rows(pages: list[int], total_len: int, query_len: int) -> torch.Tensor:
    cached = total_len - query_len
    rows = []
    for query_row in range(query_len):
        visible = cached + query_row + 1
        values = [
            pages[token // 128] * 0.25 + (token % 128) / 1024
            for token in range(visible)
        ]
        rows.append(sum(values) / visible)
    return torch.tensor(rows, dtype=torch.float16)[:, None, None].expand(
        query_len, 24, 256
    )


def random_reference(
    query: torch.Tensor,
    cache: torch.Tensor,
    layout: KVarNLayout,
    pages: list[int],
    total_len: int,
    scale: float,
) -> torch.Tensor:
    result = torch.empty_like(query, dtype=torch.float32)
    query_len = query.shape[0]
    cached = total_len - query_len
    for kv_head in range(4):
        keys, values = zip(
            *(dequant_record(cache[page, kv_head], layout) for page in pages),
            strict=True,
        )
        key = torch.cat(keys)[:total_len]
        value = torch.cat(values)[:total_len]
        for local_head in range(6):
            head = kv_head * 6 + local_head
            scores = query[:, head].float() @ key.T * scale
            causal = torch.arange(total_len)[None, :] > (
                cached + torch.arange(query_len)[:, None]
            )
            scores.masked_fill_(causal, float("-inf"))
            result[:, head] = torch.softmax(scores, dim=-1) @ value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--verify-benchmark", action="store_true")
    parser.add_argument("--dpas", action="store_true")
    args = parser.parse_args()
    os.environ["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "1" if args.dpas else "0"
    torch.ops.load_library(args.library)

    # Deliberately cross 64- and 128-token boundaries and use unequal query
    # chunks so varlen offsets, GQA head mapping, and bottom-right masking are
    # all independently observable.
    totals = [193, 321]
    query_lengths = [65, 129]
    pages = [[2, 0, 3], [1, 4, 0]]
    cache, _ = make_cache(5, 35072)
    block_table = torch.tensor(pages, dtype=torch.int32, device="xpu")
    seq_lens = torch.tensor(totals, dtype=torch.int32, device="xpu")
    cu_q = torch.tensor([0, 65, 194], dtype=torch.int32, device="xpu")
    query = torch.zeros((194, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    block_to_slot = torch.full((5,), -1, dtype=torch.int32, device="xpu")
    tail_key = torch.zeros((1, 128, 4, 256), dtype=torch.float16, device="xpu")
    tail_value = torch.zeros_like(tail_key)

    torch.ops._vllm_fa2_C.kvarn_chunk_prefill(
        query,
        cache.xpu(),
        block_table,
        seq_lens,
        cu_q,
        block_to_slot,
        tail_key,
        tail_value,
        output,
        max(query_lengths),
        max(totals),
        1.0 / 16.0,
    )
    torch.xpu.synchronize()
    expected = torch.cat(
        [
            expected_rows(request_pages, total, query_len)
            for request_pages, total, query_len in zip(
                pages, totals, query_lengths, strict=True
            )
        ]
    )
    actual = output.cpu()
    errors = (actual.float() - expected.float()).abs()
    print(
        f"compact chunk prefill: max_error={errors.max().item():.6g} "
        f"mean_error={errors.mean().item():.6g}"
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    if not torch.isfinite(actual).all():
        raise AssertionError("compact chunk prefill produced non-finite output")

    random_cache, layout = make_random_cache(5, 35072)
    packed_random_cache = maybe_swizzle(random_cache, layout, args.dpas)
    cu_k = torch.tensor([0, totals[0], sum(totals)], dtype=torch.int32, device="xpu")
    materialized_k = torch.empty(
        (sum(totals), 4, 256), dtype=torch.float16, device="xpu"
    )
    materialized_v = torch.empty_like(materialized_k)
    torch.ops._vllm_fa2_C.kvarn_materialize_packed_kv(
        packed_random_cache.xpu(),
        block_table,
        seq_lens,
        cu_k,
        block_to_slot,
        tail_key,
        tail_value,
        materialized_k,
        materialized_v,
        max(totals),
    )
    torch.xpu.synchronize()
    expected_k = []
    expected_v = []
    for request_pages, total in zip(pages, totals, strict=True):
        request_records = [
            [dequant_record(random_cache[page, head], layout) for head in range(4)]
            for page in request_pages
        ]
        expected_k.append(
            torch.stack(
                [torch.cat([record[h][0] for record in request_records])[:total]
                 for h in range(4)],
                dim=1,
            )
        )
        expected_v.append(
            torch.stack(
                [torch.cat([record[h][1] for record in request_records])[:total]
                 for h in range(4)],
                dim=1,
            )
        )
    torch.testing.assert_close(
        materialized_k.cpu(), torch.cat(expected_k).half(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        materialized_v.cpu(), torch.cat(expected_v).half(), atol=2e-3, rtol=2e-3
    )
    print("native compact materializer: exact random oracle passed")
    generator = torch.Generator().manual_seed(20260812)
    random_query = torch.randn((194, 24, 256), generator=generator).half()
    random_output = torch.empty_like(random_query, device="xpu")
    scale = 1.0 / 16.0
    torch.ops._vllm_fa2_C.kvarn_chunk_prefill(
        random_query.xpu(),
        packed_random_cache.xpu(),
        block_table,
        seq_lens,
        cu_q,
        block_to_slot,
        tail_key,
        tail_value,
        random_output,
        max(query_lengths),
        max(totals),
        scale,
    )
    torch.xpu.synchronize()
    reference = torch.cat(
        [
            random_reference(
                random_query[start:stop],
                random_cache,
                layout,
                request_pages,
                total,
                scale,
            )
            for start, stop, request_pages, total in (
                (0, 65, pages[0], totals[0]),
                (65, 194, pages[1], totals[1]),
            )
        ]
    )
    random_actual = random_output.cpu().float()
    errors = (random_actual - reference).abs()
    print(
        f"random compact chunk prefill: max_error={errors.max().item():.6g} "
        f"mean_error={errors.mean().item():.6g}"
    )
    if errors.max().item() > 2e-2:
        per_head = errors.amax(dim=(0, 2))
        print(
            "random per-head max_error: "
            + ", ".join(f"h{h}={value:.6g}" for h, value in enumerate(per_head))
        )
    torch.testing.assert_close(
        random_actual, reference, atol=2e-2, rtol=2e-2
    )

    if args.verify_benchmark:
        batch = 4
        qlen = 3
        total_len = 6000
        pages_per_request = (total_len + 127) // 128
        num_pages = batch * pages_per_request
        verify_cache, verify_layout = make_random_cache(num_pages, 35072)
        verify_cache = maybe_swizzle(verify_cache, verify_layout, args.dpas).xpu()
        request_table = torch.arange(
            num_pages, dtype=torch.int32, device="xpu"
        ).view(batch, pages_per_request)
        virtual_table = request_table.repeat_interleave(qlen, dim=0).contiguous()
        request_seq_lens = torch.full(
            (batch,), total_len, dtype=torch.int32, device="xpu"
        )
        virtual_seq_lens = torch.tensor(
            [
                total_len - qlen + offset + 1
                for _ in range(batch)
                for offset in range(qlen)
            ],
            dtype=torch.int32,
            device="xpu",
        )
        verify_cu_q = torch.arange(
            0, batch * qlen + 1, qlen, dtype=torch.int32, device="xpu"
        )
        verify_map = torch.full(
            (num_pages,), -1, dtype=torch.int32, device="xpu"
        )
        verify_query = torch.randn(
            (batch * qlen, 24, 256), dtype=torch.float16, device="xpu"
        )
        chunk_output = torch.empty_like(verify_query)
        virtual_output = torch.empty_like(verify_query)

        def run_chunk_verify():
            torch.ops._vllm_fa2_C.kvarn_chunk_prefill(
                verify_query,
                verify_cache,
                request_table,
                request_seq_lens,
                verify_cu_q,
                verify_map,
                tail_key,
                tail_value,
                chunk_output,
                qlen,
                total_len,
                1.0 / 16.0,
            )

        def run_virtual_verify():
            torch.ops._vllm_fa2_C.kvarn_decode(
                verify_query,
                verify_cache,
                virtual_table,
                virtual_seq_lens,
                verify_map,
                tail_key,
                tail_value,
                virtual_output,
                total_len,
                1.0 / 16.0,
            )

        run_chunk_verify()
        run_virtual_verify()
        torch.xpu.synchronize()
        errors = (chunk_output.float() - virtual_output.float()).abs()
        print(
            f"qlen3 chunk-vs-virtual: max_error={errors.max().item():.6g} "
            f"mean_error={errors.mean().item():.6g}"
        )
        torch.testing.assert_close(
            chunk_output, virtual_output, atol=3e-2, rtol=3e-2
        )
        for callable_ in (run_chunk_verify, run_virtual_verify):
            for _ in range(5):
                callable_()
        torch.xpu.synchronize()
        for name, callable_ in (
            ("chunk-shared-prefix", run_chunk_verify),
            ("virtual-independent-rows", run_virtual_verify),
        ):
            samples = []
            for _ in range(20):
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                callable_()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            samples.sort()
            print(
                f"qlen3 {name}: median_us={statistics.median(samples):.3f} "
                f"p95_us={samples[18]:.3f}"
            )

    if args.benchmark:
        max_total = 32768
        num_pages = max_total // 128
        timing_cache, timing_layout = make_cache(num_pages, 35072)
        timing_cache = maybe_swizzle(timing_cache, timing_layout, args.dpas)
        timing_cache_xpu = timing_cache.xpu()
        timing_table = torch.arange(
            num_pages, dtype=torch.int32, device="xpu"
        )[None, :]
        timing_map = torch.full(
            (num_pages,), -1, dtype=torch.int32, device="xpu"
        )
        timing_query = torch.zeros(
            (4096, 24, 256), dtype=torch.float16, device="xpu"
        )
        timing_output = torch.empty_like(timing_query)
        timing_cu_q = torch.tensor([0, 4096], dtype=torch.int32, device="xpu")
        for total_len in (8192, 32768):
            timing_seq = torch.tensor(
                [total_len], dtype=torch.int32, device="xpu"
            )
            timing_cu_k = torch.tensor(
                [0, total_len], dtype=torch.int32, device="xpu"
            )
            timing_k = torch.empty(
                (total_len, 4, 256), dtype=torch.float16, device="xpu"
            )
            timing_v = torch.empty_like(timing_k)
            for _ in range(3):
                torch.ops._vllm_fa2_C.kvarn_materialize_packed_kv(
                    timing_cache_xpu,
                    timing_table,
                    timing_seq,
                    timing_cu_k,
                    timing_map,
                    tail_key,
                    tail_value,
                    timing_k,
                    timing_v,
                    total_len,
                )
            torch.xpu.synchronize()
            materialize_samples = []
            for _ in range(5):
                start = time.perf_counter()
                torch.ops._vllm_fa2_C.kvarn_materialize_packed_kv(
                    timing_cache_xpu,
                    timing_table,
                    timing_seq,
                    timing_cu_k,
                    timing_map,
                    tail_key,
                    tail_value,
                    timing_k,
                    timing_v,
                    total_len,
                )
                torch.xpu.synchronize()
                materialize_samples.append((time.perf_counter() - start) * 1000)
            materialize_samples.sort()
            print(
                f"native materialize kv={total_len}: "
                f"median_ms={materialize_samples[len(materialize_samples) // 2]:.3f} "
                f"samples={[round(value, 3) for value in materialize_samples]}"
            )
            for _ in range(3):
                torch.ops._vllm_fa2_C.kvarn_chunk_prefill(
                    timing_query,
                    timing_cache_xpu,
                    timing_table,
                    timing_seq,
                    timing_cu_q,
                    timing_map,
                    tail_key,
                    tail_value,
                    timing_output,
                    4096,
                    total_len,
                    1.0 / 16.0,
                )
            torch.xpu.synchronize()
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                torch.ops._vllm_fa2_C.kvarn_chunk_prefill(
                    timing_query,
                    timing_cache_xpu,
                    timing_table,
                    timing_seq,
                    timing_cu_q,
                    timing_map,
                    tail_key,
                    tail_value,
                    timing_output,
                    4096,
                    total_len,
                    1.0 / 16.0,
                )
                torch.xpu.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            samples.sort()
            print(
                f"native compact q=4096 kv={total_len}: "
                f"median_ms={samples[len(samples) // 2]:.3f} "
                f"samples={[round(value, 3) for value in samples]}"
            )


if __name__ == "__main__":
    main()
