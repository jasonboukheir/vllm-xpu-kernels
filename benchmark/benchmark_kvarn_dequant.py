"""Correctness and XPU-event microbenchmark for native KVarN dequantization."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).parents[1]
sys.path.append(str(REPO_ROOT))

from benchmark.kvarn_utils import KVarNLayout, dequant_record  # noqa: E402


def _set_half(cache: torch.Tensor, offset: int, values: torch.Tensor) -> None:
    encoded = values.to(torch.float16).contiguous().view(torch.uint8)
    cache[..., offset : offset + encoded.shape[-1]].copy_(encoded)


def make_cache(records: int, seed: int) -> tuple[torch.Tensor, KVarNLayout]:
    generator = torch.Generator().manual_seed(seed)
    layout = KVarNLayout()
    cache = torch.randint(
        0,
        256,
        (records, 1, layout.tile_bytes_aligned),
        dtype=torch.uint8,
        generator=generator,
    )
    positive_channel = (
        torch.rand(records, 1, layout.head_dim, generator=generator) + 0.125
    )
    positive_token = (
        torch.rand(records, 1, layout.group, generator=generator) + 0.125
    )
    channel_zp = (
        torch.randn(records, 1, layout.head_dim, generator=generator) * 0.1
    )
    token_zp = torch.randn(records, 1, layout.group, generator=generator) * 0.1
    _set_half(cache, layout.k_s_col_offset, positive_channel)
    _set_half(cache, layout.k_zp_offset, channel_zp)
    _set_half(cache, layout.k_s_row_offset, positive_token)
    _set_half(cache, layout.v_s_col_offset, positive_channel)
    _set_half(cache, layout.v_s_row_offset, positive_token)
    _set_half(cache, layout.v_zp_offset, token_zp)
    return cache, layout


def benchmark(records: int, warmup: int, iterations: int, seed: int) -> dict:
    import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401

    cache_cpu, layout = make_cache(records, seed)
    reference_k, reference_v = dequant_record(cache_cpu[0, 0], layout)
    cache = cache_cpu.to("xpu")
    key = torch.empty(
        (records, 1, layout.head_dim, layout.group),
        dtype=torch.float16,
        device="xpu",
    )
    value = torch.empty(
        (records, 1, layout.group, layout.head_dim),
        dtype=torch.float16,
        device="xpu",
    )
    cache_copy = torch.empty_like(cache)

    for _ in range(warmup):
        torch.ops._vllm_fa2_C.kvarn_dequant(cache, key, value)
    torch.xpu.synchronize()
    torch.testing.assert_close(
        key[0, 0].cpu(), reference_k.T.half(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        value[0, 0].cpu(), reference_v.half(), atol=2e-2, rtol=2e-2
    )

    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        torch.ops._vllm_fa2_C.kvarn_dequant(cache, key, value)
        ends[index].record()
    torch.xpu.synchronize()
    elapsed_us = [
        starts[index].elapsed_time(ends[index]) * 1000
        for index in range(iterations)
    ]
    copy_starts = [
        torch.xpu.Event(enable_timing=True) for _ in range(iterations)
    ]
    copy_ends = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        copy_starts[index].record()
        cache_copy.copy_(cache)
        copy_ends[index].record()
    torch.xpu.synchronize()
    copy_elapsed_us = [
        copy_starts[index].elapsed_time(copy_ends[index]) * 1000
        for index in range(iterations)
    ]
    bytes_read = records * layout.tile_bytes
    bytes_written = records * layout.group * layout.head_dim * 2 * 2
    median_us = statistics.median(elapsed_us)
    copy_median_us = statistics.median(copy_elapsed_us)
    copy_bytes = cache.numel() * cache.element_size() * 2
    effective_gbps = (bytes_read + bytes_written) / median_us / 1000
    copy_effective_gbps = copy_bytes / copy_median_us / 1000
    return {
        "provider": "native_xe2_uint32",
        "records": records,
        "warmup": warmup,
        "iterations": iterations,
        "median_us": median_us,
        "mean_us": statistics.fmean(elapsed_us),
        "p05_us": sorted(elapsed_us)[max(0, int(iterations * 0.05) - 1)],
        "p95_us": sorted(elapsed_us)[
            min(iterations - 1, int(iterations * 0.95))
        ],
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "effective_gbps": effective_gbps,
        "copy_median_us": copy_median_us,
        "copy_effective_gbps": copy_effective_gbps,
        "copy_bandwidth_fraction": effective_gbps / copy_effective_gbps,
        "correct": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=188)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.records, args.warmup, args.iterations, args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
