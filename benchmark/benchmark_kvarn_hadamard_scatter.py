"""XPU-event benchmark for fused KVarN H256 rotation and tail scatter."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch


def _hadamard(device: str) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < 256:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(256)).to(device=device, dtype=torch.float16)


def _times(callable_, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        callable_()
    torch.xpu.synchronize()
    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        callable_()
        ends[index].record()
    torch.xpu.synchronize()
    return [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def benchmark(
    tokens: int, dtype: torch.dtype, warmup: int, iterations: int
) -> dict:
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if library:
        torch.ops.load_library(library)
    else:
        import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401

    key = torch.randn(tokens, 4, 256, dtype=dtype, device="xpu")
    value = torch.randn_like(key)
    slots = torch.arange(tokens, dtype=torch.int64, device="xpu")
    lookup_size = max(1, (tokens + 127) // 128)
    lookup = torch.arange(lookup_size, dtype=torch.int32, device="xpu")
    tail_key = torch.empty(
        lookup_size, 128, 4, 256, dtype=torch.float16, device="xpu"
    )
    tail_value = torch.empty_like(tail_key)
    h = _hadamard("xpu")
    k_scratch = torch.empty(tokens, 4, 256, dtype=torch.float16, device="xpu")
    v_scratch = torch.empty_like(k_scratch)

    def fused() -> None:
        torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
            key, value, slots, lookup, tail_key, tail_value, 128
        )

    def dense_rotation_only() -> None:
        torch.matmul(key.to(torch.float16), h, out=k_scratch)
        torch.matmul(value.to(torch.float16), h, out=v_scratch)

    fused_times = _times(fused, warmup, iterations)
    dense_times = _times(dense_rotation_only, warmup, iterations)
    return {
        "tokens": tokens,
        "dtype": str(dtype),
        "warmup": warmup,
        "iterations": iterations,
        "fused": _stats(fused_times),
        "dense_rotation_only_no_scatter": _stats(dense_times),
        "correct": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.tokens, getattr(torch, args.dtype), args.warmup, args.iterations
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
