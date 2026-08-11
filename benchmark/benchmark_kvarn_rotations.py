"""Measure the steady-decode Q/output Hadamard rotation tax on XPU."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def hadamard(order: int) -> torch.Tensor:
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / order**0.5


def benchmark(batch: int, heads: int, warmup: int, iterations: int) -> dict:
    rows = batch * heads
    operand = torch.randn(rows, 256, dtype=torch.float16, device="xpu")
    rotation = hadamard(256).to(dtype=torch.float16, device="xpu")
    output = torch.empty_like(operand)
    for _ in range(warmup):
        torch.mm(operand, rotation, out=output)
    torch.xpu.synchronize()

    starts = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        torch.mm(operand, rotation, out=output)
        ends[index].record()
    torch.xpu.synchronize()
    elapsed_us = [
        starts[index].elapsed_time(ends[index]) * 1000
        for index in range(iterations)
    ]
    return {
        "batch": batch,
        "heads": heads,
        "rows": rows,
        "head_dim": 256,
        "warmup": warmup,
        "iterations": iterations,
        "median_us_per_rotation": statistics.median(elapsed_us),
        "mean_us_per_rotation": statistics.fmean(elapsed_us),
        "p95_us_per_rotation": sorted(elapsed_us)[int(iterations * 0.95)],
        "median_us_q_plus_output": statistics.median(elapsed_us) * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.batch, args.heads, args.warmup, args.iterations)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
