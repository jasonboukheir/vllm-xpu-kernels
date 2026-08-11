"""Event-timed native H256 FWHT versus torch.mm for decode query rows."""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def event_median_us(launch, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        launch()
    torch.xpu.synchronize()
    samples = []
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    torch.ops.load_library(args.library)
    h = torch.ones(1, 1)
    while h.shape[0] < 256:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h = (h / 16).to(dtype=torch.float16, device="xpu")
    results = []
    for dtype in (torch.float16, torch.bfloat16):
        for rows in (24, 48, 72, 96):
            x = torch.randn(rows, 256, dtype=dtype, device="xpu")
            native_out = torch.empty(
                rows, 256, dtype=torch.float16, device="xpu"
            )
            mm_out = torch.empty_like(native_out)
            mm_input = x if dtype == torch.float16 else x.to(torch.float16)
            native_us = event_median_us(
                lambda x=x, native_out=native_out: (
                    torch.ops._vllm_fa2_C.kvarn_hadamard(x, native_out)
                ),
                args.warmup,
                args.iterations,
            )
            mm_us = event_median_us(
                lambda mm_input=mm_input, mm_out=mm_out: torch.mm(
                    mm_input, h, out=mm_out
                ),
                args.warmup,
                args.iterations,
            )
            results.append(
                {
                    "dtype": str(dtype),
                    "rows": rows,
                    "native_device_median_us": native_us,
                    "torch_mm_device_median_us": mm_us,
                    "speedup": mm_us / native_us,
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
