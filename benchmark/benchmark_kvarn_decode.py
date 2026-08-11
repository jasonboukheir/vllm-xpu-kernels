"""Benchmark the narrow native KVarN Xe2 decode kernel and emit JSON."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmark.check_kvarn_decode import make_cache
from benchmark.kvarn_utils import swizzle_record_dpas_k4v4


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def parse_seq_lens(value: str, batch: int, context: int) -> list[int]:
    lengths = [int(item) for item in value.split(",")]
    if len(lengths) != batch:
        raise ValueError(f"--seq-lens requires {batch} comma-separated values")
    if any(length < 1 or length > context for length in lengths):
        raise ValueError(f"--seq-lens values must be in [1, {context}]")
    return lengths


def nonempty_split_workgroups(
    lengths: list[int], context: int, kv_heads: int = 4, splits: int = 16
) -> int:
    max_blocks = (context + 63) // 64
    blocks_per_split = (max_blocks + splits - 1) // splits
    active_splits = sum(
        min(
            splits,
            ((length + 63) // 64 + blocks_per_split - 1) // blocks_per_split,
        )
        for length in lengths
    )
    return kv_heads * active_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument(
        "--batch", type=int, choices=(1, 2, 3, 4, 12), default=4
    )
    parser.add_argument("--context", type=int, default=6000)
    parser.add_argument(
        "--seq-lens",
        help=(
            "comma-separated ragged lengths (context remains the grid maximum)"
        ),
    )
    parser.add_argument(
        "--dpas-layout",
        action="store_true",
        help="Benchmark the feature-flagged DPAS-native packed payload layout.",
    )
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--persistent-scratch",
        action="store_true",
        help=(
            "Use caller-owned native split scratch instead of "
            "per-call allocation."
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        seq_lens_cpu = (
            parse_seq_lens(args.seq_lens, args.batch, args.context)
            if args.seq_lens
            else [args.context] * args.batch
        )
    except ValueError as error:
        parser.error(str(error))

    torch.ops.load_library(args.library)
    pages = (args.context + 127) // 128
    cpu_cache, layout = make_cache(pages)
    if args.dpas_layout:
        for block in range(cpu_cache.shape[0]):
            for head in range(cpu_cache.shape[1]):
                cpu_cache[block, head] = swizzle_record_dpas_k4v4(
                    cpu_cache[block, head], layout
                )
        os.environ["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "1"
    cache = cpu_cache.xpu()
    query = torch.randn(
        (args.batch, 24, 256), dtype=torch.float16, device="xpu"
    )
    output = torch.empty_like(query)
    table = torch.arange(pages, dtype=torch.int32, device="xpu").repeat(
        args.batch, 1
    )
    block_to_slot = torch.full((pages,), -1, dtype=torch.int32, device="xpu")
    seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int32, device="xpu")
    tail_key = torch.zeros((1, 128, 4, 256), dtype=torch.float16, device="xpu")
    tail_value = torch.zeros_like(tail_key)
    split_count = int(os.environ.get("KVARN_NATIVE_XPU_SPLITS", "1"))
    temp_output = torch.empty(
        args.batch,
        24 * split_count,
        256,
        dtype=torch.float16,
        device="xpu",
    )
    exp_sums = torch.empty(
        args.batch, 24, split_count, dtype=torch.float32, device="xpu"
    )
    max_logits = torch.empty_like(exp_sums)

    def launch() -> None:
        decode_op = (
            torch.ops._vllm_fa2_C.kvarn_decode_with_scratch
            if args.persistent_scratch
            else torch.ops._vllm_fa2_C.kvarn_decode
        )
        scratch_args = (
            (temp_output, exp_sums, max_logits)
            if args.persistent_scratch
            else ()
        )
        decode_op(
            query,
            cache,
            table,
            seq_lens,
            block_to_slot,
            tail_key,
            tail_value,
            *scratch_args,
            output,
            args.context,
            1.0 / 16.0,
        )

    for _ in range(args.warmup):
        launch()
    torch.xpu.synchronize()

    samples_us = []
    device_samples_us = []
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        start_event.record()
        launch()
        end_event.record()
        torch.xpu.synchronize()
        samples_us.append((time.perf_counter_ns() - start) / 1000)
        device_samples_us.append(start_event.elapsed_time(end_event) * 1000)

    result = {
        "implementation": "native-kvarn-k4v4-d256-g128",
        "batch": args.batch,
        "context": args.context,
        "seq_lens": seq_lens_cpu,
        "fixed_split_workgroups": args.batch * 4 * 16,
        "nonempty_split_workgroups": nonempty_split_workgroups(
            seq_lens_cpu, args.context
        ),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "p10_us": percentile(samples_us, 0.10),
        "p90_us": percentile(samples_us, 0.90),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "device_median_us": statistics.median(device_samples_us),
        "device_mean_us": statistics.mean(device_samples_us),
        "host_overhead_median_us": statistics.median(samples_us)
        - statistics.median(device_samples_us),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n")


if __name__ == "__main__":
    main()
