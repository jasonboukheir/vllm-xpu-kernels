"""Runtime correctness smoke test for the narrow native KVarN decode op."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmark.kvarn_utils import KVarNLayout, dequant_record


def _put_half(record: torch.Tensor, offset: int, values: torch.Tensor) -> None:
    raw = values.to(torch.float16).contiguous().view(torch.uint8)
    record[offset : offset + raw.numel()].copy_(raw)


def make_cache(
    num_blocks: int, record_stride: int | None = None
) -> tuple[torch.Tensor, KVarNLayout]:
    layout = KVarNLayout(record_stride=record_stride)
    cache = torch.zeros(
        num_blocks, 4, layout.tile_bytes_aligned, dtype=torch.uint8
    )
    ones_d = torch.ones(layout.head_dim)
    ones_g = torch.ones(layout.group)
    for block in range(num_blocks):
        for head in range(4):
            record = cache[block, head]
            _put_half(record, layout.k_s_col_offset, ones_d)
            _put_half(record, layout.k_zp_offset, torch.zeros_like(ones_d))
            _put_half(record, layout.k_s_row_offset, ones_g)
            _put_half(record, layout.v_s_col_offset, ones_d)
            _put_half(record, layout.v_s_row_offset, ones_g)
            # With packed V nibbles zero, each logical token is a distinct
            # constant vector. The block term makes page-table swaps visible.
            values = block * 0.25 + torch.arange(layout.group) / 1024
            _put_half(record, layout.v_zp_offset, values)
    return cache, layout


def expected_value(block_table: list[int], seq_len: int) -> float:
    values = []
    for logical_token in range(seq_len):
        block = block_table[logical_token // 128]
        token = logical_token % 128
        values.append(block * 0.25 + token / 1024)
    return sum(values) / len(values)


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.uint8)
    return (values[..., 0::2] | (values[..., 1::2] << 4)).flatten()


def make_random_cache(
    num_blocks: int, record_stride: int | None = None
) -> tuple[torch.Tensor, KVarNLayout]:
    layout = KVarNLayout(record_stride=record_stride)
    generator = torch.Generator().manual_seed(20260808)
    cache = torch.zeros(
        num_blocks, 4, layout.tile_bytes_aligned, dtype=torch.uint8
    )
    for block in range(num_blocks):
        for head in range(4):
            record = cache[block, head]
            qk = torch.randint(
                0, 16, (layout.head_dim, layout.group), generator=generator
            )
            qv = torch.randint(
                0, 16, (layout.group, layout.head_dim), generator=generator
            )
            k_raw = _pack_nibbles(qk)
            v_raw = _pack_nibbles(qv)
            record[: layout.k_packed_bytes].copy_(k_raw)
            record[
                layout.v_packed_offset : layout.v_packed_offset
                + layout.v_packed_bytes
            ].copy_(v_raw)
            _put_half(
                record,
                layout.k_s_col_offset,
                torch.rand(layout.head_dim, generator=generator) * 0.02,
            )
            _put_half(
                record,
                layout.k_zp_offset,
                torch.rand(layout.head_dim, generator=generator) * 0.04 - 0.02,
            )
            _put_half(
                record,
                layout.k_s_row_offset,
                torch.rand(layout.group, generator=generator) * 0.5 + 0.5,
            )
            _put_half(
                record,
                layout.v_s_col_offset,
                torch.rand(layout.head_dim, generator=generator) * 0.02,
            )
            _put_half(
                record,
                layout.v_s_row_offset,
                torch.rand(layout.group, generator=generator) * 0.5 + 0.5,
            )
            _put_half(
                record,
                layout.v_zp_offset,
                torch.rand(layout.group, generator=generator) * 0.04 - 0.02,
            )
    return cache, layout


def reference_decode(
    query: torch.Tensor,
    cache: torch.Tensor,
    layout: KVarNLayout,
    block_tables: list[list[int]],
    seq_len: int,
    scale: float,
) -> torch.Tensor:
    output = torch.empty_like(query, dtype=torch.float32)
    for batch, pages in enumerate(block_tables):
        for kv_head in range(4):
            keys = []
            values = []
            for page in pages[: (seq_len + 127) // 128]:
                key, value = dequant_record(cache[page, kv_head], layout)
                keys.append(key)
                values.append(value)
            key = torch.cat(keys)[:seq_len]
            value = torch.cat(values)[:seq_len]
            for local_head in range(6):
                query_head = kv_head * 6 + local_head
                scores = query[batch, query_head].float() @ key.T * scale
                output[batch, query_head] = (
                    torch.softmax(scores, dim=-1) @ value
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument(
        "--record-stride",
        type=int,
        choices=(35072, 65536),
        default=35072,
        help="bytes between adjacent KVarN head records",
    )
    args = parser.parse_args()
    torch.ops.load_library(args.library)

    cpu_cache, _ = make_cache(3, args.record_stride)
    assert cpu_cache.shape == (3, 4, args.record_stride)
    cache = cpu_cache.xpu()
    block_to_slot = torch.full((3,), -1, dtype=torch.int32, device="xpu")
    tail_key = torch.zeros((1, 128, 4, 256), dtype=torch.float16, device="xpu")
    tail_value = torch.zeros_like(tail_key)
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device="xpu")
    output = torch.empty_like(query)
    pages = [2, 0, 1]
    block_table = torch.tensor([pages], dtype=torch.int32, device="xpu")
    seq_lens = torch.empty((1,), dtype=torch.int32, device="xpu")

    for seq_len in (1, 63, 64, 65, 127, 128, 129, 255, 256, 257):
        seq_lens.fill_(seq_len)
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            cache,
            block_table,
            seq_lens,
            block_to_slot,
            tail_key,
            tail_value,
            output,
            seq_len,
            1.0 / 16.0,
        )
        torch.xpu.synchronize()
        expected = torch.full_like(output.cpu(), expected_value(pages, seq_len))
        actual = output[0, 0, 0].item()
        print(
            f"seq_len={seq_len}: actual={actual:.6f} "
            f"expected={expected[0, 0, 0].item():.6f}"
        )
        torch.testing.assert_close(output.cpu(), expected, atol=2e-3, rtol=2e-3)
        if not torch.isfinite(output).all():
            raise AssertionError(f"non-finite output at seq_len={seq_len}")
        print(f"seq_len={seq_len}: pass")

    block_to_slot[2] = 0
    tail_value.fill_(0.75)
    seq_lens.fill_(129)
    torch.ops._vllm_fa2_C.kvarn_decode(
        query,
        cache,
        block_table,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
        output,
        129,
        1.0 / 16.0,
    )
    torch.xpu.synchronize()
    hybrid_expected = torch.full_like(output.cpu(), (128 * 0.75) / 129)
    torch.testing.assert_close(
        output.cpu(), hybrid_expected, atol=2e-3, rtol=2e-3
    )
    print("hybrid FP16 page + packed page: pass")

    random_cache, random_layout = make_random_cache(6, args.record_stride)
    random_query = torch.randn(
        (4, 24, 256), generator=torch.Generator().manual_seed(7)
    ).to(torch.float16)
    random_pages = [[5, 1], [4, 0], [3, 2], [1, 5]]
    random_table = torch.tensor(random_pages, dtype=torch.int32, device="xpu")
    random_map = torch.full((6,), -1, dtype=torch.int32, device="xpu")
    random_lengths = [1, 63, 128, 129]
    random_seq_lens = torch.tensor(
        random_lengths, dtype=torch.int32, device="xpu"
    )
    random_output = torch.empty_like(random_query, device="xpu")
    scale = 1.0 / 16.0
    torch.ops._vllm_fa2_C.kvarn_decode(
        random_query.xpu(),
        random_cache.xpu(),
        random_table,
        random_seq_lens,
        random_map,
        tail_key,
        tail_value,
        random_output,
        129,
        scale,
    )
    torch.xpu.synchronize()
    reference = torch.cat(
        [
            reference_decode(
                random_query[batch : batch + 1],
                random_cache,
                random_layout,
                [random_pages[batch]],
                random_lengths[batch],
                scale,
            )
            for batch in range(4)
        ]
    )
    random_actual = random_output.cpu().float()
    errors = (random_actual - reference).abs()
    for batch in range(4):
        head_errors = errors[batch].amax(dim=-1)
        print(
            f"random batch={batch} per-head-max="
            f"{[round(value, 4) for value in head_errors.tolist()]}"
        )
    torch.testing.assert_close(random_actual, reference, atol=2e-2, rtol=2e-2)
    max_error = errors.max().item()
    mean_error = errors.mean().item()
    print(
        f"random B4/GQA: pass max_error={max_error:.6g} mean={mean_error:.6g}"
    )


if __name__ == "__main__":
    main()
