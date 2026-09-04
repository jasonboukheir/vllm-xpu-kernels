# SPDX-License-Identifier: Apache-2.0
"""Direct-XPU correctness for the raw-page KVarN Sinkhorn writer."""

import os

import pytest
import torch


RECORD_BYTES = 35_072
PACKED_BYTES = 256 * 128 // 2
K_S_COL = PACKED_BYTES
K_ZP = K_S_COL + 256 * 2
K_S_ROW = K_ZP + 256 * 2
V_PACKED = K_S_ROW + 128 * 2
V_S_COL = V_PACKED + PACKED_BYTES
V_S_ROW = V_S_COL + 256 * 2
V_ZP = V_S_ROW + 128 * 2


def _load_op() -> None:
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if library:
        torch.ops.load_library(library)
        return
    import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def xpu_runtime() -> None:
    if not torch.xpu.is_available():
        pytest.skip("an XPU is not available")
    assert torch.xpu.get_device_name(0) == "Intel(R) Arc(TM) Pro B70 Graphics"


def _imbalance(tile: torch.Tensor) -> torch.Tensor:
    col = tile.std(dim=-2)
    row = tile.std(dim=-1)
    return col.amax(dim=-1) / col.amin(dim=-1).clamp_min(1e-8) + row.amax(
        dim=-1
    ) / row.amin(dim=-1).clamp_min(1e-8)


def _sinkhorn(
    tiles: torch.Tensor, iterations: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = tiles.float()
    count, rows, columns = matrix.shape
    log_col = torch.zeros(count, 1, columns)
    log_row = torch.zeros(count, rows, 1)
    current = matrix
    best_imbalance = _imbalance(current)
    best_col = log_col.exp().clone()
    best_row = log_row.exp().clone()
    for _ in range(iterations):
        col_std = current.std(dim=1, keepdim=True).clamp(1e-3, 1e3)
        log_col = (log_col + col_std.log()).clip(-0.3, 10.0)
        current = matrix / log_col.exp() / log_row.exp()
        row_std = current.std(dim=2, keepdim=True).clamp(1e-3, 1e3)
        log_row = (log_row + row_std.log()).clip(-0.3, 10.0)
        current = matrix / log_col.exp() / log_row.exp()
        candidate = _imbalance(current)
        better = candidate <= best_imbalance
        mask = better.view(count, 1, 1)
        best_col = torch.where(mask, log_col.exp(), best_col)
        best_row = torch.where(mask, log_row.exp(), best_row)
        best_imbalance = torch.where(better, candidate, best_imbalance)
    return matrix / best_col / best_row, best_col[:, 0], best_row[:, :, 0]


def _pack_q4(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(torch.uint8)
    return tensor[..., 0::2] | tensor[..., 1::2] << 4


def _pack_dpas_k4(tensor: torch.Tensor) -> torch.Tensor:
    tiles = tensor.shape[0]
    slots = tensor.reshape(tiles, 4, 32, 2, 2, 4, 2, 8).permute(
        0, 4, 1, 5, 7, 3, 2, 6
    )
    return _pack_q4(slots).reshape(tiles, 256, 64)


def _pack_dpas_v4(tensor: torch.Tensor) -> torch.Tensor:
    tiles = tensor.shape[0]
    slots = tensor.reshape(tiles, 2, 4, 8, 2, 8, 2, 2, 8).permute(
        0, 1, 5, 2, 8, 4, 6, 3, 7
    )
    return _pack_q4(slots).reshape(tiles, 128, 128)


def _record_reference(
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    pool_slots: torch.Tensor,
    iterations: int,
    record_bytes: int,
) -> torch.Tensor:
    selected_key = tail_key.index_select(0, pool_slots)
    selected_value = tail_value.index_select(0, pool_slots)
    blocks = pool_slots.numel()
    key_tiles = selected_key.float().permute(0, 2, 3, 1).reshape(-1, 256, 128)
    value_tiles = (
        selected_value.float().permute(0, 2, 1, 3).reshape(-1, 128, 256)
    )
    key_balanced, key_col, key_row = _sinkhorn(key_tiles, iterations)
    value_balanced, value_col, value_row = _sinkhorn(value_tiles, iterations)

    key_lo = key_balanced.amin(dim=2, keepdim=True)
    key_scale = (
        (key_balanced.amax(dim=2, keepdim=True) - key_lo) / 15
    ).clamp_min(1e-10)
    key_q = torch.clamp(torch.round((key_balanced - key_lo) / key_scale), 0, 15)
    value_lo = value_balanced.amin(dim=2, keepdim=True)
    value_scale = (
        (value_balanced.amax(dim=2, keepdim=True) - value_lo) / 15
    ).clamp_min(1e-10)
    value_q = torch.clamp(
        torch.round((value_balanced - value_lo) / value_scale), 0, 15
    )
    parts = [
        _pack_dpas_k4(key_q).reshape(blocks * 4, -1),
        (key_row * key_scale.squeeze(-1)).half().view(torch.uint8),
        (key_row * key_lo.squeeze(-1)).half().view(torch.uint8),
        key_col.half().view(torch.uint8),
        _pack_dpas_v4(value_q).reshape(blocks * 4, -1),
        value_col.half().view(torch.uint8),
        (value_row * value_scale.squeeze(-1)).half().view(torch.uint8),
        (value_row * value_lo.squeeze(-1)).half().view(torch.uint8),
    ]
    record = torch.cat(parts, dim=1)
    return torch.nn.functional.pad(
        record, (0, record_bytes - record.shape[1])
    ).view(blocks, 4, record_bytes)


def _tail_fixture(
    dtype: torch.dtype, pool_size: int = 3
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260905)
    key = torch.randn(pool_size, 128, 4, 256, generator=generator).to(dtype)
    value = (torch.randn(key.shape, generator=generator) * 1.25 - 0.125).to(
        dtype
    )
    return key.contiguous(), value.contiguous()


def _byte_mismatch_evidence(
    actual: torch.Tensor, expected: torch.Tensor
) -> str:
    q_ranges = ((0, K_S_COL), (V_PACKED, V_S_COL))
    q_mismatches = sum(
        int(
            torch.count_nonzero(
                actual[:, start:stop] != expected[:, start:stop]
            )
        )
        for start, stop in q_ranges
    )
    metadata_ranges = ((K_S_COL, V_PACKED), (V_S_COL, RECORD_BYTES))
    actual_metadata = torch.cat(
        [actual[:, start:stop] for start, stop in metadata_ranges], dim=1
    ).contiguous()
    expected_metadata = torch.cat(
        [expected[:, start:stop] for start, stop in metadata_ranges], dim=1
    ).contiguous()
    delta = (
        actual_metadata.view(torch.float16).float()
        - expected_metadata.view(torch.float16).float()
    ).abs()
    return (
        f"q4_byte_mismatches={q_mismatches}; "
        f"metadata_max_abs={float(delta.max())}; "
        f"metadata_mean_abs={float(delta.mean())}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("iterations", [0, 2, 16])
@pytest.mark.parametrize("record_bytes", [RECORD_BYTES, 65_536])
def test_sinkhorn_writer_matches_reference(dtype, iterations, record_bytes):
    """Covers FP16/BF16 pages, Sinkhorn selection, and padded record tails."""
    _load_op()
    tail_key, tail_value = _tail_fixture(dtype)
    pool_slots = torch.tensor([2], dtype=torch.int64)
    block_ids = torch.tensor([3], dtype=torch.int64)
    cache = torch.full(
        (5, 4, record_bytes), 0xA5, dtype=torch.uint8, device="xpu"
    )
    torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
        tail_key.xpu(),
        tail_value.xpu(),
        pool_slots.xpu(),
        block_ids.xpu(),
        cache,
        iterations,
        True,
    )
    torch.xpu.synchronize()
    expected = _record_reference(
        tail_key, tail_value, pool_slots, iterations, record_bytes
    )
    actual = cache[3].cpu()
    # q4 decisions and fp16 metadata should be byte-identical.  If a future
    # compiler changes transcendental precision, this deliberately exposes it
    # rather than weakening cache-ABI correctness to a dequant tolerance.
    assert torch.equal(actual, expected[0]), _byte_mismatch_evidence(
        actual, expected[0]
    )
    assert torch.all(cache[[0, 1, 2, 4]] == 0xA5)


def test_sinkhorn_writer_preserves_rtn_halfway_ties():
    """Zero iterations isolates min/max, clamp and nearest-even q4 semantics."""
    _load_op()
    pattern = torch.tensor(
        [0.0, 15.0, *[code + 0.5 for code in range(15)]], dtype=torch.float16
    ).repeat(8)[:128]
    tail_key = pattern.view(1, 128, 1, 1).expand(1, 128, 4, 256).contiguous()
    value_pattern = pattern.repeat(2)[:256]
    tail_value = (
        value_pattern.view(1, 1, 1, 256).expand_as(tail_key).contiguous()
    )
    indices = torch.tensor([0], dtype=torch.int64)
    cache = torch.empty(1, 4, RECORD_BYTES, dtype=torch.uint8, device="xpu")
    torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
        tail_key.xpu(),
        tail_value.xpu(),
        indices.xpu(),
        indices.xpu(),
        cache,
        0,
        True,
    )
    torch.xpu.synchronize()
    assert torch.equal(
        cache.cpu(),
        _record_reference(tail_key, tail_value, indices, 0, RECORD_BYTES),
    )


def test_sinkhorn_writer_masks_ragged_ownership():
    _load_op()
    tail_key, tail_value = _tail_fixture(torch.float16)
    pool_slots = torch.tensor([2, -1, 0, 99], dtype=torch.int64, device="xpu")
    block_ids = torch.tensor([5, 3, 1, 2], dtype=torch.int64, device="xpu")
    cache = torch.full(
        (7, 4, RECORD_BYTES), 0xA5, dtype=torch.uint8, device="xpu"
    )
    torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
        tail_key.xpu(), tail_value.xpu(), pool_slots, block_ids, cache, 0, True
    )
    torch.xpu.synchronize()
    actual = cache.cpu()
    expected = _record_reference(
        tail_key, tail_value, torch.tensor([2, 0]), 0, RECORD_BYTES
    )
    assert torch.equal(actual[5], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.all(actual[[0, 2, 3, 4, 6]] == 0xA5)


def test_sinkhorn_writer_uses_int64_long_context_record_addressing():
    _load_op()
    tail_key, tail_value = _tail_fixture(torch.float16, pool_size=1)
    long_block = 2047  # More than 262K / 128 logical token pages.
    cache = torch.full(
        (long_block + 1, 4, RECORD_BYTES), 0xA5, dtype=torch.uint8, device="xpu"
    )
    zero = torch.tensor([0], dtype=torch.int64, device="xpu")
    block = torch.tensor([long_block], dtype=torch.int64, device="xpu")
    torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
        tail_key.xpu(), tail_value.xpu(), zero, block, cache, 0, True
    )
    torch.xpu.synchronize()
    expected = _record_reference(
        tail_key, tail_value, torch.tensor([0]), 0, RECORD_BYTES
    )
    assert torch.equal(cache[long_block].cpu(), expected[0])
    assert torch.all(cache[long_block - 1] == 0xA5)


def test_sinkhorn_writer_handles_multiple_valid_blocks_at_iteration16():
    _load_op()
    tail_key, tail_value = _tail_fixture(torch.float16, pool_size=4)
    original_key = tail_key.clone()
    original_value = tail_value.clone()
    pool_slots_cpu = torch.tensor([3, 0, 2], dtype=torch.int64)
    block_ids = torch.tensor([6, 1, 4], dtype=torch.int64, device="xpu")
    tail_key_xpu = tail_key.xpu()
    tail_value_xpu = tail_value.xpu()
    cache = torch.full(
        (8, 4, RECORD_BYTES), 0xA5, dtype=torch.uint8, device="xpu"
    )

    torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
        tail_key_xpu,
        tail_value_xpu,
        pool_slots_cpu.xpu(),
        block_ids,
        cache,
        16,
        True,
    )
    torch.xpu.synchronize()

    expected = _record_reference(
        tail_key, tail_value, pool_slots_cpu, 16, RECORD_BYTES
    )
    actual = cache.cpu()
    for output_block, reference_index in zip([6, 1, 4], range(3)):
        assert torch.equal(
            actual[output_block], expected[reference_index]
        ), _byte_mismatch_evidence(actual[output_block], expected[reference_index])
    assert torch.all(actual[[0, 2, 3, 5, 7]] == 0xA5)
    assert torch.equal(tail_key_xpu.cpu(), original_key)
    assert torch.equal(tail_value_xpu.cpu(), original_value)


def test_sinkhorn_writer_empty_schedule_is_noop():
    _load_op()
    tail_key, tail_value = _tail_fixture(torch.float16, pool_size=1)
    tail_key_xpu = tail_key.xpu()
    tail_value_xpu = tail_value.xpu()
    empty = torch.empty(0, dtype=torch.int64, device="xpu")
    cache = torch.full(
        (2, 4, RECORD_BYTES), 0xA5, dtype=torch.uint8, device="xpu"
    )

    torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
        tail_key_xpu, tail_value_xpu, empty, empty, cache, 16, True
    )
    torch.xpu.synchronize()

    assert torch.all(cache == 0xA5)
    assert torch.equal(tail_key_xpu.cpu(), tail_key)
    assert torch.equal(tail_value_xpu.cpu(), tail_value)


def test_sinkhorn_writer_rejects_non_abi_inputs():
    _load_op()
    tail_key, tail_value = _tail_fixture(torch.float16, pool_size=1)
    zero = torch.tensor([0], dtype=torch.int64, device="xpu")
    cache = torch.empty(1, 4, RECORD_BYTES, dtype=torch.uint8, device="xpu")
    with pytest.raises(RuntimeError, match="requires xe2_dpas"):
        torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
            tail_key.xpu(), tail_value.xpu(), zero, zero, cache, 16, False
        )
    with pytest.raises(RuntimeError, match="between 0 and 64"):
        torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
            tail_key.xpu(), tail_value.xpu(), zero, zero, cache, 65, True
        )
    with pytest.raises(RuntimeError, match="between 0 and 64"):
        torch.ops._vllm_fa2_C.kvarn_sinkhorn_pack_kv(
            tail_key.xpu(), tail_value.xpu(), zero, zero, cache, -1, True
        )
