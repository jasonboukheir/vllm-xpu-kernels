# SPDX-License-Identifier: Apache-2.0
import math
import os

import pytest
import torch


def _hadamard_256(device: str = "cpu") -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < 256:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(256)).to(device)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 4, 17])
def test_kvarn_hadamard_scatter_matches_fp32(dtype, tokens):
    _load_op()
    generator = torch.Generator().manual_seed(20260808 + tokens)
    key_cpu = torch.randn(tokens, 4, 256, generator=generator).to(dtype)
    value_cpu = torch.randn(tokens, 4, 256, generator=generator).to(dtype)
    slots_cpu = torch.arange(tokens, dtype=torch.int64) * 131
    blocks = int(slots_cpu.max().item() // 128) + 1
    block_to_slot_cpu = torch.arange(blocks, dtype=torch.int32).flip(0)
    pool_size = blocks
    tail_key = torch.full(
        (pool_size, 128, 4, 256), -123.0, dtype=torch.float16, device="xpu"
    )
    tail_value = torch.full_like(tail_key, -123.0)

    torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
        key_cpu.to("xpu"),
        value_cpu.to("xpu"),
        slots_cpu.to("xpu"),
        block_to_slot_cpu.to("xpu"),
        tail_key,
        tail_value,
        128,
        False,
    )
    torch.xpu.synchronize()
    h = _hadamard_256()
    key_ref = torch.matmul(key_cpu.float(), h).half()
    value_ref = torch.matmul(value_cpu.float(), h).half()
    actual_key = tail_key.cpu()
    actual_value = tail_value.cpu()
    for token, logical_slot in enumerate(slots_cpu.tolist()):
        block, position = divmod(logical_slot, 128)
        pool_slot = int(block_to_slot_cpu[block])
        torch.testing.assert_close(
            actual_key[pool_slot, position],
            key_ref[token],
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            actual_value[pool_slot, position],
            value_ref[token],
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 4])
def test_kvarn_fused_qkv_hadamard_scatter_matches_separate_ops(dtype, tokens):
    _load_op()
    generator = torch.Generator().manual_seed(20260904 + tokens)
    qkv = torch.randn(
        tokens, 24 * 256 + 2 * 4 * 256, generator=generator
    ).to(dtype)
    query = qkv[:, : 24 * 256].view(tokens, 24, 256).xpu()
    key = qkv[:, 24 * 256 : 28 * 256].view(tokens, 4, 256).xpu()
    value = qkv[:, 28 * 256 :].view(tokens, 4, 256).xpu()
    slots = torch.arange(tokens, dtype=torch.int64, device="xpu") * 129
    blocks = (tokens - 1) * 129 // 128 + 1
    lookup = torch.arange(blocks, dtype=torch.int32, device="xpu").flip(0)

    separate_query = torch.empty(
        tokens, 24, 256, dtype=torch.float16, device="xpu"
    )
    fused_query = torch.empty_like(separate_query)
    separate_key = torch.full(
        (blocks, 128, 4, 256), -123.0, dtype=torch.float16, device="xpu"
    )
    separate_value = torch.full_like(separate_key, -123.0)
    fused_key = separate_key.clone()
    fused_value = separate_value.clone()

    torch.ops._vllm_fa2_C.kvarn_hadamard(
        query.reshape(-1, 256), separate_query.reshape(-1, 256)
    )
    torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
        key,
        value,
        slots,
        lookup,
        separate_key,
        separate_value,
        128,
        False,
    )
    torch.ops._vllm_fa2_C.kvarn_hadamard_qkv_scatter(
        query,
        key,
        value,
        slots,
        lookup,
        fused_query,
        fused_key,
        fused_value,
        128,
        False,
    )
    torch.xpu.synchronize()

    assert torch.equal(fused_query.cpu(), separate_query.cpu())
    assert torch.equal(fused_key.cpu(), separate_key.cpu())
    assert torch.equal(fused_value.cpu(), separate_value.cpu())


@pytest.mark.parametrize("metadata", ["slot_mapping", "block_to_slot"])
def test_kvarn_fused_qkv_rejects_noncontiguous_metadata(metadata):
    _load_op()
    tokens = 2
    query = torch.zeros(tokens, 24, 256, dtype=torch.float16, device="xpu")
    key = torch.zeros(tokens, 4, 256, dtype=torch.float16, device="xpu")
    value = torch.zeros_like(key)
    slots = torch.arange(tokens * 2, dtype=torch.int64, device="xpu")[::2]
    lookup = torch.arange(4, dtype=torch.int32, device="xpu")[::2]
    if metadata == "slot_mapping":
        assert not slots.is_contiguous()
    else:
        slots = slots.contiguous()
        assert not lookup.is_contiguous()
    query_output = torch.empty_like(query)
    tail_key = torch.empty(2, 128, 4, 256, dtype=torch.float16, device="xpu")
    tail_value = torch.empty_like(tail_key)

    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch.ops._vllm_fa2_C.kvarn_hadamard_qkv_scatter(
            query,
            key,
            value,
            slots,
            lookup,
            query_output,
            tail_key,
            tail_value,
            128,
            False,
        )


def test_kvarn_hadamard_scatter_structured_and_invalid_rows():
    _load_op()
    key = torch.zeros(6, 4, 256, dtype=torch.float16)
    value = torch.zeros_like(key)
    key[0, :, 0] = 16
    value[0, :, 255] = 16
    key[1] = 1
    value[1] = -1
    slots = torch.tensor([0, 129, -1, 256, 384, 640], dtype=torch.int64)
    # Block 2 maps one past the pool, block 3 has no pool slot, and block 5 is
    # outside the lookup. All three rows must leave the canary untouched.
    block_to_slot = torch.tensor([1, 0, 2, -1], dtype=torch.int32)
    sentinel = -77.0
    tail_key = torch.full(
        (2, 128, 4, 256), sentinel, dtype=torch.float16, device="xpu"
    )
    tail_value = torch.full_like(tail_key, sentinel)
    torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
        key.to("xpu"),
        value.to("xpu"),
        slots.to("xpu"),
        block_to_slot.to("xpu"),
        tail_key,
        tail_value,
        128,
        False,
    )
    torch.xpu.synchronize()
    actual_k = tail_key.cpu()
    actual_v = tail_value.cpu()
    torch.testing.assert_close(
        actual_k[1, 0], torch.ones(4, 256, dtype=torch.float16)
    )
    expected_impulse = _hadamard_256()[255].half().mul(16).expand(4, -1)
    torch.testing.assert_close(actual_v[1, 0], expected_impulse)
    expected_constant = torch.zeros(4, 256, dtype=torch.float16)
    expected_constant[:, 0] = 16
    torch.testing.assert_close(actual_k[0, 1], expected_constant)
    torch.testing.assert_close(actual_v[0, 1], -expected_constant)
    untouched = actual_k == sentinel
    assert untouched.sum().item() == untouched.numel() - 2 * 4 * 256
    assert (
        actual_v == sentinel
    ).sum().item() == untouched.numel() - 2 * 4 * 256


def test_kvarn_hadamard_scatter_repeated_is_deterministic():
    _load_op()
    key = torch.randn(4, 4, 256, dtype=torch.bfloat16, device="xpu")
    value = torch.randn_like(key)
    slots = torch.tensor([0, 1, 128, 129], dtype=torch.int64, device="xpu")
    lookup = torch.tensor([1, 0], dtype=torch.int32, device="xpu")
    outputs = []
    for _ in range(3):
        tail_key = torch.full(
            (2, 128, 4, 256), -1, dtype=torch.float16, device="xpu"
        )
        tail_value = torch.full_like(tail_key, -1)
        torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
            key, value, slots, lookup, tail_key, tail_value, 128, False
        )
        outputs.append((tail_key.cpu(), tail_value.cpu()))
    for candidate in outputs[1:]:
        assert torch.equal(candidate[0], outputs[0][0])
        assert torch.equal(candidate[1], outputs[0][1])


def test_kvarn_hadamard_scatter_survives_input_allocator_reuse():
    """Temporary inputs must remain live until the async kernel completes."""
    _load_op()
    generator = torch.Generator().manual_seed(20260812)
    key_cpu = torch.randn(128, 4, 256, generator=generator).bfloat16()
    value_cpu = torch.randn(128, 4, 256, generator=generator).bfloat16()
    slots_cpu = torch.arange(128, dtype=torch.int64)
    lookup_cpu = torch.tensor([0], dtype=torch.int32)
    h = _hadamard_256("xpu").half()
    key_ref = torch.matmul(key_cpu.xpu().half(), h).cpu()
    value_ref = torch.matmul(value_cpu.xpu().half(), h).cpu()

    for _ in range(8):
        tail_key = torch.empty(
            1, 128, 4, 256, dtype=torch.float16, device="xpu"
        )
        tail_value = torch.empty_like(tail_key)
        torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
            key_cpu.xpu(),
            value_cpu.xpu(),
            slots_cpu.xpu(),
            lookup_cpu.xpu(),
            tail_key,
            tail_value,
            128,
            False,
        )
        # Drop every temporary input immediately, then pressure the caching
        # allocator with matching allocations before synchronizing.
        for _ in range(8):
            torch.empty(
                128, 4, 256, dtype=torch.bfloat16, device="xpu"
            ).fill_(37)
        torch.xpu.synchronize()
        torch.testing.assert_close(
            tail_key.cpu(), key_ref.unsqueeze(0), atol=2e-3, rtol=2e-3
        )
        torch.testing.assert_close(
            tail_value.cpu(), value_ref.unsqueeze(0), atol=2e-3, rtol=2e-3
        )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_kvarn_hadamard_scatter_matches_backend_strides_and_appends(batch_size):
    """Match the QKV projection views used by the production backend.

    K and V are non-contiguous views into a token-major QKV allocation.  Calls
    are intentionally split into one-token appends around the KVarN page
    boundary, as they are during autoregressive decode.
    """
    _load_op()
    generator = torch.Generator().manual_seed(46613 + batch_size)
    qkv = torch.randn(
        batch_size, 24 * 256 + 2 * 4 * 256, generator=generator
    ).to(torch.bfloat16)
    key = qkv[:, 24 * 256 : 28 * 256].view(batch_size, 4, 256)
    value = qkv[:, 28 * 256 :].view(batch_size, 4, 256)
    if batch_size > 1:
        assert key.stride(0) == qkv.shape[1]
        assert value.stride(0) == qkv.shape[1]

    logical_slots = torch.tensor([127, 128, 129, 257][:batch_size])
    blocks = int(logical_slots.max().item() // 128) + 1
    block_to_slot = torch.arange(blocks, dtype=torch.int32).flip(0)
    native_key = torch.full(
        (blocks, 128, 4, 256), -123.0, dtype=torch.float16, device="xpu"
    )
    native_value = torch.full_like(native_key, -123.0)
    key_xpu, value_xpu = key.xpu(), value.xpu()
    slots_xpu = logical_slots.xpu()
    lookup_xpu = block_to_slot.xpu()

    # Separate calls cover the decode-time lifetime/stride behavior hidden by
    # the previous one-shot contiguous-input oracle.
    for token in range(batch_size):
        torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
            key_xpu[token : token + 1],
            value_xpu[token : token + 1],
            slots_xpu[token : token + 1],
            lookup_xpu,
            native_key,
            native_value,
            128,
            False,
        )
    torch.xpu.synchronize()

    # This is the established production fallback, including its bf16->fp16
    # boundary cast and XPU fp16 GEMM rounding.  Comparing only with a shared
    # fp32 Sylvester oracle missed the end-to-end accuracy regression.
    h = _hadamard_256("xpu").half()
    key_ref = torch.matmul(key_xpu.half(), h).cpu()
    value_ref = torch.matmul(value_xpu.half(), h).cpu()
    actual_key, actual_value = native_key.cpu(), native_value.cpu()
    for token, logical_slot in enumerate(logical_slots.tolist()):
        block, position = divmod(logical_slot, 128)
        pool_slot = int(block_to_slot[block])
        torch.testing.assert_close(
            actual_key[pool_slot, position],
            key_ref[token],
            atol=2e-3,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            actual_value[pool_slot, position],
            value_ref[token],
            atol=2e-3,
            rtol=2e-3,
        )


def _pack_q4(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(torch.uint8) & 0xF
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


def _balanced_record_reference(
    key_balanced: torch.Tensor,
    key_sinkhorn_col: torch.Tensor,
    key_sinkhorn_row: torch.Tensor,
    value_balanced: torch.Tensor,
    value_sinkhorn_col: torch.Tensor,
    value_sinkhorn_row: torch.Tensor,
    record_bytes: int,
) -> torch.Tensor:
    k_lo = key_balanced.amin(dim=2, keepdim=True)
    k_scale = ((key_balanced.amax(dim=2, keepdim=True) - k_lo) / 15).clamp_min(
        1e-10
    )
    k_q = torch.clamp(torch.round((key_balanced - k_lo) / k_scale), 0, 15)
    v_lo = value_balanced.amin(dim=2, keepdim=True)
    v_scale = (
        (value_balanced.amax(dim=2, keepdim=True) - v_lo) / 15
    ).clamp_min(1e-10)
    v_q = torch.clamp(torch.round((value_balanced - v_lo) / v_scale), 0, 15)
    parts = [
        _pack_dpas_k4(k_q).reshape(key_balanced.shape[0], -1),
        (key_sinkhorn_row * k_scale.squeeze(-1)).half().view(torch.uint8),
        (key_sinkhorn_row * k_lo.squeeze(-1)).half().view(torch.uint8),
        key_sinkhorn_col.half().view(torch.uint8),
        _pack_dpas_v4(v_q).reshape(value_balanced.shape[0], -1),
        value_sinkhorn_col.half().view(torch.uint8),
        (value_sinkhorn_row * v_scale.squeeze(-1)).half().view(torch.uint8),
        (value_sinkhorn_row * v_lo.squeeze(-1)).half().view(torch.uint8),
    ]
    record = torch.cat(parts, dim=1)
    return torch.nn.functional.pad(record, (0, record_bytes - record.shape[1]))


@pytest.mark.parametrize("num_flush_blocks", [1, 3])
@pytest.mark.parametrize("record_bytes", [35072, 65536])
def test_kvarn_balanced_writer_matches_dpas_record_bytes(
    num_flush_blocks, record_bytes
):
    """Cover full pages, ragged block ids, and the padded-record tail."""
    _load_op()
    tiles = num_flush_blocks * 4
    channel = torch.arange(256, dtype=torch.float32)
    token = torch.arange(128, dtype=torch.float32)
    tile = torch.arange(tiles, dtype=torch.float32)

    # Exact integer ranges make the RTN scale exactly one and exercise every
    # q4 code without admitting a CPU/device transcendental tolerance.
    key_balanced = (
        token.remainder(16)[None, None, :]
        + channel.remainder(5)[None, :, None]
        + tile.remainder(3)[:, None, None]
    ).contiguous()
    value_balanced = (
        channel.remainder(16)[None, None, :]
        - token.remainder(7)[None, :, None]
        + tile.remainder(3)[:, None, None]
    ).contiguous()
    key_sinkhorn_col = (
        (tile[:, None] + token[None, :].remainder(11) + 1) / 16
    ).contiguous()
    key_sinkhorn_row = (
        (tile[:, None] + channel[None, :].remainder(13) + 1) / 16
    ).contiguous()
    value_sinkhorn_col = (
        (tile[:, None] + channel[None, :].remainder(17) + 1) / 16
    ).contiguous()
    value_sinkhorn_row = (
        (tile[:, None] + token[None, :].remainder(19) + 1) / 16
    ).contiguous()

    block_ids = torch.tensor([5, 1, 6][:num_flush_blocks], dtype=torch.int64)
    canary = 0xA5
    packed_cache = torch.full(
        (7, 4, record_bytes), canary, dtype=torch.uint8, device="xpu"
    )
    torch.ops._vllm_fa2_C.kvarn_pack_balanced_kv(
        key_balanced.xpu(),
        key_sinkhorn_col.xpu(),
        key_sinkhorn_row.xpu(),
        value_balanced.xpu(),
        value_sinkhorn_col.xpu(),
        value_sinkhorn_row.xpu(),
        block_ids.xpu(),
        packed_cache,
        True,
    )
    torch.xpu.synchronize()

    expected = _balanced_record_reference(
        key_balanced,
        key_sinkhorn_col,
        key_sinkhorn_row,
        value_balanced,
        value_sinkhorn_col,
        value_sinkhorn_row,
        record_bytes,
    ).reshape(num_flush_blocks, 4, record_bytes)
    actual = packed_cache.cpu()
    for index, block in enumerate(block_ids.tolist()):
        assert torch.equal(actual[block], expected[index])
    untouched = set(range(packed_cache.shape[0])) - set(block_ids.tolist())
    for block in untouched:
        assert torch.all(actual[block] == canary)


def test_kvarn_balanced_writer_skips_invalid_ragged_block_ids():
    _load_op()
    block_ids = torch.tensor([-1, 2], dtype=torch.int64, device="xpu")
    tiles = block_ids.numel() * 4
    key_balanced = torch.zeros(tiles, 256, 128, device="xpu")
    value_balanced = torch.zeros(tiles, 128, 256, device="xpu")
    key_sinkhorn_col = torch.ones(tiles, 128, device="xpu")
    key_sinkhorn_row = torch.ones(tiles, 256, device="xpu")
    value_sinkhorn_col = torch.ones(tiles, 256, device="xpu")
    value_sinkhorn_row = torch.ones(tiles, 128, device="xpu")
    packed_cache = torch.full(
        (1, 4, 35072), 0xA5, dtype=torch.uint8, device="xpu"
    )
    torch.ops._vllm_fa2_C.kvarn_pack_balanced_kv(
        key_balanced,
        key_sinkhorn_col,
        key_sinkhorn_row,
        value_balanced,
        value_sinkhorn_col,
        value_sinkhorn_row,
        block_ids,
        packed_cache,
        True,
    )
    torch.xpu.synchronize()
    assert torch.all(packed_cache == 0xA5)
