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
