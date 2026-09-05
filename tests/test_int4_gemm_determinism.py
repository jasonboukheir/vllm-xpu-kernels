# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


GROUP_SIZE = 128
SHAPES = [
    pytest.param(1, 5120, 8192, id="decode-attention"),
    pytest.param(162, 5120, 8192, id="prefill-attention"),
    pytest.param(1, 5120, 16384, id="decode-gdn-qkvz"),
    pytest.param(162, 5120, 16384, id="prefill-gdn-qkvz"),
    pytest.param(107, 5120, 16384, id="gdn-qkvz-t107"),
    pytest.param(107, 5120, 96, id="gdn-ba-t107"),
    pytest.param(107, 6144, 5120, id="gdn-out-t107"),
    pytest.param(107, 5120, 34816, id="mlp-gate-up-t107"),
    pytest.param(107, 17408, 5120, id="mlp-down-t107"),
]

BATCH_SEPARABILITY_SHAPES = [
    pytest.param(5120, 8192, id="attention"),
    pytest.param(5120, 16384, id="gdn-qkvz"),
    pytest.param(5120, 96, id="gdn-ba"),
    pytest.param(6144, 5120, id="gdn-out"),
    pytest.param(5120, 34816, id="mlp-gate-up"),
    pytest.param(17408, 5120, id="mlp-down"),
]


def _packed_int4_weight(k, n, device):
    packed_bytes = torch.randint(
        -128,
        128,
        (k * n // 2,),
        dtype=torch.int8,
        device=device,
    )
    weight = packed_bytes.view(torch.int32).reshape(k // 8, n)
    # oneDNN expects packed weights in NT layout: shape [K/8, N], stride [1,
    # K/8]. Keep the non-contiguous view alive through the call.
    return weight.transpose(0, 1).contiguous().transpose(0, 1)


@pytest.mark.parametrize("m,k,n", SHAPES)
@torch.inference_mode()
def test_int4_gemm_w4a16_bf16_exact_shape_is_deterministic(m, k, n):
    """Production BF16 W4 group-128 GEMMs must be bitwise repeatable."""
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260829 + n + m)

    input_tensor = torch.randn((m, k), dtype=dtype, device=device)
    weight = _packed_int4_weight(k, n, device)
    scales = torch.rand((k // GROUP_SIZE, n), dtype=dtype, device=device).mul_(
        0.05
    )
    decoy_input = torch.randn_like(input_tensor)
    decoy_weight = _packed_int4_weight(k, n, device)
    decoy_scales = torch.rand_like(scales).mul_(0.05)
    # compressed-tensors symmetric INT4 uses a scalar packed-domain zero point.
    zero_points = torch.tensor([8], dtype=torch.int8, device=device)

    repeats = 4 if m > 1 else 8
    expected = None
    for repeat in range(repeats):
        # Production reuses the cached oneDNN primitive shape for many layers
        # with distinct weight/scale pointers. Exercise that handle update
        # before restoring the target tensors on every repeat after warm-up.
        if repeat > 0:
            decoy_output = torch.ops._xpu_C.int4_gemm_w4a16(
                decoy_input,
                decoy_weight,
                None,
                decoy_scales,
                zero_points,
                GROUP_SIZE,
                None,
            )
            del decoy_output

        # Change output/scratch allocation placement while retaining the fixed
        # inputs, packed weights, scales, and primitive shape.
        pressure = torch.empty(
            ((repeat + 1) * 1_048_573,), dtype=torch.uint8, device=device
        )
        pressure.fill_(repeat + 1)

        output = torch.ops._xpu_C.int4_gemm_w4a16(
            input_tensor,
            weight,
            None,
            scales,
            zero_points,
            GROUP_SIZE,
            None,
        )
        torch.xpu.synchronize()
        snapshot = output.cpu()

        if expected is None:
            expected = snapshot
        elif not torch.equal(snapshot, expected):
            mismatch = snapshot != expected
            max_abs_diff = (
                (snapshot.float() - expected.float()).abs().max().item()
            )
            pytest.fail(
                f"BF16 W4 GEMM changed at M={m}, N={n}, K={k}, "
                f"repeat={repeat}: mismatches={mismatch.sum().item()}, "
                f"max_abs_diff={max_abs_diff}"
            )

        del output, pressure
        torch.xpu.empty_cache()


@pytest.mark.parametrize("k,n", BATCH_SEPARABILITY_SHAPES)
@torch.inference_mode()
def test_int4_gemm_w4a16_bf16_is_batch_separable(
        k, n, record_property):
    """M=4 and four M=1 calls must remain numerically equivalent.

    oneDNN determinism guarantees repeatability for a fixed primitive shape;
    it does not by itself promise bitwise identity between different M values.
    Keep exactness as recorded diagnostic evidence, while the correctness gate
    allows two BF16 rounding quanta for a shape-dependent accumulation path.
    """
    device = "xpu"
    dtype = torch.bfloat16
    torch.manual_seed(20260830 + k + n)

    input_tensor = torch.randn((4, k), dtype=dtype, device=device)
    weight = _packed_int4_weight(k, n, device)
    scales = torch.rand(
        (k // GROUP_SIZE, n), dtype=dtype, device=device).mul_(0.05)
    zero_points = torch.tensor([8], dtype=torch.int8, device=device)

    batched = torch.ops._xpu_C.int4_gemm_w4a16(
        input_tensor,
        weight,
        None,
        scales,
        zero_points,
        GROUP_SIZE,
        None,
    )
    independent = torch.cat([
        torch.ops._xpu_C.int4_gemm_w4a16(
            input_tensor[row:row + 1],
            weight,
            None,
            scales,
            zero_points,
            GROUP_SIZE,
            None,
        ) for row in range(input_tensor.shape[0])
    ])
    torch.xpu.synchronize()

    batched_cpu = batched.cpu()
    independent_cpu = independent.cpu()
    delta = (batched_cpu.float() - independent_cpu.float()).abs()
    exact = torch.equal(batched_cpu, independent_cpu)
    mismatches = int((batched_cpu != independent_cpu).sum().item())
    max_abs_diff = float(delta.max().item())
    mean_abs_diff = float(delta.mean().item())
    record_property("bitwise_exact", exact)
    record_property("mismatched_elements", mismatches)
    record_property("max_abs_diff", max_abs_diff)
    record_property("mean_abs_diff", mean_abs_diff)
    print(
        f"W4A16 M=4 vs 4xM=1 at K={k}, N={n}: exact={exact}, "
        f"mismatches={mismatches}, max_abs_diff={max_abs_diff}, "
        f"mean_abs_diff={mean_abs_diff}")

    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        batched_cpu.float(),
        independent_cpu.float(),
        atol=tolerance,
        rtol=tolerance,
    )
