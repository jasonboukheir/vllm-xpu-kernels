import math
import os

import pytest
import torch


def _load_op() -> None:
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if library:
        torch.ops.load_library(library)
        return
    import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401


def _hadamard_256() -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < 256:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(256)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [24, 48, 72, 96])
def test_kvarn_hadamard_matches_independent_fp32_oracle(dtype, rows):
    _load_op()
    generator = torch.Generator().manual_seed(20260808 + rows)
    input_cpu = torch.randn(rows, 256, generator=generator).to(dtype)
    output = torch.empty(rows, 256, dtype=torch.float16, device="xpu")
    torch.ops._vllm_fa2_C.kvarn_hadamard(input_cpu.to("xpu"), output)
    torch.xpu.synchronize()
    expected = torch.mm(input_cpu.float(), _hadamard_256()).half()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_kvarn_hadamard_structured_vectors_and_determinism():
    _load_op()
    input_cpu = torch.zeros(3, 256, dtype=torch.bfloat16)
    input_cpu[0, 0] = 16
    input_cpu[1, 255] = 16
    input_cpu[2] = 1
    expected = torch.mm(input_cpu.float(), _hadamard_256()).half()
    outputs = []
    for _ in range(3):
        output = torch.empty_like(input_cpu, dtype=torch.float16, device="xpu")
        torch.ops._vllm_fa2_C.kvarn_hadamard(input_cpu.to("xpu"), output)
        outputs.append(output.cpu())
    torch.testing.assert_close(outputs[0], expected)
    assert torch.equal(outputs[1], outputs[0])
    assert torch.equal(outputs[2], outputs[0])
