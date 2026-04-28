"""Validate SwiGLU KTDP MLIR against vLLM kernel using ktir_cpu interpreter."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.silu_and_mul.wrapper import silu_and_mul

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "silu_and_mul" / "kernel.ktir.mlir")

NUM_ROWS = 32
D = 1024


def vllm_reference(x_np: np.ndarray) -> np.ndarray:
    """Run vLLM SwiGLU kernel on GPU and return result as numpy."""
    x = torch.from_numpy(x_np.astype(np.float16)).cuda()
    out = silu_and_mul(x)
    return out.cpu().numpy()


def test_silu_and_mul_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((NUM_ROWS, 2 * D)).astype(np.float16)
    Y = np.zeros((NUM_ROWS, D), dtype=np.float16)

    outputs = interp.execute_function("silu_and_mul_kernel", X=X, Y=Y, d=D)
    result = outputs["Y"]
    expected = vllm_reference(X)

    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_silu_and_mul_zeros():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    X = np.zeros((NUM_ROWS, 2 * D), dtype=np.float16)
    Y = np.zeros((NUM_ROWS, D), dtype=np.float16)

    outputs = interp.execute_function("silu_and_mul_kernel", X=X, Y=Y, d=D)
    result = outputs["Y"]
    np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-3)
    print("PASS: zeros input")


