"""Validate matmul KTDP MLIR against Triton kernel using ktir_cpu interpreter.

Run: pytest tests/ktir/test_matmul.py -v
"""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.vllm.matmul.wrapper import matmul

MLIR_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "kernels" / "vllm" / "matmul" / "kernel.ktir"
)

M = 128
N = 128
K = 128


def triton_reference(a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
    """Run Triton matmul kernel on GPU and return result as numpy."""
    a = torch.from_numpy(a_np.astype(np.float16)).cuda()
    b = torch.from_numpy(b_np.astype(np.float16)).cuda()
    out = matmul(a, b)
    return out.cpu().numpy()


def test_matmul_ktir():
    """Run matmul KTDP kernel and compare against Triton kernel."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    outputs = interp.execute_function(
        "matmul_kernel",
        A=A,
        B=B,
        C=C,
        M=M,
        N=N,
        K=K,
    )

    result = outputs["C"]
    expected = triton_reference(A, B)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-2, atol=1e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_matmul_ktir_identity():
    """Multiplying by identity matrix should return the original."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(7)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = np.eye(K, N, dtype=np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    outputs = interp.execute_function(
        "matmul_kernel",
        A=A,
        B=B,
        C=C,
        M=M,
        N=N,
        K=K,
    )

    result = outputs["C"]
    np.testing.assert_allclose(
        result.astype(np.float32), A.astype(np.float32),
        rtol=1e-3, atol=1e-3,
    )
    print("PASS: identity matrix multiplication")


def test_matmul_ktir_zeros():
    """Multiplying by zeros should give zeros."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(99)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = np.zeros((K, N), dtype=np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    outputs = interp.execute_function(
        "matmul_kernel",
        A=A,
        B=B,
        C=C,
        M=M,
        N=N,
        K=K,
    )

    result = outputs["C"]
    np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-4)
    print("PASS: zero matrix multiplication")
