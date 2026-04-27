"""Validate SwiGLU KTDP MLIR against NumPy reference using ktir_cpu interpreter."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

NUM_ROWS = 32
D = 1024


def silu_and_mul_ref(x: np.ndarray, limit: float = 7.0) -> np.ndarray:
    """NumPy reference: output = clamp(silu(gate)) * clamp(up), where x = [gate || up]."""
    d = x.shape[1] // 2
    gate = x[:, :d].astype(np.float32)
    up = x[:, d:].astype(np.float32)
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    silu_gate = sigmoid_gate * gate
    gate_clamped = np.minimum(silu_gate, limit)
    up_clamped = np.clip(up, -limit, limit)
    return (gate_clamped * up_clamped).astype(np.float16)


def test_silu_and_mul_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((NUM_ROWS, 2 * D)).astype(np.float16)
    Y = np.zeros((NUM_ROWS, D), dtype=np.float16)

    outputs = interp.execute_function("silu_and_mul_kernel", X=X, Y=Y, d=D)
    result = outputs["Y"]
    expected = silu_and_mul_ref(X)

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


if __name__ == "__main__":
    test_silu_and_mul_ktir()
    test_silu_and_mul_zeros()
    print("\nAll SwiGLU KTIR validation tests passed!")
