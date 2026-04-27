# SPDX-License-Identifier: Apache-2.0
"""Validate RMSNorm KTDP MLIR against NumPy reference using ktir_cpu interpreter.

Run from the ktir_cpu venv:
    cd external/ktir_cpu && uv run python ../../kernels/rms_norm/test_ktir.py
"""

import sys
from pathlib import Path

import numpy as np

# Ensure ktir_cpu is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))

from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(
    Path(__file__).resolve().parent / "kernel.ktir.mlir"
)

# Concrete sizes matching the MLIR
NUM_ROWS = 32
N_COLS = 4096
BLOCK_SIZE = 1024
EPS = np.float16(1e-5)


def rms_norm_ref(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """NumPy reference RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight."""
    x_f32 = x.astype(np.float32)
    w_f32 = weight.astype(np.float32)
    mean_sq = np.mean(x_f32 ** 2, axis=1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_sq + float(eps))
    return (x_f32 * inv_rms * w_f32).astype(np.float16)


def test_rms_norm_ktir():
    """Run RMSNorm KTDP kernel on 32 cores and compare against NumPy reference."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((NUM_ROWS, N_COLS)).astype(np.float16)
    W = rng.standard_normal(N_COLS).astype(np.float16)
    Y = np.zeros((NUM_ROWS, N_COLS), dtype=np.float16)

    outputs = interp.execute_function(
        "rms_norm_fwd",
        X=X,
        W=W,
        Y=Y,
        N=N_COLS,
        eps=EPS,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    result_Y = outputs["Y"]
    expected_Y = rms_norm_ref(X, W, EPS)

    np.testing.assert_allclose(result_Y, expected_Y, rtol=1e-2, atol=1e-2)
    print(f"PASS: max abs error = {np.max(np.abs(result_Y.astype(np.float32) - expected_Y.astype(np.float32))):.6f}")


def test_rms_norm_ktir_zeros():
    """RMSNorm of all-zeros input should produce all zeros."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    X = np.zeros((NUM_ROWS, N_COLS), dtype=np.float16)
    W = np.ones(N_COLS, dtype=np.float16)
    Y = np.zeros((NUM_ROWS, N_COLS), dtype=np.float16)

    outputs = interp.execute_function(
        "rms_norm_fwd",
        X=X, W=W, Y=Y,
        N=N_COLS, eps=EPS, BLOCK_SIZE=BLOCK_SIZE,
    )
    result_Y = outputs["Y"]
    # With all zeros, rms = sqrt(0 + eps), so y = 0 * inv_rms * w = 0
    np.testing.assert_allclose(result_Y, np.zeros_like(result_Y), atol=1e-3)
    print("PASS: zeros input")


if __name__ == "__main__":
    test_rms_norm_ktir()
    test_rms_norm_ktir_zeros()
    print("\nAll RMSNorm KTIR validation tests passed!")
