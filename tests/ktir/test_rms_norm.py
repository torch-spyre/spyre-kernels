# SPDX-License-Identifier: Apache-2.0
"""Validate RMSNorm KTDP MLIR against vLLM kernel using ktir_cpu interpreter.

Run from the ktir_cpu venv:
    cd external/ktir_cpu && uv run python ../../tests/ktir/test_rms_norm.py
"""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.rms_norm.wrapper import rms_norm

MLIR_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "kernels" / "rms_norm" / "kernel.ktir.mlir"
)

NUM_ROWS = 32
N_COLS = 4096
BLOCK_SIZE = 1024
EPS = np.float16(1e-5)


def vllm_reference(x_np: np.ndarray, w_np: np.ndarray, eps: float) -> np.ndarray:
    """Run vLLM RMSNorm kernel on GPU and return result as numpy."""
    x = torch.from_numpy(x_np.astype(np.float16)).cuda()
    w = torch.from_numpy(w_np.astype(np.float16)).cuda()
    out = rms_norm(x, w, eps=float(eps))
    return out.cpu().numpy()


def test_rms_norm_ktir():
    """Run RMSNorm KTDP kernel on 32 cores and compare against vLLM kernel."""
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
    expected_Y = vllm_reference(X, W, EPS)

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
    np.testing.assert_allclose(result_Y, np.zeros_like(result_Y), atol=1e-3)
    print("PASS: zeros input")


