# SPDX-License-Identifier: Apache-2.0
"""Validate the generated RMSNorm KTIR against a NumPy reference.

The KTIR under test is *generated* from the Triton kernel source by
``scripts/gen_ktir.py`` (driver: ``kernels/rms_norm/lower.py``), so the
function name and signature mirror the lowered kernel exactly:

    func.func @_rms_norm_kernel_td(
        %arg0: index,  // input_ptr   [n_rows, n_cols] f16
        %arg1: index,  // weight_ptr  [1, n_cols]      f16
        %arg2: index,  // output_ptr  [n_rows, n_cols] f16
        %arg3: i32,    // n_rows
        %arg4: i32,    // n_cols
        %arg5: f16,    // eps
    )

``BLOCK_SIZE`` and ``ROWS_PER_PROGRAM`` are constexprs baked into the
KTIR (see lower.py), so they are not runtime args. The reference is a
NumPy RMSNorm (f32 accumulation, matching the kernel), which keeps this
test runnable with only ktir_cpu installed — no GPU / Spyre-Triton build
needed.

Run:
    .venv/bin/python -m pytest tests/ktir/test_rms_norm.py -v
"""

from pathlib import Path

import numpy as np

from ktir_cpu import KTIRInterpreter

MLIR_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "kernels" / "rms_norm" / "tensor_descriptor.ktir"
)

# Concrete shapes baked into the generated KTIR (32-core grid, one row each).
NUM_ROWS = 32
N_COLS = 4096
EPS = np.float16(1e-5)

FUNC = "_rms_norm_kernel_td"


def numpy_reference(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm in f32, matching the kernel's upcast-accumulate-downcast.

    y = x / sqrt(mean(x^2, axis=-1) + eps) * weight
    """
    xf = x.astype(np.float32)
    inv_rms = 1.0 / np.sqrt((xf * xf).mean(axis=1, keepdims=True) + eps)
    out = xf * inv_rms * w.astype(np.float32)
    return out.astype(np.float16)


def _run(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH.read_text())

    Y = np.zeros((NUM_ROWS, N_COLS), dtype=np.float16)
    outputs = interp.execute_function(
        FUNC,
        arg0=X,                      # input  [n_rows, n_cols]
        arg1=W.reshape(1, N_COLS),   # weight [1, n_cols]
        arg2=Y,                      # output [n_rows, n_cols]
        arg3=np.int32(NUM_ROWS),
        arg4=np.int32(N_COLS),
        arg5=EPS,
    )
    return outputs["arg2"]


def test_rms_norm_ktir():
    """Generated RMSNorm KTIR matches the NumPy reference within f16 tol."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((NUM_ROWS, N_COLS)).astype(np.float16)
    W = rng.standard_normal(N_COLS).astype(np.float16)

    result = _run(X, W)
    expected = numpy_reference(X, W, float(EPS))

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-2, atol=1e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_rms_norm_ktir_zeros():
    """RMSNorm of all-zeros input should produce all zeros."""
    X = np.zeros((NUM_ROWS, N_COLS), dtype=np.float16)
    W = np.ones(N_COLS, dtype=np.float16)

    result = _run(X, W)
    np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-3)
    print("PASS: zeros input")
