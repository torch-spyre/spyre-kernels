"""Validate MRoPE KTDP MLIR against NumPy reference."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

NUM_TOKENS = 32
NUM_HEADS = 8
HEAD_DIM = 64
HALF_DIM = HEAD_DIM // 2


def mrope_ref(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple:
    """NumPy reference for rotary embedding on Q and K.

    q, k: [32, 512]  (num_tokens x num_heads*head_dim)
    cos, sin: [32, 32]  (num_tokens x head_dim/2)
    Returns: (q_result, k_result) with rotary embedding applied.
    """
    q_result = q.copy().astype(np.float32)
    k_result = k.copy().astype(np.float32)
    cos_f32 = cos.astype(np.float32)
    sin_f32 = sin.astype(np.float32)

    for t in range(NUM_TOKENS):
        for h in range(NUM_HEADS):
            col = h * HEAD_DIM
            # Q
            x1 = q_result[t, col:col + HALF_DIM].copy()
            x2 = q_result[t, col + HALF_DIM:col + HEAD_DIM].copy()
            q_result[t, col:col + HALF_DIM] = x1 * cos_f32[t] - x2 * sin_f32[t]
            q_result[t, col + HALF_DIM:col + HEAD_DIM] = x2 * cos_f32[t] + x1 * sin_f32[t]
            # K
            k1 = k_result[t, col:col + HALF_DIM].copy()
            k2 = k_result[t, col + HALF_DIM:col + HEAD_DIM].copy()
            k_result[t, col:col + HALF_DIM] = k1 * cos_f32[t] - k2 * sin_f32[t]
            k_result[t, col + HALF_DIM:col + HEAD_DIM] = k2 * cos_f32[t] + k1 * sin_f32[t]

    return q_result.astype(np.float16), k_result.astype(np.float16)


def test_mrope_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    q = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    k = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    cos = rng.uniform(-1, 1, (NUM_TOKENS, HALF_DIM)).astype(np.float16)
    sin = rng.uniform(-1, 1, (NUM_TOKENS, HALF_DIM)).astype(np.float16)

    outputs = interp.execute_function(
        "mrope_kernel",
        q=q.copy(),
        k=k.copy(),
        cos_ptr=cos,
        sin_ptr=sin,
        num_q_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
    )
    result_q = outputs["q"]
    result_k = outputs["k"]
    expected_q, expected_k = mrope_ref(q, k, cos, sin)

    np.testing.assert_allclose(
        result_q.astype(np.float32), expected_q.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    np.testing.assert_allclose(
        result_k.astype(np.float32), expected_k.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    max_err_q = np.max(np.abs(result_q.astype(np.float32) - expected_q.astype(np.float32)))
    max_err_k = np.max(np.abs(result_k.astype(np.float32) - expected_k.astype(np.float32)))
    print(f"PASS: Q max abs error = {max_err_q:.6f}, K max abs error = {max_err_k:.6f}")


def test_mrope_identity():
    """cos=1, sin=0 → output should equal input."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(7)
    q = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    k = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    cos = np.ones((NUM_TOKENS, HALF_DIM), dtype=np.float16)
    sin = np.zeros((NUM_TOKENS, HALF_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "mrope_kernel",
        q=q.copy(),
        k=k.copy(),
        cos_ptr=cos,
        sin_ptr=sin,
        num_q_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
    )

    np.testing.assert_allclose(
        outputs["q"].astype(np.float32), q.astype(np.float32),
        rtol=1e-3, atol=1e-3,
    )
    np.testing.assert_allclose(
        outputs["k"].astype(np.float32), k.astype(np.float32),
        rtol=1e-3, atol=1e-3,
    )
    print("PASS: identity (cos=1, sin=0)")


if __name__ == "__main__":
    test_mrope_ktir()
    test_mrope_identity()
    print("\nAll MRoPE KTIR validation tests passed!")
