"""Validate Prefill Attention (SDPA) KTDP MLIR against NumPy reference."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

SEQ_LEN = 16
NUM_HEADS = 4
HEAD_DIM = 64
FLAT_DIM = NUM_HEADS * HEAD_DIM
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def sdpa_ref(q: np.ndarray, k: np.ndarray, v: np.ndarray, is_causal: bool = True) -> np.ndarray:
    """NumPy reference for multi-head SDPA.

    q, k, v: [16, 256]  (seq_len x num_heads*head_dim)
    Returns: output [16, 256]
    """
    output = np.zeros_like(q, dtype=np.float32)
    for h in range(NUM_HEADS):
        col = h * HEAD_DIM
        qh = q[:, col:col + HEAD_DIM].astype(np.float32)
        kh = k[:, col:col + HEAD_DIM].astype(np.float32)
        vh = v[:, col:col + HEAD_DIM].astype(np.float32)

        scores = qh @ kh.T * SCALE
        if is_causal:
            causal_mask = np.triu(np.full((SEQ_LEN, SEQ_LEN), -10000.0), k=1)
            scores = scores + causal_mask
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        output[:, col:col + HEAD_DIM] = weights @ vh

    return output.astype(np.float16)


def _make_causal_mask(seq_len):
    """Create causal mask: 0 on lower triangle (including diagonal), -1e8 on upper."""
    mask = np.zeros((seq_len, seq_len), dtype=np.float16)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i, j] = np.float16(-10000.0)  # -1e4 fits in f16 range
    return mask


def test_prefill_attention_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    q = (rng.standard_normal((SEQ_LEN, FLAT_DIM)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ_LEN, FLAT_DIM)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ_LEN, FLAT_DIM)) * 0.1).astype(np.float16)
    output = np.zeros((SEQ_LEN, FLAT_DIM), dtype=np.float16)
    causal_mask = _make_causal_mask(SEQ_LEN)

    outputs = interp.execute_function(
        "prefill_attention_kernel",
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        output_ptr=output,
        causal_mask_ptr=causal_mask,
        num_heads=NUM_HEADS,
    )
    result = outputs["output_ptr"]
    expected = sdpa_ref(q, k, v, is_causal=True)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-1, atol=5e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_prefill_attention_uniform():
    """Uniform Q,K with causal mask → row i attends to positions 0..i."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    q = np.ones((SEQ_LEN, FLAT_DIM), dtype=np.float16) * 0.01
    k = np.ones((SEQ_LEN, FLAT_DIM), dtype=np.float16) * 0.01
    rng = np.random.default_rng(7)
    v = rng.standard_normal((SEQ_LEN, FLAT_DIM)).astype(np.float16)
    output = np.zeros((SEQ_LEN, FLAT_DIM), dtype=np.float16)
    causal_mask = _make_causal_mask(SEQ_LEN)

    outputs = interp.execute_function(
        "prefill_attention_kernel",
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        output_ptr=output,
        causal_mask_ptr=causal_mask,
        num_heads=NUM_HEADS,
    )
    result = outputs["output_ptr"]
    expected = sdpa_ref(q, k, v, is_causal=True)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-1, atol=5e-2,
    )
    print("PASS: uniform Q,K with causal mask")


if __name__ == "__main__":
    test_prefill_attention_ktir()
    test_prefill_attention_uniform()
    print("\nAll Prefill Attention KTIR validation tests passed!")
