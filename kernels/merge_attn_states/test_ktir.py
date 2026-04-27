"""Validate Merge Attention States KTDP MLIR against NumPy reference."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

NUM_TOKENS = 32
NUM_HEADS = 8
HEAD_DIM = 64
FLAT_DIM = NUM_HEADS * HEAD_DIM


def merge_attn_states_ref(
    prefix_output: np.ndarray,
    suffix_output: np.ndarray,
    prefix_lse: np.ndarray,
    suffix_lse: np.ndarray,
) -> np.ndarray:
    """NumPy reference for merging two partial attention outputs.

    prefix/suffix_output: [32, 512]
    prefix/suffix_lse: [8, 32]
    Returns: merged output [32, 512]
    """
    output = np.zeros_like(prefix_output, dtype=np.float32)
    for t in range(NUM_TOKENS):
        for h in range(NUM_HEADS):
            p_lse = float(prefix_lse[h, t])
            s_lse = float(suffix_lse[h, t])
            max_lse = max(p_lse, s_lse)
            p_se = np.exp(p_lse - max_lse)
            s_se = np.exp(s_lse - max_lse)
            out_se = p_se + s_se
            p_scale = p_se / out_se
            s_scale = s_se / out_se
            col = h * HEAD_DIM
            p_out = prefix_output[t, col:col + HEAD_DIM].astype(np.float32)
            s_out = suffix_output[t, col:col + HEAD_DIM].astype(np.float32)
            output[t, col:col + HEAD_DIM] = p_out * p_scale + s_out * s_scale
    return output.astype(np.float16)


def test_merge_attn_states_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    prefix_output = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    suffix_output = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    prefix_lse = rng.uniform(-2, 2, (NUM_HEADS, NUM_TOKENS)).astype(np.float16)
    suffix_lse = rng.uniform(-2, 2, (NUM_HEADS, NUM_TOKENS)).astype(np.float16)
    output = np.zeros((NUM_TOKENS, FLAT_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "merge_attn_states_kernel",
        prefix_output=prefix_output,
        suffix_output=suffix_output,
        prefix_lse=prefix_lse,
        suffix_lse=suffix_lse,
        output=output,
        num_heads=NUM_HEADS,
    )
    result = outputs["output"]
    expected = merge_attn_states_ref(prefix_output, suffix_output, prefix_lse, suffix_lse)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_merge_equal_lse():
    """Equal LSEs → output is average of prefix and suffix."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(7)
    prefix_output = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    suffix_output = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    lse = np.ones((NUM_HEADS, NUM_TOKENS), dtype=np.float16)
    output = np.zeros((NUM_TOKENS, FLAT_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "merge_attn_states_kernel",
        prefix_output=prefix_output,
        suffix_output=suffix_output,
        prefix_lse=lse.copy(),
        suffix_lse=lse.copy(),
        output=output,
        num_heads=NUM_HEADS,
    )
    result = outputs["output"]
    expected = ((prefix_output.astype(np.float32) + suffix_output.astype(np.float32)) / 2).astype(np.float16)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    print("PASS: equal LSEs → average")


if __name__ == "__main__":
    test_merge_attn_states_ktir()
    test_merge_equal_lse()
    print("\nAll Merge Attention States KTIR validation tests passed!")
