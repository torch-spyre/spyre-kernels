"""Validate Ranks KTDP MLIR against NumPy reference using ktir_cpu interpreter."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

NUM_ROWS = 32
VOCAB_SIZE = 4096
BLOCK_SIZE = 1024


def ranks_ref(logits: np.ndarray, ref_logits: np.ndarray) -> np.ndarray:
    """NumPy reference: count how many logits >= ref_logit per row."""
    counts = np.zeros(NUM_ROWS, dtype=np.float16)
    for i in range(NUM_ROWS):
        counts[i] = np.sum(logits[i] >= ref_logits[i]).astype(np.float16)
    return counts


def test_ranks_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    logits = rng.standard_normal((NUM_ROWS, VOCAB_SIZE)).astype(np.float16)
    token_ids = rng.integers(0, VOCAB_SIZE, size=NUM_ROWS)
    ref_logits = np.array([logits[i, token_ids[i]] for i in range(NUM_ROWS)], dtype=np.float16)
    output = np.zeros(NUM_ROWS, dtype=np.float16)

    outputs = interp.execute_function(
        "ranks_kernel",
        logits=logits,
        ref_logits=ref_logits,
        output=output,
        vocab_size=VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result = outputs["output"]
    expected = ranks_ref(logits, ref_logits)

    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1.0)
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.1f}")


def test_ranks_all_same():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    logits = np.ones((NUM_ROWS, VOCAB_SIZE), dtype=np.float16)
    ref_logits = np.ones(NUM_ROWS, dtype=np.float16)
    output = np.zeros(NUM_ROWS, dtype=np.float16)

    outputs = interp.execute_function(
        "ranks_kernel",
        logits=logits,
        ref_logits=ref_logits,
        output=output,
        vocab_size=VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result = outputs["output"]
    expected = np.full(NUM_ROWS, VOCAB_SIZE, dtype=np.float16)
    np.testing.assert_allclose(result, expected, atol=1.0)
    print("PASS: all-same input (expect all vocab_size)")


if __name__ == "__main__":
    test_ranks_ktir()
    test_ranks_all_same()
    print("\nAll Ranks KTIR validation tests passed!")
