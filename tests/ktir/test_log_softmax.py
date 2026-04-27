"""Validate Log-softmax KTDP MLIR against NumPy reference using ktir_cpu interpreter."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

NUM_ROWS = 32
VOCAB_SIZE = 4096
TOPK = 8
BLOCK_SIZE = 1024


def log_softmax_ref(logits: np.ndarray, topk_logits: np.ndarray) -> np.ndarray:
    """NumPy reference: log-softmax at pre-extracted top-k positions."""
    output = np.zeros((NUM_ROWS, TOPK), dtype=np.float16)
    for i in range(NUM_ROWS):
        row = logits[i].astype(np.float32)
        max_val = np.max(row)
        lse = np.log(np.sum(np.exp(row - max_val)))
        for k in range(TOPK):
            output[i, k] = np.float16(float(topk_logits[i, k]) - max_val - lse)
    return output


def test_log_softmax_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    logits = rng.standard_normal((NUM_ROWS, VOCAB_SIZE)).astype(np.float16)
    topk_ids = np.array([
        rng.choice(VOCAB_SIZE, size=TOPK, replace=False) for _ in range(NUM_ROWS)
    ])
    topk_logits = np.array([
        [logits[i, topk_ids[i, k]] for k in range(TOPK)] for i in range(NUM_ROWS)
    ], dtype=np.float16)
    output = np.zeros((NUM_ROWS, TOPK), dtype=np.float16)

    outputs = interp.execute_function(
        "log_softmax_kernel",
        logits=logits,
        topk_logits=topk_logits,
        output=output,
        vocab_size=VOCAB_SIZE,
        topk=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result = outputs["output"]
    expected = log_softmax_ref(logits, topk_logits)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_log_softmax_uniform():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    logits = np.ones((NUM_ROWS, VOCAB_SIZE), dtype=np.float16)
    topk_logits = np.ones((NUM_ROWS, TOPK), dtype=np.float16)
    output = np.zeros((NUM_ROWS, TOPK), dtype=np.float16)

    outputs = interp.execute_function(
        "log_softmax_kernel",
        logits=logits,
        topk_logits=topk_logits,
        output=output,
        vocab_size=VOCAB_SIZE,
        topk=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result = outputs["output"]
    expected_val = -np.log(VOCAB_SIZE)
    np.testing.assert_allclose(
        result.astype(np.float32),
        np.full((NUM_ROWS, TOPK), expected_val, dtype=np.float32),
        rtol=5e-2, atol=5e-2,
    )
    print(f"PASS: uniform input (expected ~{expected_val:.4f})")


if __name__ == "__main__":
    test_log_softmax_ktir()
    test_log_softmax_uniform()
    print("\nAll Log-softmax KTIR validation tests passed!")
