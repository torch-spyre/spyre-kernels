"""Validate Decode softmax+reduceV KTDP MLIR against NumPy reference."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ktir_cpu"))
from ktir_cpu import KTIRInterpreter

MLIR_PATH = str(Path(__file__).resolve().parent / "kernel.ktir.mlir")

NUM_BATCHES = 4
NUM_HEADS = 8
NUM_SPLITS = 4
LV = 64


def decode_softmax_reducev_ref(mid_o: np.ndarray) -> tuple:
    """NumPy reference for online softmax merge across KV splits.

    mid_o: [128, 65] — for each (batch, head), 4 rows of [V(64) | logit(1)]
    Returns: output [32, 64], lse [32]
    """
    total_pairs = NUM_BATCHES * NUM_HEADS
    output = np.zeros((total_pairs, LV), dtype=np.float32)
    lse = np.zeros(total_pairs, dtype=np.float32)

    for pair_id in range(total_pairs):
        base_row = pair_id * NUM_SPLITS
        e_max = -np.inf
        e_sum = 0.0
        acc = np.zeros(LV, dtype=np.float32)

        for split in range(NUM_SPLITS):
            row = base_row + split
            v = mid_o[row, :LV].astype(np.float32)
            tlogic = float(mid_o[row, LV])

            n_e_max = max(tlogic, e_max)
            old_scale = np.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = np.exp(tlogic - n_e_max)
            acc += exp_logic * v
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

        output[pair_id] = acc / e_sum
        lse[pair_id] = e_max + np.log(e_sum)

    return output.astype(np.float16), lse.astype(np.float16)


def test_decode_softmax_reducev_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    mid_o = rng.standard_normal((NUM_BATCHES * NUM_HEADS * NUM_SPLITS, LV + 1)).astype(np.float16)
    output = np.zeros((NUM_BATCHES * NUM_HEADS, LV), dtype=np.float16)
    lse_out = np.zeros(NUM_BATCHES * NUM_HEADS, dtype=np.float16)

    outputs = interp.execute_function(
        "decode_softmax_reducev_kernel",
        mid_o=mid_o,
        lse_out=lse_out,
        output=output,
        num_splits=NUM_SPLITS,
    )
    result_out = outputs["output"]
    result_lse = outputs["lse_out"]

    expected_out, expected_lse = decode_softmax_reducev_ref(mid_o)

    np.testing.assert_allclose(
        result_out.astype(np.float32), expected_out.astype(np.float32),
        rtol=5e-2, atol=2.0,
    )
    max_err_out = np.max(np.abs(result_out.astype(np.float32) - expected_out.astype(np.float32)))

    np.testing.assert_allclose(
        result_lse.astype(np.float32), expected_lse.astype(np.float32),
        rtol=5e-2, atol=0.1,
    )
    max_err_lse = np.max(np.abs(result_lse.astype(np.float32) - expected_lse.astype(np.float32)))

    print(f"PASS: output max abs error = {max_err_out:.6f}, lse max abs error = {max_err_lse:.6f}")


def test_decode_softmax_uniform():
    """All logits equal → uniform attention → output = mean(V), lse = logit + log(splits)."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    total_rows = NUM_BATCHES * NUM_HEADS * NUM_SPLITS
    mid_o = np.zeros((total_rows, LV + 1), dtype=np.float16)
    rng = np.random.default_rng(7)
    for pair_id in range(NUM_BATCHES * NUM_HEADS):
        base = pair_id * NUM_SPLITS
        for s in range(NUM_SPLITS):
            mid_o[base + s, :LV] = rng.standard_normal(LV).astype(np.float16)
            mid_o[base + s, LV] = np.float16(1.0)  # all logits = 1.0

    output = np.zeros((NUM_BATCHES * NUM_HEADS, LV), dtype=np.float16)
    lse_out = np.zeros(NUM_BATCHES * NUM_HEADS, dtype=np.float16)

    outputs = interp.execute_function(
        "decode_softmax_reducev_kernel",
        mid_o=mid_o,
        lse_out=lse_out,
        output=output,
        num_splits=NUM_SPLITS,
    )
    result_out = outputs["output"]
    result_lse = outputs["lse_out"]
    expected_out, expected_lse = decode_softmax_reducev_ref(mid_o)

    np.testing.assert_allclose(
        result_out.astype(np.float32), expected_out.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    print("PASS: uniform logits")


if __name__ == "__main__":
    test_decode_softmax_reducev_ktir()
    test_decode_softmax_uniform()
    print("\nAll Decode softmax+reduceV KTIR validation tests passed!")
