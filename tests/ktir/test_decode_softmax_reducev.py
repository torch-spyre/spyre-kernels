"""Validate Decode softmax+reduceV KTDP MLIR against vLLM kernel."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.decode_softmax_reducev.wrapper import decode_softmax_reducev

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "decode_softmax_reducev" / "kernel.ktir")

NUM_BATCHES = 4
NUM_HEADS = 8
NUM_SPLITS = 4
LV = 64


def vllm_reference(mid_o_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run vLLM decode stage2 kernel on GPU.

    mid_o_flat: [B*H*S, Lv+1] (KTIR layout)
    Returns: output [B*H, Lv] f16, lse [B*H] f16
    """
    mid_o_4d = mid_o_flat.reshape(NUM_BATCHES, NUM_HEADS, NUM_SPLITS, LV + 1)
    mid_o = torch.from_numpy(mid_o_4d.astype(np.float16)).cuda()
    b_seq_len = torch.full((NUM_BATCHES,), NUM_SPLITS * 64, device="cuda", dtype=torch.int32)
    o, lse = decode_softmax_reducev(mid_o, b_seq_len, NUM_SPLITS)
    o_flat = o.reshape(NUM_BATCHES * NUM_HEADS, LV).cpu().numpy().astype(np.float16)
    lse_flat = lse.reshape(NUM_BATCHES * NUM_HEADS).cpu().numpy().astype(np.float16)
    return o_flat, lse_flat


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

    expected_out, expected_lse = vllm_reference(mid_o)

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
    """All logits equal -> uniform attention -> output = mean(V), lse = logit + log(splits)."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    total_rows = NUM_BATCHES * NUM_HEADS * NUM_SPLITS
    mid_o = np.zeros((total_rows, LV + 1), dtype=np.float16)
    rng = np.random.default_rng(7)
    for pair_id in range(NUM_BATCHES * NUM_HEADS):
        base = pair_id * NUM_SPLITS
        for s in range(NUM_SPLITS):
            mid_o[base + s, :LV] = rng.standard_normal(LV).astype(np.float16)
            mid_o[base + s, LV] = np.float16(1.0)

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
    expected_out, expected_lse = vllm_reference(mid_o)

    np.testing.assert_allclose(
        result_out.astype(np.float32), expected_out.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    print("PASS: uniform logits")


