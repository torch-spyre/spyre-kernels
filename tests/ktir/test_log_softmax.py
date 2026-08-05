"""Validate Log-softmax KTDP MLIR against vLLM kernel using ktir_cpu interpreter."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.vllm.log_softmax.wrapper import topk_log_softmax

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "vllm" / "log_softmax" / "kernel.ktir")

NUM_ROWS = 32
VOCAB_SIZE = 4096
TOPK = 8
BLOCK_SIZE = 1024


def vllm_reference(logits_np: np.ndarray, topk_ids: np.ndarray) -> np.ndarray:
    """Run vLLM topk_log_softmax kernel on GPU and return result as numpy f16."""
    logits = torch.from_numpy(logits_np.astype(np.float16)).cuda()
    ids = torch.from_numpy(topk_ids.astype(np.int64)).cuda()
    out = topk_log_softmax(logits, ids, TOPK)
    return out.cpu().numpy().astype(np.float16)


def test_log_softmax_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    logits = rng.standard_normal((NUM_ROWS, VOCAB_SIZE)).astype(np.float16)
    topk_ids = np.array([
        rng.choice(VOCAB_SIZE, size=TOPK, replace=False) for _ in range(NUM_ROWS)
    ], dtype=np.int64)
    output = np.zeros((NUM_ROWS, TOPK), dtype=np.float16)

    outputs = interp.execute_function(
        "log_softmax_kernel",
        logits=logits,
        topk_ids=topk_ids.astype(np.int64),
        output=output,
        vocab_size=VOCAB_SIZE,
        topk=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result = outputs["output"]
    expected = vllm_reference(logits, topk_ids)

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
    topk_ids = np.tile(np.arange(TOPK, dtype=np.int64), (NUM_ROWS, 1))
    output = np.zeros((NUM_ROWS, TOPK), dtype=np.float16)

    outputs = interp.execute_function(
        "log_softmax_kernel",
        logits=logits,
        topk_ids=topk_ids,
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


