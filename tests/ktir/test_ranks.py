"""Validate Ranks KTDP MLIR against vLLM kernel using ktir_cpu interpreter."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.vllm.ranks.wrapper import ranks

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "vllm" / "ranks" / "kernel.ktir")

NUM_ROWS = 32
VOCAB_SIZE = 4096
BLOCK_SIZE = 1024


def vllm_reference(logits_np: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Run vLLM ranks kernel on GPU and return result as numpy f16."""
    logits = torch.from_numpy(logits_np.astype(np.float16)).cuda()
    tids = torch.from_numpy(token_ids.astype(np.int64)).cuda()
    out = ranks(logits, tids)
    return out.cpu().numpy().astype(np.float16)


def test_ranks_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    logits = rng.standard_normal((NUM_ROWS, VOCAB_SIZE)).astype(np.float16)
    token_ids = rng.integers(0, VOCAB_SIZE, size=NUM_ROWS).astype(np.int64)
    output = np.zeros(NUM_ROWS, dtype=np.float16)

    outputs = interp.execute_function(
        "ranks_kernel",
        logits=logits,
        token_ids=token_ids,
        output=output,
        vocab_size=VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result = outputs["output"]
    expected = vllm_reference(logits, token_ids)

    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1.0)
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.1f}")


def test_ranks_all_same():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    logits = np.ones((NUM_ROWS, VOCAB_SIZE), dtype=np.float16)
    token_ids = np.zeros(NUM_ROWS, dtype=np.int64)
    output = np.zeros(NUM_ROWS, dtype=np.float16)

    outputs = interp.execute_function(
        "ranks_kernel",
        logits=logits,
        token_ids=token_ids,
        output=output,
        vocab_size=VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    result = outputs["output"]
    expected = np.full(NUM_ROWS, VOCAB_SIZE, dtype=np.float16)
    np.testing.assert_allclose(result, expected, atol=1.0)
    print("PASS: all-same input (expect all vocab_size)")


