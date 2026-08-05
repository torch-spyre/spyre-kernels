"""Validate Embedding KTDP MLIR against liger-kernel Triton kernel using ktir_cpu interpreter."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.vllm.embedding.wrapper import embedding

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "vllm" / "embedding" / "kernel.ktir")

N_TOKENS = 32
VOCAB_SIZE = 4096
EMBEDDING_DIM = 1024


def triton_reference(table_np: np.ndarray, indices_np: np.ndarray) -> np.ndarray:
    """Run embedding kernel on GPU and return result as numpy f16."""
    table = torch.from_numpy(table_np.astype(np.float16)).cuda()
    idx = torch.from_numpy(indices_np.astype(np.int64)).cuda()
    out = embedding(table, idx)
    return out.cpu().numpy().astype(np.float16)


def test_embedding_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    table = rng.standard_normal((VOCAB_SIZE, EMBEDDING_DIM)).astype(np.float16)
    indices = rng.integers(0, VOCAB_SIZE, size=N_TOKENS).astype(np.int64)
    output = np.zeros((N_TOKENS, EMBEDDING_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "embedding_kernel",
        embeddings=table,
        indices=indices,
        output=output,
        n_elements=N_TOKENS,
        embedding_dim=EMBEDDING_DIM,
    )
    result = outputs["output"]
    expected = triton_reference(table, indices)

    np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    print("PASS: embedding KTIR matches Triton reference exactly")


def test_embedding_ktir_repeated_indices():
    """Same index repeated — all rows should be identical copies of that row."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(7)
    table = rng.standard_normal((VOCAB_SIZE, EMBEDDING_DIM)).astype(np.float16)
    indices = np.full(N_TOKENS, 42, dtype=np.int64)
    output = np.zeros((N_TOKENS, EMBEDDING_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "embedding_kernel",
        embeddings=table,
        indices=indices,
        output=output,
        n_elements=N_TOKENS,
        embedding_dim=EMBEDDING_DIM,
    )
    result = outputs["output"]

    expected = np.broadcast_to(table[42], (N_TOKENS, EMBEDDING_DIM))
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    print("PASS: repeated indices produce identical rows")
