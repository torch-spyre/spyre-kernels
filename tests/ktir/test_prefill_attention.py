"""Validate Prefill Attention (SDPA) KTDP MLIR against vLLM kernel."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.prefill_attention.wrapper import context_attention_fwd

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "prefill_attention" / "kernel.ktir.mlir")

SEQ_LEN = 16
NUM_HEADS = 4
HEAD_DIM = 64
FLAT_DIM = NUM_HEADS * HEAD_DIM
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def vllm_reference(
    q_flat: np.ndarray, k_flat: np.ndarray, v_flat: np.ndarray,
    is_causal: bool = True,
) -> np.ndarray:
    """Run vLLM prefill attention kernel on GPU.

    KTIR layout: q, k, v [T, H*D]
    Wrapper layout: q, k, v [T, H, D]
    Returns: output [T, H*D] (KTIR layout)
    """
    q = torch.from_numpy(q_flat.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(np.float16)).cuda()
    k = torch.from_numpy(k_flat.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(np.float16)).cuda()
    v = torch.from_numpy(v_flat.reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM).astype(np.float16)).cuda()
    o = torch.zeros_like(q)

    b_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)
    b_seq_len = torch.tensor([SEQ_LEN], device="cuda", dtype=torch.int32)

    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, SEQ_LEN, is_causal=is_causal)
    return o.reshape(SEQ_LEN, FLAT_DIM).cpu().numpy().astype(np.float16)




def test_prefill_attention_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    q = (rng.standard_normal((SEQ_LEN, FLAT_DIM)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ_LEN, FLAT_DIM)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ_LEN, FLAT_DIM)) * 0.1).astype(np.float16)
    output = np.zeros((SEQ_LEN, FLAT_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "prefill_attention_kernel",
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        output_ptr=output,
        num_heads=NUM_HEADS,
    )
    result = outputs["output_ptr"]
    expected = vllm_reference(q, k, v, is_causal=True)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-1, atol=5e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_prefill_attention_uniform():
    """Uniform Q,K with causal mask -> row i attends to positions 0..i."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    q = np.ones((SEQ_LEN, FLAT_DIM), dtype=np.float16) * 0.01
    k = np.ones((SEQ_LEN, FLAT_DIM), dtype=np.float16) * 0.01
    rng = np.random.default_rng(7)
    v = rng.standard_normal((SEQ_LEN, FLAT_DIM)).astype(np.float16)
    output = np.zeros((SEQ_LEN, FLAT_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "prefill_attention_kernel",
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        output_ptr=output,
        num_heads=NUM_HEADS,
    )
    result = outputs["output_ptr"]
    expected = vllm_reference(q, k, v, is_causal=True)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-1, atol=5e-2,
    )
    print("PASS: uniform Q,K with causal mask")


