"""Validate Merge Attention States KTDP MLIR against vLLM kernel."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.merge_attn_states.wrapper import merge_attn_states

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "merge_attn_states" / "kernel.ktir.mlir")

NUM_TOKENS = 32
NUM_HEADS = 8
HEAD_DIM = 64
FLAT_DIM = NUM_HEADS * HEAD_DIM


def vllm_reference(
    prefix_output_flat: np.ndarray,
    suffix_output_flat: np.ndarray,
    prefix_lse_ht: np.ndarray,
    suffix_lse_ht: np.ndarray,
) -> np.ndarray:
    """Run vLLM merge_attn_states kernel on GPU.

    KTIR layout: output [T, H*D], lse [H, T]
    Wrapper layout: output [T, H, D], lse [T, H]
    Returns: merged output [T, H*D] (KTIR layout)
    """
    p_out = torch.from_numpy(
        prefix_output_flat.reshape(NUM_TOKENS, NUM_HEADS, HEAD_DIM).astype(np.float16)
    ).cuda()
    s_out = torch.from_numpy(
        suffix_output_flat.reshape(NUM_TOKENS, NUM_HEADS, HEAD_DIM).astype(np.float16)
    ).cuda()
    p_lse = torch.from_numpy(prefix_lse_ht.T.astype(np.float32)).cuda()
    s_lse = torch.from_numpy(suffix_lse_ht.T.astype(np.float32)).cuda()

    out = merge_attn_states(p_out, p_lse, s_out, s_lse)
    return out.reshape(NUM_TOKENS, FLAT_DIM).cpu().numpy().astype(np.float16)


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
    expected = vllm_reference(prefix_output, suffix_output, prefix_lse, suffix_lse)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


def test_merge_equal_lse():
    """Equal LSEs -> output is average of prefix and suffix."""
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
    print("PASS: equal LSEs -> average")


def test_merge_inf_prefix_lse():
    """Exercise the FA2 +inf LSE edge case.

    Flash Attention 2 returns LSE = +inf when a head attends to zero KV
    tokens. This happens in chunked prefill when the prefix chunk is empty
    for a request but the suffix chunk is not.

    Example: batch has 2 requests, request B's prefix chunk is empty:
      prefix_lse[head, token_B] = +inf   (no prefix KV)
      suffix_lse[head, token_B] = 1.5    (normal suffix)

    With the inf→-inf handling (replace +inf with -inf before the merge):
      max_lse = max(-inf, 1.5) = 1.5
      p_se = exp(-inf - 1.5) = 0           → prefix zeroed out
      s_se = exp(1.5 - 1.5) = 1            → suffix passes through
      output = suffix_output               ✓
    """
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(99)
    prefix_output = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    suffix_output = rng.standard_normal((NUM_TOKENS, FLAT_DIM)).astype(np.float16)
    prefix_lse = np.full((NUM_HEADS, NUM_TOKENS), np.inf, dtype=np.float16)
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
    expected = vllm_reference(prefix_output, suffix_output, prefix_lse, suffix_lse)

    # result[0, :8]: [-0.3381 -0.3994 -0.9243 -1.348   0.1267 -0.3818  1.302   1.088 ]
    # suffix[0, :8]: [-0.3381 -0.3994 -0.9243 -1.348   0.1267 -0.3818  1.302   1.088 ]
    # prefix[0, :8]: [ 0.0825 -0.4644  0.0505  0.686  -1.757   1.685  -0.4578 -0.596 ]
    # exact match with suffix: True
    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    print("PASS: +inf prefix LSE -> output = suffix")


