"""Validate MRoPE KTDP MLIR against vLLM kernel."""

from pathlib import Path

import numpy as np
import torch

from ktir_cpu import KTIRInterpreter
from kernels.mrope.wrapper import triton_mrope

MLIR_PATH = str(Path(__file__).resolve().parent.parent.parent / "kernels" / "mrope" / "kernel.ktir")

NUM_TOKENS = 32
NUM_HEADS = 8
HEAD_DIM = 64
HALF_DIM = HEAD_DIM // 2


def vllm_reference(
    q_flat: np.ndarray, k_flat: np.ndarray,
    cos_2d: np.ndarray, sin_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run vLLM MRoPE kernel on GPU.

    KTIR layout: q, k [T, H*D], cos/sin [T, half_dim] (pre-merged)
    Wrapper layout: q, k [T, H*D], cos/sin [3, T, half_dim] (3 sections)

    We use mrope_section=[half_dim, 0, 0] so the t-section covers everything,
    making the vLLM kernel equivalent to the KTIR simplified version.
    """
    q = torch.from_numpy(q_flat.copy().astype(np.float16)).cuda()
    k = torch.from_numpy(k_flat.copy().astype(np.float16)).cuda()

    cos_3d = np.zeros((3, NUM_TOKENS, HALF_DIM), dtype=np.float16)
    sin_3d = np.zeros((3, NUM_TOKENS, HALF_DIM), dtype=np.float16)
    cos_3d[0] = cos_2d
    sin_3d[0] = sin_2d

    cos = torch.from_numpy(cos_3d).cuda().reshape(3 * NUM_TOKENS, HALF_DIM)
    sin = torch.from_numpy(sin_3d).cuda().reshape(3 * NUM_TOKENS, HALF_DIM)

    q_out, k_out = triton_mrope(
        q, k, cos, sin,
        mrope_section=[HALF_DIM, 0, 0],
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        mrope_interleaved=False,
    )
    return q_out.cpu().numpy(), k_out.cpu().numpy()


def test_mrope_ktir():
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(42)
    q = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    k = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    cos = rng.uniform(-1, 1, (NUM_TOKENS, HALF_DIM)).astype(np.float16)
    sin = rng.uniform(-1, 1, (NUM_TOKENS, HALF_DIM)).astype(np.float16)

    outputs = interp.execute_function(
        "mrope_kernel",
        q=q.copy(),
        k=k.copy(),
        cos_ptr=cos,
        sin_ptr=sin,
        num_q_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
    )
    result_q = outputs["q"]
    result_k = outputs["k"]
    expected_q, expected_k = vllm_reference(q, k, cos, sin)

    np.testing.assert_allclose(
        result_q.astype(np.float32), expected_q.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    np.testing.assert_allclose(
        result_k.astype(np.float32), expected_k.astype(np.float32),
        rtol=5e-2, atol=5e-2,
    )
    max_err_q = np.max(np.abs(result_q.astype(np.float32) - expected_q.astype(np.float32)))
    max_err_k = np.max(np.abs(result_k.astype(np.float32) - expected_k.astype(np.float32)))
    print(f"PASS: Q max abs error = {max_err_q:.6f}, K max abs error = {max_err_k:.6f}")


def test_mrope_identity():
    """cos=1, sin=0 -> output should equal input."""
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH)

    rng = np.random.default_rng(7)
    q = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    k = rng.standard_normal((NUM_TOKENS, NUM_HEADS * HEAD_DIM)).astype(np.float16)
    cos = np.ones((NUM_TOKENS, HALF_DIM), dtype=np.float16)
    sin = np.zeros((NUM_TOKENS, HALF_DIM), dtype=np.float16)

    outputs = interp.execute_function(
        "mrope_kernel",
        q=q.copy(),
        k=k.copy(),
        cos_ptr=cos,
        sin_ptr=sin,
        num_q_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
    )

    np.testing.assert_allclose(
        outputs["q"].astype(np.float32), q.astype(np.float32),
        rtol=1e-3, atol=1e-3,
    )
    np.testing.assert_allclose(
        outputs["k"].astype(np.float32), k.astype(np.float32),
        rtol=1e-3, atol=1e-3,
    )
    print("PASS: identity (cos=1, sin=0)")


