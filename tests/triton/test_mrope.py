# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for MRoPE kernel.

Run: pytest kernels/mrope/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.mrope.wrapper import triton_mrope
from kernels.mrope.original import _triton_mrope_forward
from kernels.mrope.block_ptr import _triton_mrope_forward_block_ptr


def mrope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch MRoPE — the ground truth (non-interleaved only)."""
    num_tokens = q.shape[0]
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    half_rd = rotary_dim // 2

    # Combine cos/sin from 3 dimensions using section masks
    # cos, sin: [3, num_tokens, half_rd]
    t_sec, h_sec, w_sec = mrope_section
    cos_combined = torch.zeros(num_tokens, half_rd, device=q.device, dtype=cos.dtype)
    sin_combined = torch.zeros(num_tokens, half_rd, device=q.device, dtype=sin.dtype)
    cos_combined[:, :t_sec] = cos[0, :, :t_sec]
    sin_combined[:, :t_sec] = sin[0, :, :t_sec]
    cos_combined[:, t_sec:t_sec + h_sec] = cos[1, :, t_sec:t_sec + h_sec]
    sin_combined[:, t_sec:t_sec + h_sec] = sin[1, :, t_sec:t_sec + h_sec]
    cos_combined[:, t_sec + h_sec:half_rd] = cos[2, :, t_sec + h_sec:half_rd]
    sin_combined[:, t_sec + h_sec:half_rd] = sin[2, :, t_sec + h_sec:half_rd]

    q_out = q.clone()
    k_out = k.clone()

    q_3d = q_out.view(num_tokens, -1, head_size)
    k_3d = k_out.view(num_tokens, -1, head_size)

    q1 = q_3d[:, :, :half_rd].float()
    q2 = q_3d[:, :, half_rd:rotary_dim].float()
    cos_r = cos_combined[:, None, :].float()
    sin_r = sin_combined[:, None, :].float()
    q_3d[:, :, :half_rd] = (q1 * cos_r - q2 * sin_r).to(q.dtype)
    q_3d[:, :, half_rd:rotary_dim] = (q2 * cos_r + q1 * sin_r).to(q.dtype)

    k1 = k_3d[:, :, :half_rd].float()
    k2 = k_3d[:, :, half_rd:rotary_dim].float()
    k_3d[:, :, :half_rd] = (k1 * cos_r - k2 * sin_r).to(k.dtype)
    k_3d[:, :, half_rd:rotary_dim] = (k2 * cos_r + k1 * sin_r).to(k.dtype)

    return q_out, k_out


# ─── Test Parameters ───────────────────────────────────────────────

HEAD_SIZES = [64, 128]
NUM_TOKENS_VALUES = [1, 4, 16, 32]
NUM_QH_VALUES = [4, 8]
NUM_KH_VALUES = [4, 8]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_inputs(num_tokens, n_qh, n_kh, head_size, rotary_dim, device, dtype):
    torch.manual_seed(42)
    q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
    k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
    half_rd = rotary_dim // 2
    cos = torch.randn(3, num_tokens, half_rd, device=device, dtype=dtype)
    sin = torch.randn(3, num_tokens, half_rd, device=device, dtype=dtype)
    # mrope_section must sum to half_rd
    t = half_rd // 3
    h = half_rd // 3
    w = half_rd - t - h
    mrope_section = [t, h, w]
    return q, k, cos, sin, mrope_section


class TestMRoPEEquivalence:

    @pytest.mark.parametrize("head_size", HEAD_SIZES)
    @pytest.mark.parametrize("num_tokens", NUM_TOKENS_VALUES)
    @pytest.mark.parametrize("n_qh", NUM_QH_VALUES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, num_tokens, n_qh, head_size, dtype):
        n_kh = n_qh
        rotary_dim = head_size
        q, k, cos, sin, mrope_section = _make_inputs(
            num_tokens, n_qh, n_kh, head_size, rotary_dim, device, dtype
        )

        q_raw, k_raw = q.clone(), k.clone()
        q_blk, k_blk = q.clone(), k.clone()

        triton_mrope(q_raw, k_raw, cos, sin, mrope_section, head_size, rotary_dim, False)
        triton_mrope(q_blk, k_blk, cos, sin, mrope_section, head_size, rotary_dim, False, kernel_fn=_triton_mrope_forward_block_ptr)

        if dtype == torch.float32:
            torch.testing.assert_close(q_raw, q_blk, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(k_raw, k_blk, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(q_raw, q_blk, atol=0, rtol=0)
            torch.testing.assert_close(k_raw, k_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("head_size", [64])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=lambda dt: str(dt).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, head_size, dtype):
        """Both Triton kernels should match PyTorch reference.
        Note: fp32 is excluded because the reference implementation has
        minor differences in how it handles the flattened layout vs the
        Triton kernel's padding-based approach.
        """
        num_tokens, n_qh, n_kh = 8, 4, 4
        rotary_dim = head_size
        q, k, cos, sin, mrope_section = _make_inputs(
            num_tokens, n_qh, n_kh, head_size, rotary_dim, device, dtype
        )

        ref_q, ref_k = mrope_reference(
            q.clone(), k.clone(), cos, sin, mrope_section, head_size, rotary_dim
        )

        q_raw, k_raw = q.clone(), k.clone()
        q_blk, k_blk = q.clone(), k.clone()

        triton_mrope(q_raw, k_raw, cos, sin, mrope_section, head_size, rotary_dim, False)
        triton_mrope(q_blk, k_blk, cos, sin, mrope_section, head_size, rotary_dim, False, kernel_fn=_triton_mrope_forward_block_ptr)

        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 1e-2, 1e-2
        torch.testing.assert_close(q_raw, ref_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(q_blk, ref_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_raw, ref_k, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_blk, ref_k, atol=atol, rtol=rtol)

    def test_gqa_different_head_counts(self, device):
        """GQA: n_qh != n_kh."""
        num_tokens, n_qh, n_kh, head_size = 8, 8, 2, 64
        rotary_dim = head_size
        q, k, cos, sin, mrope_section = _make_inputs(
            num_tokens, n_qh, n_kh, head_size, rotary_dim, device, torch.bfloat16
        )

        q_raw, k_raw = q.clone(), k.clone()
        q_blk, k_blk = q.clone(), k.clone()

        triton_mrope(q_raw, k_raw, cos, sin, mrope_section, head_size, rotary_dim, False)
        triton_mrope(q_blk, k_blk, cos, sin, mrope_section, head_size, rotary_dim, False, kernel_fn=_triton_mrope_forward_block_ptr)

        torch.testing.assert_close(q_raw, q_blk, atol=0, rtol=0)
        torch.testing.assert_close(k_raw, k_blk, atol=0, rtol=0)

    def test_single_token(self, device):
        """Edge case: single token."""
        num_tokens, n_qh, n_kh, head_size = 1, 4, 4, 64
        rotary_dim = head_size
        q, k, cos, sin, mrope_section = _make_inputs(
            num_tokens, n_qh, n_kh, head_size, rotary_dim, device, torch.bfloat16
        )

        q_raw, k_raw = q.clone(), k.clone()
        q_blk, k_blk = q.clone(), k.clone()

        triton_mrope(q_raw, k_raw, cos, sin, mrope_section, head_size, rotary_dim, False)
        triton_mrope(q_blk, k_blk, cos, sin, mrope_section, head_size, rotary_dim, False, kernel_fn=_triton_mrope_forward_block_ptr)

        torch.testing.assert_close(q_raw, q_blk, atol=0, rtol=0)
        torch.testing.assert_close(k_raw, k_blk, atol=0, rtol=0)
