# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the tensor-descriptor MRoPE kernel."""

import pytest
import torch

from kernels.mrope.tensor_descriptor import _mrope_kernel_td
from kernels.mrope.wrapper import triton_mrope


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_inputs(num_tokens, n_qh, n_kh, head_size, rotary_dim, device, dtype, scale=1.0):
    torch.manual_seed(42)
    q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype) * scale
    k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype) * scale
    half_rd = rotary_dim // 2
    cos = torch.randn(3, num_tokens, half_rd, device=device, dtype=dtype)
    sin = torch.randn(3, num_tokens, half_rd, device=device, dtype=dtype)
    t = half_rd // 3
    h = half_rd // 3
    w = half_rd - t - h
    return q, k, cos, sin, [t, h, w]


# Same per-token rotation, but raw-pointer+mask (original) vs descriptor+where
# (td) lower differently, so the compiler may contract a*cos - b*sin into FMA on
# one path and not the other -> ~1 ULP per operand. The rotation is a
# *subtraction* (a*cos - b*sin), so when the two terms nearly cancel the small
# operand error becomes a large *relative* error on a near-zero result -> rel
# tol must be loose even though abs stays ~ULP. fp16 (10-bit mantissa) needs
# ~2 ULP abs; bf16 (7-bit) more. Confirmed on A100: fp32/bf16 pass tight, fp16
# tripped at abs 1.5e-3 / rel 2.7e-2 on a cancelling lane.
TOL = {
    torch.float32: dict(atol=1e-6, rtol=1e-6),
    torch.float16: dict(atol=2e-3, rtol=3e-2),
    torch.bfloat16: dict(atol=8e-3, rtol=8e-3),
}


def _run_td(q, k, cos, sin, sections, head_size, rotary_dim, interleaved=False):
    return triton_mrope(
        q,
        k,
        cos,
        sin,
        sections,
        head_size,
        rotary_dim,
        interleaved,
        kernel_fn=_mrope_kernel_td,
    )


def _assert_matches(device, dtype, head_size, rotary_dim, num_tokens, n_qh, n_kh,
                    interleaved=False, scale=1.0):
    """Run original + _td through the wrapper, assert bitwise-equal q/k."""
    q, k, cos, sin, sections = _make_inputs(
        num_tokens, n_qh, n_kh, head_size, rotary_dim, device, dtype, scale
    )
    q_original, k_original = q.clone(), k.clone()
    q_td, k_td = q.clone(), k.clone()

    triton_mrope(q_original, k_original, cos, sin, sections, head_size, rotary_dim,
                 interleaved)
    _run_td(q_td, k_td, cos, sin, sections, head_size, rotary_dim, interleaved)

    torch.testing.assert_close(q_td, q_original, **TOL[dtype])
    torch.testing.assert_close(k_td, k_original, **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("head_size", "rotary_dim"),
    [
        (64, 64),
        (128, 64),
        (96, 64),
    ],
)
def test_mrope_td_matches_original(device, dtype, head_size, rotary_dim):
    _assert_matches(device, dtype, head_size, rotary_dim,
                    num_tokens=8, n_qh=4, n_kh=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mrope_td_interleaved(device, dtype):
    _assert_matches(device, dtype, head_size=64, rotary_dim=64,
                    num_tokens=8, n_qh=4, n_kh=2, interleaved=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mrope_td_minimum_size(device, dtype):
    # Single token, single head each — smallest valid problem.
    _assert_matches(device, dtype, head_size=64, rotary_dim=64,
                    num_tokens=1, n_qh=1, n_kh=1)


@pytest.mark.parametrize(("n_qh", "n_kh"), [(3, 1), (5, 3), (1, 1)])
def test_mrope_td_non_pow2_heads(device, n_qh, n_kh):
    # Non-power-of-2 head counts exercise the pad_n_qh/pad_n_kh OOB padding.
    _assert_matches(device, torch.bfloat16, head_size=64, rotary_dim=64,
                    num_tokens=6, n_qh=n_qh, n_kh=n_kh)


def test_mrope_td_large_values(device):
    _assert_matches(device, torch.float32, head_size=128, rotary_dim=64,
                    num_tokens=4, n_qh=4, n_kh=2, scale=1e4)


def test_mrope_td_partial_rotary_preserves_tail(device):
    head_size = 128
    rotary_dim = 64
    q, k, cos, sin, sections = _make_inputs(
        num_tokens=4,
        n_qh=4,
        n_kh=2,
        head_size=head_size,
        rotary_dim=rotary_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q_tail = q.view(4, 4, head_size)[:, :, rotary_dim:].clone()
    k_tail = k.view(4, 2, head_size)[:, :, rotary_dim:].clone()

    _run_td(q, k, cos, sin, sections, head_size, rotary_dim)

    # Tail is never written by the kernel -> must stay bitwise-identical (no math, no FMA).
    torch.testing.assert_close(q.view(4, 4, head_size)[:, :, rotary_dim:], q_tail, atol=0, rtol=0)
    torch.testing.assert_close(k.view(4, 2, head_size)[:, :, rotary_dim:], k_tail, atol=0, rtol=0)
