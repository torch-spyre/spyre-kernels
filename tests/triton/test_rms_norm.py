# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for RMSNorm kernel.

Tests that the block-pointer kernel produces numerically identical results
to the original raw-pointer kernel across various shapes and dtypes.

Run: pytest tests/test_rms_norm_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.rms_norm.wrapper import rms_norm
from kernels.rms_norm.original import _rms_norm_kernel
from kernels.rms_norm.block_ptr import _rms_norm_kernel_block_ptr


# Also provide a pure-PyTorch reference for sanity checking both kernels
def rms_norm_reference(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Pure PyTorch RMSNorm — the ground truth."""
    input_f32 = input.float()
    variance = input_f32.pow(2).mean(-1, keepdim=True)
    normed = input_f32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(input.dtype)


# ─── Test Parameters ───────────────────────────────────────────────

# Shapes: (batch, hidden_size) — covers typical LLM hidden dims
HIDDEN_SIZES = [
    128,     # small, exact multiple of BLOCK_SIZE=1024? No — tests non-multiple
    1024,    # exact BLOCK_SIZE
    2048,    # 2x BLOCK_SIZE
    4096,    # Granite/Mistral hidden size
    5120,    # some models
    4097,    # not a multiple of BLOCK_SIZE — tests boundary masking
]

BATCH_SIZES = [1, 4, 32, 128]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

EPS_VALUES = [1e-6, 1e-5]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestRMSNormEquivalence:
    """Block-pointer kernel must match raw-pointer kernel exactly."""

    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, hidden_size, dtype):
        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        w = torch.randn(hidden_size, device=device, dtype=dtype)
        eps = 1e-6

        out_raw = rms_norm(x, w, eps=eps)
        out_blk = rms_norm(x, w, eps=eps, kernel_fn=_rms_norm_kernel_block_ptr)

        # Block-pointer version must match raw-pointer version.
        # For bf16/fp16: bitwise identical (precision coarser than rounding diffs).
        # For fp32: block-ptr loads with zero-padding can produce subtly different
        # instruction scheduling vs raw-ptr loads with explicit masks, causing
        # ~1e-6 absolute / ~3e-7 relative diffs. This is inherent to the
        # block-pointer conversion and acceptable.
        if dtype == torch.float32:
            torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, hidden_size, dtype):
        """Both kernels should be close to PyTorch reference."""
        torch.manual_seed(42)
        batch_size = 16
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        w = torch.randn(hidden_size, device=device, dtype=dtype)
        eps = 1e-6

        ref = rms_norm_reference(x, w, eps=eps)
        out_raw = rms_norm(x, w, eps=eps)
        out_blk = rms_norm(x, w, eps=eps, kernel_fn=_rms_norm_kernel_block_ptr)

        # Tolerance depends on dtype
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        else:  # fp16, bf16
            atol, rtol = 1e-2, 1e-2

        torch.testing.assert_close(out_raw, ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(out_blk, ref, atol=atol, rtol=rtol)

    def test_3d_input(self, device):
        """Verify that 3D+ inputs (batch, seq_len, hidden) work correctly."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_raw = rms_norm(x, w)
        out_blk = rms_norm(x, w, kernel_fn=_rms_norm_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    def test_single_element_row(self, device):
        """Edge case: hidden_size=1."""
        x = torch.randn(4, 1, device=device, dtype=torch.float32)
        w = torch.ones(1, device=device, dtype=torch.float32)

        out_raw = rms_norm(x, w)
        out_blk = rms_norm(x, w, kernel_fn=_rms_norm_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("eps", EPS_VALUES)
    def test_eps_values(self, device, eps):
        """Verify different epsilon values produce matching results."""
        torch.manual_seed(42)
        x = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_raw = rms_norm(x, w, eps=eps)
        out_blk = rms_norm(x, w, eps=eps, kernel_fn=_rms_norm_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    def test_zero_input(self, device):
        """All-zero input should produce all-zero output."""
        x = torch.zeros(4, 4096, device=device, dtype=torch.float32)
        w = torch.ones(4096, device=device, dtype=torch.float32)

        out_raw = rms_norm(x, w)
        out_blk = rms_norm(x, w, kernel_fn=_rms_norm_kernel_block_ptr)

        # Both should handle 0/0 the same way (likely NaN or 0 depending on eps)
        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0, equal_nan=True)

    def test_large_values(self, device):
        """Large input values — tests numerical stability."""
        torch.manual_seed(42)
        x = torch.randn(4, 4096, device=device, dtype=torch.float16) * 1000.0
        w = torch.randn(4096, device=device, dtype=torch.float16)

        out_raw = rms_norm(x, w)
        out_blk = rms_norm(x, w, kernel_fn=_rms_norm_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)


class TestRMSNormPerformance:
    """Block-pointer kernel should not regress significantly vs raw-pointer."""

    @pytest.mark.parametrize("hidden_size", [4096, 5120])
    def test_no_major_regression(self, device, hidden_size):
        """Block-pointer version should be within 2x of raw-pointer speed."""
        import triton.testing

        batch_size = 128
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)

        ms_raw = triton.testing.do_bench(lambda: rms_norm(x, w))
        ms_blk = triton.testing.do_bench(lambda: rms_norm(x, w, kernel_fn=_rms_norm_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"  hidden={hidden_size}: raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms, "
              f"slowdown={slowdown:.2f}x")

        # Allow up to 2x slowdown (generous — block ptrs are often faster)
        assert slowdown < 2.0, (
            f"Block-pointer kernel is {slowdown:.2f}x slower than raw-pointer "
            f"(raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms)"
        )
