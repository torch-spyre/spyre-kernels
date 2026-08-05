# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for SwiGLU (silu_and_mul) kernel.

Tests that the block-pointer kernel produces numerically identical results
to the original raw-pointer kernel across various shapes and dtypes.

Run: pytest kernels/vllm/silu_and_mul/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.vllm.silu_and_mul.wrapper import silu_and_mul
from kernels.vllm.silu_and_mul.original import _swiglustep_and_mul_kernel
from kernels.vllm.silu_and_mul.block_ptr import _swiglustep_and_mul_kernel_block_ptr


def silu_and_mul_reference(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """Pure PyTorch SwiGLU — the ground truth."""
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate_silu = torch.sigmoid(gate) * gate
    gate_clamped = torch.clamp(gate_silu, max=limit)
    up_clamped = torch.clamp(up, min=-limit, max=limit)
    result = gate_clamped * up_clamped
    return result.to(x.dtype)


# ─── Test Parameters ───────────────────────────────────────────────

HALF_HIDDEN_SIZES = [
    128,
    512,
    1024,
    2048,
    4096,
    4097,  # not a multiple of BLOCK_SIZE — tests boundary
]

BATCH_SIZES = [1, 4, 32, 128]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestSiluAndMulEquivalence:
    """Block-pointer kernel must match raw-pointer kernel exactly."""

    @pytest.mark.parametrize("d", HALF_HIDDEN_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, d, dtype):
        torch.manual_seed(42)
        x = torch.randn(batch_size, 2 * d, device=device, dtype=dtype)

        out_raw = silu_and_mul(x)
        out_blk = silu_and_mul(x, kernel_fn=_swiglustep_and_mul_kernel_block_ptr)

        if dtype == torch.float32:
            torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("d", HALF_HIDDEN_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, d, dtype):
        """Both kernels should be close to PyTorch reference."""
        torch.manual_seed(42)
        batch_size = 16
        x = torch.randn(batch_size, 2 * d, device=device, dtype=dtype)

        ref = silu_and_mul_reference(x)
        out_raw = silu_and_mul(x)
        out_blk = silu_and_mul(x, kernel_fn=_swiglustep_and_mul_kernel_block_ptr)

        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        else:
            atol, rtol = 1e-2, 1e-2

        torch.testing.assert_close(out_raw, ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(out_blk, ref, atol=atol, rtol=rtol)

    def test_3d_input(self, device):
        """Verify that 3D+ inputs (batch, seq_len, 2*d) work correctly."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 2 * 4096, device=device, dtype=torch.bfloat16)

        out_raw = silu_and_mul(x)
        out_blk = silu_and_mul(x, kernel_fn=_swiglustep_and_mul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    def test_clamping_active(self, device):
        """Large values should be clamped — verify both kernels clamp identically."""
        torch.manual_seed(42)
        x = torch.randn(4, 2 * 1024, device=device, dtype=torch.float32) * 100.0

        out_raw = silu_and_mul(x, limit=7.0)
        out_blk = silu_and_mul(x, limit=7.0, kernel_fn=_swiglustep_and_mul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)

    def test_zero_input(self, device):
        """All-zero input: silu(0)=0, so output should be all zeros."""
        x = torch.zeros(4, 2 * 4096, device=device, dtype=torch.float32)

        out_raw = silu_and_mul(x)
        out_blk = silu_and_mul(x, kernel_fn=_swiglustep_and_mul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)
        assert torch.all(out_blk == 0)

    @pytest.mark.parametrize("limit", [1.0, 7.0, 100.0])
    def test_different_limits(self, device, limit):
        """Verify different clamp limits produce matching results."""
        torch.manual_seed(42)
        x = torch.randn(8, 2 * 4096, device=device, dtype=torch.bfloat16)

        out_raw = silu_and_mul(x, limit=limit)
        out_blk = silu_and_mul(x, limit=limit, kernel_fn=_swiglustep_and_mul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)


class TestSiluAndMulPerformance:
    """Block-pointer kernel should not regress significantly vs raw-pointer."""

    @pytest.mark.parametrize("d", [4096, 5120])
    def test_no_major_regression(self, device, d):
        """Block-pointer version should be within 2x of raw-pointer speed."""
        import triton.testing

        batch_size = 128
        x = torch.randn(batch_size, 2 * d, device=device, dtype=torch.bfloat16)

        ms_raw = triton.testing.do_bench(lambda: silu_and_mul(x))
        ms_blk = triton.testing.do_bench(lambda: silu_and_mul(x, kernel_fn=_swiglustep_and_mul_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"  d={d}: raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms, "
              f"slowdown={slowdown:.2f}x")

        assert slowdown < 2.0, (
            f"Block-pointer kernel is {slowdown:.2f}x slower than raw-pointer "
            f"(raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms)"
        )
