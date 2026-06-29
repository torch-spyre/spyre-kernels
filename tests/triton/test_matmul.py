# SPDX-License-Identifier: MIT
"""
Phase 2 Validation: GPU equivalence tests for matmul kernel.

Tests that the block-pointer kernel produces numerically identical results
to the original raw-pointer kernel across various shapes.

Run: pytest tests/triton/test_matmul.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.matmul.wrapper import matmul
from kernels.matmul.original import matmul_kernel
from kernels.matmul.block_ptr import matmul_kernel_block_ptr


# ─── Test Parameters ───────────────────────────────────────────────

MATRIX_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (128, 256, 512),
    (256, 128, 64),
    (513, 257, 129),  # non-power-of-2 — tests boundary handling
]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestMatmulEquivalence:
    """Block-pointer kernel must match raw-pointer kernel."""

    @pytest.mark.parametrize("M,N,K", MATRIX_SHAPES)
    def test_numerical_equivalence(self, device, M, N, K):
        torch.manual_seed(42)
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_raw = matmul(a, b)
        out_blk = matmul(a, b, kernel_fn=matmul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-2, rtol=0)

    @pytest.mark.parametrize("M,N,K", MATRIX_SHAPES)
    def test_correctness_vs_pytorch(self, device, M, N, K):
        """Both kernels should be close to PyTorch reference (cuBLAS)."""
        torch.manual_seed(42)
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        ref = torch.matmul(a, b)
        out_raw = matmul(a, b)
        out_blk = matmul(a, b, kernel_fn=matmul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, ref, atol=5e-2, rtol=1e-2)
        torch.testing.assert_close(out_blk, ref, atol=5e-2, rtol=1e-2)

    def test_rectangular_tall(self, device):
        """Tall matrix: M >> N."""
        torch.manual_seed(42)
        a = torch.randn((2048, 128), device=device, dtype=torch.float16)
        b = torch.randn((128, 64), device=device, dtype=torch.float16)

        out_raw = matmul(a, b)
        out_blk = matmul(a, b, kernel_fn=matmul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-2, rtol=0)

    def test_rectangular_wide(self, device):
        """Wide matrix: N >> M."""
        torch.manual_seed(42)
        a = torch.randn((64, 128), device=device, dtype=torch.float16)
        b = torch.randn((128, 2048), device=device, dtype=torch.float16)

        out_raw = matmul(a, b)
        out_blk = matmul(a, b, kernel_fn=matmul_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-2, rtol=0)


class TestMatmulPerformance:
    """Block-pointer kernel should not regress significantly vs raw-pointer."""

    @pytest.mark.parametrize("size", [512, 1024, 2048])
    def test_no_major_regression(self, device, size):
        """Block-pointer version should be within 2x of raw-pointer speed."""
        import triton.testing

        a = torch.randn((size, size), device=device, dtype=torch.float16)
        b = torch.randn((size, size), device=device, dtype=torch.float16)

        ms_raw = triton.testing.do_bench(lambda: matmul(a, b))
        ms_blk = triton.testing.do_bench(lambda: matmul(a, b, kernel_fn=matmul_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"  size={size}: raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms, "
              f"slowdown={slowdown:.2f}x")

        assert slowdown < 2.0, (
            f"Block-pointer kernel is {slowdown:.2f}x slower than raw-pointer "
            f"(raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms)"
        )
