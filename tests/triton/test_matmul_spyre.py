# SPDX-License-Identifier: MIT
"""
Numerical tests for the Spyre-aware matmul kernel.

Compares matmul_kernel_spyre output against the original raw-pointer kernel
across various shapes and core counts.

Run: pytest tests/triton/test_matmul_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.matmul.spyre import matmul_kernel_spyre
from kernels.matmul.wrapper import matmul as matmul_original


# ─── Helpers ──────────────────────────────────────────────────────

def matmul_spyre(
    a: torch.Tensor,
    b: torch.Tensor,
    activation: str = "",
    num_cores: int = 32,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (num_cores,)
    matmul_kernel_spyre[grid](
        a, b, c,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ACTIVATION=activation,
    )
    return c


# ─── Test Parameters ───────────────────────────────────────────────

MATRIX_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (128, 256, 512),
    (256, 128, 64),
    (513, 257, 129),  # non-power-of-2 — tests tail handling
    (65, 33, 17),     # small non-aligned
]

CORE_COUNTS = [1, 4, 16, 32]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestMatmulSpyreCorrectness:
    """Spyre kernel must match original kernel."""

    @pytest.mark.parametrize("M,N,K", MATRIX_SHAPES)
    def test_vs_original_kernel(self, device, M, N, K):
        torch.manual_seed(42)
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b)
        out_spyre = matmul_spyre(a, b)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    def test_activation_leaky_relu(self, device):
        torch.manual_seed(42)
        M, N, K = 256, 256, 256
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b, activation="leaky_relu")
        out_spyre = matmul_spyre(a, b, activation="leaky_relu")

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)


class TestMatmulSpyreDistribution:
    """Verify correctness across different core counts."""

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    def test_core_count_invariance(self, device, num_cores):
        """Result must be identical regardless of how many cores are used."""
        torch.manual_seed(42)
        M, N, K = 256, 256, 256
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b)
        out_spyre = matmul_spyre(a, b, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    def test_non_divisible_with_varying_cores(self, device, num_cores):
        """Non-divisible shapes work for any core count."""
        torch.manual_seed(42)
        M, N, K = 513, 257, 129
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b)
        out_spyre = matmul_spyre(a, b, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)


class TestMatmulSpyreEdgeCases:
    """Edge cases for the Spyre kernel."""

    def test_rectangular_tall(self, device):
        """Tall matrix: M >> N."""
        torch.manual_seed(42)
        a = torch.randn((2048, 128), device=device, dtype=torch.float16)
        b = torch.randn((128, 64), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b)
        out_spyre = matmul_spyre(a, b)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    def test_rectangular_wide(self, device):
        """Wide matrix: N >> M."""
        torch.manual_seed(42)
        a = torch.randn((64, 128), device=device, dtype=torch.float16)
        b = torch.randn((128, 2048), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b)
        out_spyre = matmul_spyre(a, b)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    def test_k_not_divisible_by_block_k(self, device):
        """K dimension not a multiple of BLOCK_K."""
        torch.manual_seed(42)
        a = torch.randn((128, 100), device=device, dtype=torch.float16)
        b = torch.randn((100, 128), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b)
        out_spyre = matmul_spyre(a, b, BLOCK_K=64)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    def test_more_cores_than_blocks(self, device):
        """Grid has more cores than output tiles — some cores do no work."""
        torch.manual_seed(42)
        a = torch.randn((64, 64), device=device, dtype=torch.float16)
        b = torch.randn((64, 64), device=device, dtype=torch.float16)

        out_original = matmul_original(a, b)
        out_spyre = matmul_spyre(a, b, num_cores=32, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)
