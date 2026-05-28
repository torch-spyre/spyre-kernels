# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the Spyre-aware SiLU-and-mul kernel.

Compares _swiglustep_and_mul_kernel_spyre output against the original kernel
across various shapes and core counts.

Run: pytest tests/triton/test_silu_and_mul_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.silu_and_mul.spyre import _swiglustep_and_mul_kernel_spyre
from kernels.silu_and_mul.wrapper import silu_and_mul as silu_and_mul_original


# ─── Helpers ──────────────────────────────────────────────────────

def silu_and_mul_spyre(
    x: torch.Tensor,
    limit: float = 7.0,
    num_cores: int = 32,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """Launch the Spyre SiLU-and-mul kernel with a fixed grid."""
    original_shape = x.shape
    assert original_shape[-1] % 2 == 0
    d = original_shape[-1] // 2

    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    n_rows = x_2d.shape[0]

    output = torch.empty(n_rows, d, device=x.device, dtype=x.dtype)

    grid = (num_cores,)
    _swiglustep_and_mul_kernel_spyre[grid](
        x_2d,
        output,
        n_rows,
        d,
        limit,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(*original_shape[:-1], d)


# ─── Test Parameters ───────────────────────────────────────────────

D_SIZES = [
    64,      # smaller than BLOCK_SIZE
    1024,    # exact BLOCK_SIZE
    2048,    # 2x BLOCK_SIZE
    4096,    # typical MLP hidden dim
    4097,    # non-divisible
]

BATCH_SIZES = [1, 4, 32, 128]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

CORE_COUNTS = [1, 4, 16, 32]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestSiluAndMulSpyreCorrectness:
    """Spyre kernel must match original kernel."""

    @pytest.mark.parametrize("d", D_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, d, dtype):
        torch.manual_seed(42)
        x = torch.randn(batch_size, 2 * d, device=device, dtype=dtype)

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x)

        if dtype == torch.float32:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize("d", D_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_varying_block_size(self, device, d, dtype):
        """Verify correctness with different BLOCK_SIZE constexprs."""
        torch.manual_seed(42)
        batch_size = 16
        x = torch.randn(batch_size, 2 * d, device=device, dtype=dtype)

        out_original = silu_and_mul_original(x)

        for block_size in [256, 512, 1024, 2048]:
            out_spyre = silu_and_mul_spyre(x, BLOCK_SIZE=block_size)
            if dtype == torch.float32:
                torch.testing.assert_close(
                    out_spyre, out_original, atol=1e-5, rtol=1e-5,
                    msg=f"Failed with BLOCK_SIZE={block_size}",
                )
            else:
                torch.testing.assert_close(
                    out_spyre, out_original, atol=2e-3, rtol=1e-3,
                    msg=f"Failed with BLOCK_SIZE={block_size}",
                )


class TestSiluAndMulSpyreDistribution:
    """Verify correctness across different core counts."""

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    @pytest.mark.parametrize("d", [4096, 4097])
    def test_core_count_invariance(self, device, num_cores, d):
        """Result must be identical regardless of how rows are distributed."""
        torch.manual_seed(42)
        batch_size = 64
        x = torch.randn(batch_size, 2 * d, device=device, dtype=torch.bfloat16)

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_more_cores_than_rows(self, device):
        """When num_cores > n_rows, some cores have no work — must not crash."""
        torch.manual_seed(42)
        x = torch.randn(4, 8192, device=device, dtype=torch.float16)

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x, num_cores=32)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_single_core(self, device):
        """Single core must process all rows sequentially."""
        torch.manual_seed(42)
        x = torch.randn(64, 8192, device=device, dtype=torch.bfloat16)

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x, num_cores=1)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)


class TestSiluAndMulSpyreEdgeCases:
    """Edge cases for the Spyre kernel."""

    def test_single_element(self, device):
        """d=1: single element per half."""
        torch.manual_seed(42)
        x = torch.randn(4, 2, device=device, dtype=torch.float32)

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x, BLOCK_SIZE=64)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_single_row(self, device):
        """batch_size=1: only one row to distribute."""
        torch.manual_seed(42)
        x = torch.randn(1, 8192, device=device, dtype=torch.bfloat16)

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x, num_cores=32)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_3d_input(self, device):
        """3D input (batch, seq_len, 2*d) — flattened to 2D internally."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 8192, device=device, dtype=torch.bfloat16)

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_large_values(self, device):
        """Large inputs test the clamping logic."""
        torch.manual_seed(42)
        x = torch.randn(4, 8192, device=device, dtype=torch.float16) * 100.0

        out_original = silu_and_mul_original(x)
        out_spyre = silu_and_mul_spyre(x)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize("limit", [1.0, 7.0, 100.0])
    def test_limit_values(self, device, limit):
        """Different limit values produce matching results."""
        torch.manual_seed(42)
        x = torch.randn(8, 8192, device=device, dtype=torch.bfloat16)

        out_original = silu_and_mul_original(x, limit=limit)
        out_spyre = silu_and_mul_spyre(x, limit=limit)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)
