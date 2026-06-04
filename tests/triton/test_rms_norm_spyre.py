# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the Spyre-aware RMS norm kernel.

Compares _rms_norm_kernel_spyre output against the original kernel
across various shapes and core counts.

Run: pytest tests/triton/test_rms_norm_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.rms_norm.spyre import _rms_norm_kernel_spyre
from kernels.rms_norm.wrapper import rms_norm as rms_norm_original


# ─── Helpers ──────────────────────────────────────────────────────

def rms_norm_spyre(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    num_cores: int = 32,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """Launch the Spyre RMS norm kernel with a fixed grid."""
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0]

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)

    grid = (num_cores,)
    _rms_norm_kernel_spyre[grid](
        input_2d,
        weight,
        output,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


# ─── Test Parameters ───────────────────────────────────────────────

HIDDEN_SIZES = [
    128,     # smaller than BLOCK_SIZE
    1024,    # exact BLOCK_SIZE
    2048,    # 2x BLOCK_SIZE
    4096,    # typical LLM hidden dim
    4097,    # not a multiple of BLOCK_SIZE — tests OOB handling
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

class TestRMSNormSpyreCorrectness:
    """Spyre kernel must match original kernel."""

    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, hidden_size, dtype):
        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        w = torch.randn(hidden_size, device=device, dtype=dtype)
        eps = 1e-6

        out_original = rms_norm_original(x, w, eps=eps)
        out_spyre = rms_norm_spyre(x, w, eps=eps)

        if dtype == torch.float32:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)
        else:
            # Small diffs expected: Spyre accumulates element-wise across tiles
            # then reduces, vs original which reduces each tile immediately.
            # Different FP addition order → ≤1 ULP difference in half precision.
            torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
    def test_varying_block_size(self, device, hidden_size, dtype):
        """Verify correctness with different BLOCK_SIZE constexprs."""
        torch.manual_seed(42)
        batch_size = 16
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        w = torch.randn(hidden_size, device=device, dtype=dtype)

        out_original = rms_norm_original(x, w)

        for block_size in [256, 512, 1024, 2048]:
            out_spyre = rms_norm_spyre(x, w, BLOCK_SIZE=block_size)
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


class TestRMSNormSpyreDistribution:
    """Verify correctness across different core counts."""

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    @pytest.mark.parametrize("hidden_size", [4096, 4097])
    def test_core_count_invariance(self, device, num_cores, hidden_size):
        """Result must be identical regardless of how rows are distributed."""
        torch.manual_seed(42)
        batch_size = 64
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_more_cores_than_rows(self, device):
        """When num_cores > n_rows, some cores have no work — must not crash."""
        torch.manual_seed(42)
        x = torch.randn(4, 4096, device=device, dtype=torch.float16)
        w = torch.randn(4096, device=device, dtype=torch.float16)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w, num_cores=32)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_single_core(self, device):
        """Single core must process all rows sequentially."""
        torch.manual_seed(42)
        x = torch.randn(64, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w, num_cores=1)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)


class TestRMSNormSpyreEdgeCases:
    """Edge cases for the Spyre kernel."""

    def test_single_element_row(self, device):
        """hidden_size=1: single element per row."""
        torch.manual_seed(42)
        x = torch.randn(4, 1, device=device, dtype=torch.float32)
        w = torch.ones(1, device=device, dtype=torch.float32)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w, BLOCK_SIZE=64)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_single_row(self, device):
        """batch_size=1: only one row to distribute."""
        torch.manual_seed(42)
        x = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w, num_cores=32)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_hidden_smaller_than_block(self, device):
        """hidden_size < BLOCK_SIZE: descriptor handles OOB with zero padding."""
        torch.manual_seed(42)
        x = torch.randn(8, 64, device=device, dtype=torch.float16)
        w = torch.randn(64, device=device, dtype=torch.float16)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w, BLOCK_SIZE=1024)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_3d_input(self, device):
        """3D input (batch, seq_len, hidden) — flattened to 2D internally."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    def test_zero_input(self, device):
        """All-zero input — tests eps handling (avoids division by zero)."""
        x = torch.zeros(4, 4096, device=device, dtype=torch.float32)
        w = torch.ones(4096, device=device, dtype=torch.float32)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w)

        torch.testing.assert_close(
            out_spyre, out_original, atol=0, rtol=0, equal_nan=True,
        )

    def test_large_values(self, device):
        """Large inputs — tests numerical stability of f32 accumulation."""
        torch.manual_seed(42)
        x = torch.randn(4, 4096, device=device, dtype=torch.float16) * 1000.0
        w = torch.randn(4096, device=device, dtype=torch.float16)

        out_original = rms_norm_original(x, w)
        out_spyre = rms_norm_spyre(x, w)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
    def test_eps_values(self, device, eps):
        """Different epsilon values produce matching results."""
        torch.manual_seed(42)
        x = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w, eps=eps)
        out_spyre = rms_norm_spyre(x, w, eps=eps)

        torch.testing.assert_close(out_spyre, out_original, atol=2e-3, rtol=1e-3)
