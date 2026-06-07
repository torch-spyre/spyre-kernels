# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the tensor-descriptor RMS norm kernel.

Compares _rms_norm_kernel_td output against the original kernel across
various shapes and dtypes. This kernel uses tensor descriptors but keeps
the original one-program-per-row grid (no Spyre distribution loop).

Run: pytest tests/triton/test_rms_norm_td.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch

from kernels.rms_norm.tensor_descriptor import _rms_norm_kernel_td
from kernels.rms_norm.wrapper import rms_norm as rms_norm_original

try:
    from kernels._tma import ensure_triton_allocator
except ImportError:  # pragma: no cover
    ensure_triton_allocator = None


# ─── Helpers ──────────────────────────────────────────────────────

def rms_norm_td(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """Launch the tensor-descriptor RMS norm kernel (one program per row)."""
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0]

    if ensure_triton_allocator is not None:
        ensure_triton_allocator()

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)

    grid = (n_rows,)
    _rms_norm_kernel_td[grid](
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


# ─── Tolerances ────────────────────────────────────────────────────
#
# The td kernel and the original differ only in the order they accumulate
# the float32 sum-of-squares (the td kernel keeps a [1, BLOCK_SIZE] vector
# accumulator and reduces once at the end; the original reduces each tile
# immediately). Float addition is non-associative, so for multi-tile rows
# inv_rms can differ by a few f32 ULPs. That single scalar then scales
# every element, and the product is rounded back to the input dtype — so
# the per-element output differs by at most ~1 ULP of that dtype.
#
# 1 ULP near 1.0 is 2^-mantissa_bits: ~1e-3 for fp16 (10-bit mantissa),
# ~8e-3 for bf16 (7-bit mantissa). f32 is exact here up to scheduling.
TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=8e-3, rtol=8e-3),
}


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestRMSNormTDCorrectness:
    """Tensor-descriptor kernel must match original kernel."""

    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, hidden_size, dtype):
        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        w = torch.randn(hidden_size, device=device, dtype=dtype)
        eps = 1e-6

        out_original = rms_norm_original(x, w, eps=eps)
        out_td = rms_norm_td(x, w, eps=eps)

        torch.testing.assert_close(out_td, out_original, **TOL[dtype])

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
            out_td = rms_norm_td(x, w, BLOCK_SIZE=block_size)
            torch.testing.assert_close(
                out_td, out_original, **TOL[dtype],
                msg=f"Failed with BLOCK_SIZE={block_size}",
            )


class TestRMSNormTDEdgeCases:
    """Edge cases for the tensor-descriptor kernel."""

    def test_single_element_row(self, device):
        """hidden_size=1: single element per row."""
        torch.manual_seed(42)
        x = torch.randn(4, 1, device=device, dtype=torch.float32)
        w = torch.ones(1, device=device, dtype=torch.float32)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w, BLOCK_SIZE=64)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.float32])

    def test_single_row(self, device):
        """batch_size=1: only one row."""
        torch.manual_seed(42)
        x = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])

    def test_hidden_smaller_than_block(self, device):
        """hidden_size < BLOCK_SIZE: descriptor handles OOB with zero padding."""
        torch.manual_seed(42)
        x = torch.randn(8, 64, device=device, dtype=torch.float16)
        w = torch.randn(64, device=device, dtype=torch.float16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w, BLOCK_SIZE=1024)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.float16])

    def test_3d_input(self, device):
        """3D input (batch, seq_len, hidden) — flattened to 2D internally."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])

    def test_zero_input(self, device):
        """All-zero input — tests eps handling (avoids division by zero)."""
        x = torch.zeros(4, 4096, device=device, dtype=torch.float32)
        w = torch.ones(4096, device=device, dtype=torch.float32)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w)

        torch.testing.assert_close(
            out_td, out_original, atol=0, rtol=0, equal_nan=True,
        )

    def test_large_values(self, device):
        """Large inputs — tests numerical stability of f32 accumulation."""
        torch.manual_seed(42)
        x = torch.randn(4, 4096, device=device, dtype=torch.float16) * 1000.0
        w = torch.randn(4096, device=device, dtype=torch.float16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.float16])

    @pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
    def test_eps_values(self, device, eps):
        """Different epsilon values produce matching results."""
        torch.manual_seed(42)
        x = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w, eps=eps)
        out_td = rms_norm_td(x, w, eps=eps)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])
