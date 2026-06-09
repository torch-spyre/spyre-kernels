# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the tensor-descriptor RMS norm kernel.

Compares _rms_norm_kernel_td output against the original kernel across
various shapes and dtypes. This kernel uses tensor descriptors and batches
rows across the grid: each program processes a contiguous block of rows,
so the result must be independent of the number of programs launched.

Run: pytest tests/triton/test_rms_norm_td.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

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
    rows_per_program: int = 1,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """Launch the row-batched tensor-descriptor RMS norm kernel.

    Each program processes rows_per_program rows, so the grid has
    cdiv(n_rows, rows_per_program) programs. rows_per_program=1 is one
    program per row.
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0]

    if ensure_triton_allocator is not None:
        ensure_triton_allocator()

    original_shape = input.shape
    # No .contiguous() on the input: its real row stride is passed straight
    # to the descriptor, so a strided (e.g. column-sliced) 2D input exercises
    # the non-contiguous row path. empty_like preserves those strides, so the
    # store path is exercised strided too.
    input_2d = input.reshape(-1, input.shape[-1])
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)

    grid = (triton.cdiv(n_rows, rows_per_program),)
    _rms_norm_kernel_td[grid](
        input_2d,
        weight,
        output,
        n_rows,
        n_cols,
        input_2d.stride(0),
        output.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROGRAM=rows_per_program,
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

ROWS_PER_PROGRAM = [1, 2, 4, 8, 16, 32, 64]


# ─── Tolerances ────────────────────────────────────────────────────
#
# The td kernel and the original differ only in the order they accumulate
# the float32 sum-of-squares (the td kernel keeps a [ROWS_PER_PROGRAM,
# BLOCK_SIZE] tile accumulator and reduces once at the end; the original
# reduces each tile immediately). Float addition is non-associative, so for
# multi-tile rows
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


class TestRMSNormTDRowBatching:
    """Result must be independent of how many rows each program processes."""

    @pytest.mark.parametrize("rows_per_program", ROWS_PER_PROGRAM)
    @pytest.mark.parametrize("hidden_size", [4096, 4097])
    def test_rows_per_program_invariance(self, device, rows_per_program, hidden_size):
        """Same output regardless of how many rows a program batches."""
        torch.manual_seed(42)
        batch_size = 64
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w, rows_per_program=rows_per_program)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])

    # rows_per_program is the descriptor block_shape[0], which the
    # tensor-descriptor API requires to be a power of 2. n_rows=50 is not a
    # multiple of any of these, so the last program's tile spills past
    # n_rows and exercises OOB-row zero-padding on load and store.
    @pytest.mark.parametrize("rows_per_program", [4, 8, 16, 32, 64])
    def test_uneven_batching(self, device, rows_per_program):
        """n_rows not a multiple of rows_per_program — OOB rows load as zero."""
        torch.manual_seed(42)
        x = torch.randn(50, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w, rows_per_program=rows_per_program)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])

    def test_rows_per_program_exceeds_rows(self, device):
        """rows_per_program > n_rows — single program, OOB rows padded."""
        torch.manual_seed(42)
        x = torch.randn(4, 4096, device=device, dtype=torch.float16)
        w = torch.randn(4096, device=device, dtype=torch.float16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w, rows_per_program=32)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.float16])

    def test_all_rows_one_program(self, device):
        """A single program processes every row in one tile."""
        torch.manual_seed(42)
        x = torch.randn(64, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w)
        out_td = rms_norm_td(x, w, rows_per_program=64)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])


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

    @pytest.mark.parametrize("rows_per_program", [1, 8])
    def test_noncontiguous_rows(self, device, rows_per_program):
        """Input row stride > n_cols: a column-slice of a wider tensor.

        full[:, :n_cols] is a 2D view whose row stride is the full width
        (8192), not n_cols (4096). The td kernel must use that stride; a
        kernel hardcoding strides=[n_cols, 1] would read row r from the
        middle of physical row r-1 and produce wrong results for every
        row after the first.
        """
        torch.manual_seed(42)
        n_cols = 4096
        full = torch.randn(64, 8192, device=device, dtype=torch.bfloat16)
        x = full[:, :n_cols]  # strided view: stride(0) == 8192, stride(1) == 1
        assert not x.is_contiguous() and x.stride(0) == 8192
        w = torch.randn(n_cols, device=device, dtype=torch.bfloat16)

        # Reference: same logical data, made contiguous, through the wrapper.
        out_original = rms_norm_original(x.contiguous(), w)
        out_td = rms_norm_td(x, w, rows_per_program=rows_per_program)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])

    @pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
    def test_eps_values(self, device, eps):
        """Different epsilon values produce matching results."""
        torch.manual_seed(42)
        x = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)
        w = torch.randn(4096, device=device, dtype=torch.bfloat16)

        out_original = rms_norm_original(x, w, eps=eps)
        out_td = rms_norm_td(x, w, eps=eps)

        torch.testing.assert_close(out_td, out_original, **TOL[torch.bfloat16])
