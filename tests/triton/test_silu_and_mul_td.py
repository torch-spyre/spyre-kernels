# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the tensor-descriptor silu_and_mul (SwiGLU) kernel.

Compares _silu_and_mul_kernel_td output against the original kernel across
various shapes and dtypes. Both kernels are launched through
kernels/silu_and_mul/wrapper.py (via its kernel_fn= dispatch) — no forked
launch path.

Unlike a reduction kernel, this kernel is purely elementwise: every output
element is sigmoid(gate)*gate, clamped, times clamped up — the same float ops
in the same order as the original, with no cross-tile accumulation. The td
kernel only changes how memory is addressed (descriptors vs raw pointers), so
the result is bitwise-identical to the original for every dtype. All
comparisons assert atol=0, rtol=0; a loose tolerance here would hide a
regression.

Run: pytest tests/triton/test_silu_and_mul_td.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch

from kernels.silu_and_mul.tensor_descriptor import _silu_and_mul_kernel_td
from kernels.silu_and_mul.wrapper import silu_and_mul


# ─── Launch helpers (both go through the wrapper) ──────────────────

def silu_ref(x, **kwargs):
    """Reference: original kernel via the wrapper."""
    return silu_and_mul(x, **kwargs)


def silu_td(x, **kwargs):
    """Tensor-descriptor kernel via the wrapper's kernel_fn dispatch."""
    return silu_and_mul(x, kernel_fn=_silu_and_mul_kernel_td, **kwargs)


# ─── Test Parameters ───────────────────────────────────────────────

# d is the half-width (output width); input width is 2*d. BLOCK_SIZE is 1024.
HALF_HIDDEN_SIZES = [
    128,     # smaller than BLOCK_SIZE
    1024,    # exact BLOCK_SIZE
    2048,    # 2x BLOCK_SIZE
    4096,    # typical LLM intermediate dim
    4097,    # not a multiple of BLOCK_SIZE — exercises the column tail
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

class TestSiluAndMulTDCorrectness:
    """Tensor-descriptor kernel must match the original bit-for-bit."""

    @pytest.mark.parametrize("d", HALF_HIDDEN_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, d, dtype):
        torch.manual_seed(42)
        x = torch.randn(batch_size, 2 * d, device=device, dtype=dtype)

        out_original = silu_ref(x)
        out_td = silu_td(x)

        # Elementwise, same op order as original → bitwise-identical.
        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)


class TestSiluAndMulTDEdgeCases:
    """Edge cases for the tensor-descriptor kernel."""

    def test_minimum_size(self, device):
        """d=1: single output column per row (input width 2). The descriptor
        block_shape last dim is BLOCK_SIZE (>=16 bytes) regardless of d, so
        the 16-byte rule holds; the d=1 shape boundary zero-fills the rest."""
        torch.manual_seed(42)
        x = torch.randn(4, 2, device=device, dtype=torch.float32)

        out_original = silu_ref(x)
        out_td = silu_td(x)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)

    def test_single_row(self, device):
        """batch_size=1: only one row."""
        torch.manual_seed(42)
        x = torch.randn(1, 2 * 4096, device=device, dtype=torch.bfloat16)

        out_original = silu_ref(x)
        out_td = silu_td(x)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)

    @pytest.mark.parametrize("d", [127, 1023, 4097])
    def test_asymmetric_nondivisible(self, device, d):
        """d not a multiple of BLOCK_SIZE — column tail is partial."""
        torch.manual_seed(42)
        x = torch.randn(17, 2 * d, device=device, dtype=torch.float16)

        out_original = silu_ref(x)
        out_td = silu_td(x)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)

    def test_3d_input(self, device):
        """3D input (batch, seq_len, 2*d) — flattened to 2D by the wrapper."""
        torch.manual_seed(42)
        x = torch.randn(2, 8, 2 * 4096, device=device, dtype=torch.bfloat16)

        out_original = silu_ref(x)
        out_td = silu_td(x)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)

    def test_zero_input(self, device):
        """All-zero input: silu(0)=0, so output should be all zeros."""
        x = torch.zeros(4, 2 * 4096, device=device, dtype=torch.float32)

        out_original = silu_ref(x)
        out_td = silu_td(x)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)
        assert torch.all(out_td == 0)

    def test_large_values(self, device):
        """Large inputs exercise the clamp path on both gate and up."""
        torch.manual_seed(42)
        x = torch.randn(4, 2 * 1024, device=device, dtype=torch.float32) * 100.0

        out_original = silu_ref(x, limit=7.0)
        out_td = silu_td(x, limit=7.0)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)

    @pytest.mark.parametrize("limit", [1.0, 7.0, 100.0])
    def test_different_limits(self, device, limit):
        """Different clamp limits must match the original."""
        torch.manual_seed(42)
        x = torch.randn(8, 2 * 4096, device=device, dtype=torch.bfloat16)

        out_original = silu_ref(x, limit=limit)
        out_td = silu_td(x, limit=limit)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)

    def test_noncontiguous_rows(self, device):
        """Input row stride > 2*d: a column-slice of a wider tensor.

        full[:, :2*d] is a 2D view whose row stride is the full width, not 2*d.
        The wrapper calls .contiguous() on its reshape, so production always
        sees a packed buffer; this test confirms the td kernel matches the
        original on the same logical data.
        """
        torch.manual_seed(42)
        d = 2048
        full = torch.randn(32, 2 * d + 512, device=device, dtype=torch.bfloat16)
        x = full[:, : 2 * d]
        assert not x.is_contiguous()

        out_original = silu_ref(x)
        out_td = silu_td(x)

        torch.testing.assert_close(out_td, out_original, atol=0, rtol=0)
