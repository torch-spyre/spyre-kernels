# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the tensor-descriptor matmul kernel.

Compares _matmul_kernel_td output against the original kernel across various
shapes. Both kernels are launched through kernels/vllm/matmul/wrapper.py (via its
kernel_fn= dispatch) — no forked launch path.

Tolerances: both kernels autotune independently. If they pick different
BLOCK_SIZE_K the f32 partial-sum order differs (float add isn't associative),
so we assert a small tolerance rather than atol=0.

Run: pytest tests/triton/test_matmul_td.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch

from kernels.vllm.matmul.tensor_descriptor import _matmul_kernel_td
from kernels.vllm.matmul.wrapper import matmul

# Kernel-vs-kernel tolerance (see module docstring). Both cast to f16 at the
# end, so the only gap is f32 accumulation order rounding into f16 — a couple
# of f16 ULPs at most (f16 mantissa ~1e-3 relative).
ATOL, RTOL = 1e-3, 1e-3


# ─── Launch helpers (both go through the wrapper) ──────────────────

def matmul_ref(a, b, **kwargs):
    """Reference: original kernel via the wrapper."""
    return matmul(a, b, **kwargs)


def matmul_td(a, b, **kwargs):
    """Tensor-descriptor kernel via the wrapper's kernel_fn dispatch."""
    return matmul(a, b, kernel_fn=_matmul_kernel_td, **kwargs)


# ─── Test Parameters ───────────────────────────────────────────────

MATRIX_SHAPES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (128, 256, 512),
    (256, 128, 64),
    (513, 257, 129),   # non-power-of-2 in all dims — tests OOB on M, N, K
    (127, 129, 65),    # all dims below/around a single small tile
    (1, 1, 64),         # minimum M, N
    (64, 64, 1),        # minimum K (single-element K reduction)
]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestMatmulTDCorrectness:
    """Tensor-descriptor kernel must match the original kernel."""

    @pytest.mark.parametrize("M,N,K", MATRIX_SHAPES)
    def test_numerical_equivalence(self, device, M, N, K):
        torch.manual_seed(42)
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("M,N,K", MATRIX_SHAPES)
    def test_correctness_vs_pytorch(self, device, M, N, K):
        """The td kernel should also be close to the PyTorch reference."""
        torch.manual_seed(42)
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        ref = torch.matmul(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, ref, atol=5e-2, rtol=1e-2)

    def test_activation_leaky_relu(self, device):
        """leaky_relu activation must match the original."""
        torch.manual_seed(42)
        M, N, K = 256, 256, 256
        a = torch.randn((M, K), device=device, dtype=torch.float16)
        b = torch.randn((K, N), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b, activation="leaky_relu")
        out_td = matmul_td(a, b, activation="leaky_relu")

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)


class TestMatmulTDEdgeCases:
    """Edge cases for the tensor-descriptor kernel."""

    def test_rectangular_tall(self, device):
        """Tall matrix: M >> N."""
        torch.manual_seed(42)
        a = torch.randn((2048, 128), device=device, dtype=torch.float16)
        b = torch.randn((128, 64), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)

    def test_rectangular_wide(self, device):
        """Wide matrix: N >> M."""
        torch.manual_seed(42)
        a = torch.randn((64, 128), device=device, dtype=torch.float16)
        b = torch.randn((128, 2048), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)

    def test_nondivisible_all_dims(self, device):
        """Every dim non-divisible — exercises OOB zero-fill on M, N and K tail."""
        torch.manual_seed(42)
        a = torch.randn((100, 70), device=device, dtype=torch.float16)
        b = torch.randn((70, 90), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)

    def test_minimum_size(self, device):
        """1x1 output from a single-element contraction."""
        torch.manual_seed(42)
        a = torch.randn((1, 1), device=device, dtype=torch.float16)
        b = torch.randn((1, 1), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)

    def test_large_k_many_tiles(self, device):
        """Large K with a non-divisible tail — many K tiles plus a partial one."""
        torch.manual_seed(42)
        a = torch.randn((128, 4097), device=device, dtype=torch.float16)
        b = torch.randn((4097, 128), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)

    def test_large_values(self, device):
        """Large inputs — tests stability of the f32 accumulation path."""
        torch.manual_seed(42)
        a = torch.randn((256, 256), device=device, dtype=torch.float16) * 100.0
        b = torch.randn((256, 256), device=device, dtype=torch.float16) * 100.0

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        # Inputs scaled x100 -> outputs in the thousands, so the absolute
        # config-divergence gap scales up too. rtol still governs; atol is
        # raised off the unit-scale floor so it isn't the binding constraint.
        torch.testing.assert_close(out_td, out_original, atol=1e0, rtol=RTOL, equal_nan=True)

    def test_zero_input(self, device):
        """All-zero input produces all-zero output."""
        a = torch.zeros((128, 128), device=device, dtype=torch.float16)
        b = torch.zeros((128, 128), device=device, dtype=torch.float16)

        out_original = matmul_ref(a, b)
        out_td = matmul_td(a, b)

        torch.testing.assert_close(out_td, out_original, atol=ATOL, rtol=RTOL)
