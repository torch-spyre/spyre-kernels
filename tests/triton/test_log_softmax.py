# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for top-k log-softmax kernel.

Tests that the block-pointer kernel produces numerically identical results
to the original raw-pointer kernel across various shapes and dtypes.

Run: pytest kernels/vllm/log_softmax/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.vllm.log_softmax.wrapper import topk_log_softmax
from kernels.vllm.log_softmax.original import _topk_log_softmax_kernel
from kernels.vllm.log_softmax.block_ptr import _topk_log_softmax_kernel_block_ptr


def topk_log_softmax_reference(
    logits: torch.Tensor, topk_ids: torch.Tensor, topk: int
) -> torch.Tensor:
    """Pure PyTorch top-k log-softmax — the ground truth."""
    logits_f32 = logits.float()
    log_softmax = logits_f32 - logits_f32.logsumexp(dim=-1, keepdim=True)
    output = torch.gather(log_softmax, 1, topk_ids.long())
    return output


# ─── Test Parameters ───────────────────────────────────────────────

VOCAB_SIZES = [128, 1024, 4096, 32000, 32001]

BATCH_SIZES = [1, 4, 32]

TOPK_VALUES = [1, 5, 10, 16]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestLogSoftmaxEquivalence:
    """Block-pointer kernel must match raw-pointer kernel."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("topk", TOPK_VALUES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, vocab_size, topk, dtype):
        torch.manual_seed(42)
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
        topk_ids = torch.randint(0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64)

        out_raw = topk_log_softmax(logits, topk_ids, topk)
        out_blk = topk_log_softmax(logits, topk_ids, topk, kernel_fn=_topk_log_softmax_kernel_block_ptr)

        # Output is always float32. Block-pointer zero-padding vs raw-pointer
        # mask+other can cause tiny differences due to exp(0-max) vs not loading.
        torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("vocab_size", [4096, 32000])
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, vocab_size, dtype):
        """Both kernels should be close to PyTorch reference."""
        torch.manual_seed(42)
        batch_size = 8
        topk = 5
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
        topk_ids = torch.randint(0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64)

        ref = topk_log_softmax_reference(logits, topk_ids, topk)
        out_raw = topk_log_softmax(logits, topk_ids, topk)
        out_blk = topk_log_softmax(logits, topk_ids, topk, kernel_fn=_topk_log_softmax_kernel_block_ptr)

        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 1e-2, 1e-2

        torch.testing.assert_close(out_raw, ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(out_blk, ref, atol=atol, rtol=rtol)

    def test_topk_1(self, device):
        """Edge case: topk=1."""
        torch.manual_seed(42)
        logits = torch.randn(4, 32000, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, 32000, (4, 1), device=device, dtype=torch.int64)

        out_raw = topk_log_softmax(logits, topk_ids, 1)
        out_blk = topk_log_softmax(logits, topk_ids, 1, kernel_fn=_topk_log_softmax_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)

    def test_small_vocab(self, device):
        """Vocab smaller than BLOCK_SIZE."""
        torch.manual_seed(42)
        logits = torch.randn(4, 64, device=device, dtype=torch.float32)
        topk_ids = torch.randint(0, 64, (4, 5), device=device, dtype=torch.int64)

        out_raw = topk_log_softmax(logits, topk_ids, 5)
        out_blk = topk_log_softmax(logits, topk_ids, 5, kernel_fn=_topk_log_softmax_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)

    def test_uniform_logits(self, device):
        """All logits equal — log-softmax should be -log(vocab_size)."""
        vocab_size = 1024
        logits = torch.ones(4, vocab_size, device=device, dtype=torch.float32)
        topk_ids = torch.randint(0, vocab_size, (4, 5), device=device, dtype=torch.int64)

        out_raw = topk_log_softmax(logits, topk_ids, 5)
        out_blk = topk_log_softmax(logits, topk_ids, 5, kernel_fn=_topk_log_softmax_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)

        import math
        expected = -math.log(vocab_size)
        torch.testing.assert_close(
            out_blk, torch.full_like(out_blk, expected), atol=1e-4, rtol=1e-4
        )


class TestLogSoftmaxPerformance:
    """Block-pointer kernel should not regress significantly vs raw-pointer."""

    @pytest.mark.parametrize("vocab_size", [32000])
    def test_no_major_regression(self, device, vocab_size):
        import triton.testing

        batch_size = 32
        topk = 10
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64)

        ms_raw = triton.testing.do_bench(lambda: topk_log_softmax(logits, topk_ids, topk))
        ms_blk = triton.testing.do_bench(lambda: topk_log_softmax(logits, topk_ids, topk, kernel_fn=_topk_log_softmax_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"  vocab={vocab_size}: raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms, "
              f"slowdown={slowdown:.2f}x")

        assert slowdown < 2.0, (
            f"Block-pointer kernel is {slowdown:.2f}x slower than raw-pointer "
            f"(raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms)"
        )
