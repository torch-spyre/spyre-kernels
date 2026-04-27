# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for ranks kernel.

Tests that the block-pointer kernel produces identical results
to the original raw-pointer kernel across various shapes and dtypes.

Run: pytest kernels/ranks/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.ranks.wrapper import ranks
from kernels.ranks.original import _ranks_kernel
from kernels.ranks.block_ptr import _ranks_kernel_block_ptr


def ranks_reference(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch ranks — the ground truth."""
    num_requests = logits.shape[0]
    output = torch.empty(num_requests, device=logits.device, dtype=torch.int32)
    for i in range(num_requests):
        ref_logit = logits[i, token_ids[i]]
        output[i] = (logits[i] >= ref_logit).sum().to(torch.int32)
    return output


# ─── Test Parameters ───────────────────────────────────────────────

VOCAB_SIZES = [128, 1024, 4096, 32000, 32001]

BATCH_SIZES = [1, 4, 32]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestRanksEquivalence:
    """Block-pointer kernel must match raw-pointer kernel exactly."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, batch_size, vocab_size, dtype):
        torch.manual_seed(42)
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
        token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)

        out_raw = ranks(logits, token_ids)
        out_blk = ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, vocab_size, dtype):
        """Both kernels should match PyTorch reference."""
        torch.manual_seed(42)
        batch_size = 16
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
        token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)

        ref = ranks_reference(logits, token_ids)
        out_raw = ranks(logits, token_ids)
        out_blk = ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr)

        torch.testing.assert_close(out_raw, ref, atol=0, rtol=0)
        torch.testing.assert_close(out_blk, ref, atol=0, rtol=0)

    def test_negative_logits(self, device):
        """When ref logit is negative, OOB padding must not falsely count."""
        torch.manual_seed(42)
        logits = torch.randn(8, 4097, device=device, dtype=torch.float32) - 5.0
        token_ids = torch.randint(0, 4097, (8,), device=device, dtype=torch.int64)

        out_raw = ranks(logits, token_ids)
        out_blk = ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    def test_all_same_logits(self, device):
        """All logits identical — rank should equal vocab_size."""
        vocab_size = 2048
        logits = torch.ones(4, vocab_size, device=device, dtype=torch.float32)
        token_ids = torch.zeros(4, device=device, dtype=torch.int64)

        out_raw = ranks(logits, token_ids)
        out_blk = ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)
        assert torch.all(out_blk == vocab_size)

    def test_single_request(self, device):
        """Edge case: single request."""
        logits = torch.randn(1, 32000, device=device, dtype=torch.bfloat16)
        token_ids = torch.tensor([15000], device=device, dtype=torch.int64)

        out_raw = ranks(logits, token_ids)
        out_blk = ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)


class TestRanksPerformance:
    """Block-pointer kernel should not regress significantly vs raw-pointer."""

    @pytest.mark.parametrize("vocab_size", [32000])
    def test_no_major_regression(self, device, vocab_size):
        import triton.testing

        batch_size = 64
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)

        ms_raw = triton.testing.do_bench(lambda: ranks(logits, token_ids))
        ms_blk = triton.testing.do_bench(lambda: ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"  vocab={vocab_size}: raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms, "
              f"slowdown={slowdown:.2f}x")

        assert slowdown < 2.0, (
            f"Block-pointer kernel is {slowdown:.2f}x slower than raw-pointer "
            f"(raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms)"
        )
