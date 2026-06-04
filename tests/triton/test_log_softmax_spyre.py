# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the Spyre-aware top-k log-softmax kernel.

Compares _topk_log_softmax_kernel_spyre output against the original
raw-pointer kernel across various shapes, dtypes, and core counts.

Run: pytest tests/triton/test_log_softmax_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.log_softmax.spyre import _topk_log_softmax_kernel_spyre
from kernels.log_softmax.wrapper import topk_log_softmax as topk_log_softmax_original


# ─── Helpers ──────────────────────────────────────────────────────

def topk_log_softmax_spyre(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_cores: int = 32,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    assert logits.dim() == 2
    assert topk_ids.dim() == 2
    assert topk_ids.shape[1] == topk
    logits = logits.contiguous()
    topk_ids = topk_ids.contiguous()

    num_requests, vocab_size = logits.shape
    output = torch.empty(num_requests, topk, device=logits.device, dtype=torch.float32)
    PADDED_TOPK = triton.next_power_of_2(topk)
    grid = (num_cores,)
    _topk_log_softmax_kernel_spyre[grid](
        output,
        logits,
        topk_ids,
        num_requests,
        topk,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        PADDED_TOPK=PADDED_TOPK,
    )
    return output


# ─── Test Parameters ───────────────────────────────────────────────

VOCAB_SIZES = [128, 1024, 4096, 32000, 32001]

BATCH_SIZES = [1, 4, 32]

TOPK_VALUES = [1, 5, 10, 16]

CORE_COUNTS = [1, 4, 16, 32]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class TestLogSoftmaxSpyreCorrectness:
    """Spyre kernel must match original kernel."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("topk", TOPK_VALUES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_vs_original(self, device, batch_size, vocab_size, topk, dtype):
        torch.manual_seed(42)
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
        topk_ids = torch.randint(
            0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64
        )

        out_original = topk_log_softmax_original(logits, topk_ids, topk)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, topk)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_activation_uniform_logits(self, device):
        """All logits equal — log-softmax should be -log(vocab_size)."""
        import math

        vocab_size = 1024
        topk = 5
        logits = torch.ones(4, vocab_size, device=device, dtype=torch.float32)
        topk_ids = torch.randint(
            0, vocab_size, (4, topk), device=device, dtype=torch.int64
        )

        out_original = topk_log_softmax_original(logits, topk_ids, topk)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, topk)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

        expected = -math.log(vocab_size)
        torch.testing.assert_close(
            out_spyre, torch.full_like(out_spyre, expected), atol=1e-4, rtol=1e-4
        )


class TestLogSoftmaxSpyreDistribution:
    """Verify correctness across different core counts."""

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    def test_core_count_invariance(self, device, num_cores):
        """Result must match original regardless of core count."""
        torch.manual_seed(42)
        vocab_size = 32000
        batch_size = 8
        topk = 5
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(
            0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64
        )

        out_original = topk_log_softmax_original(logits, topk_ids, topk)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, topk, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    def test_more_cores_than_requests(self, device, num_cores):
        """Fewer requests than cores — some cores idle."""
        torch.manual_seed(42)
        vocab_size = 4096
        batch_size = 2
        topk = 3
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
        topk_ids = torch.randint(
            0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64
        )

        out_original = topk_log_softmax_original(logits, topk_ids, topk)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, topk, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)


class TestLogSoftmaxSpyreEdgeCases:
    """Edge cases for the Spyre kernel."""

    def test_topk_1(self, device):
        torch.manual_seed(42)
        logits = torch.randn(4, 32000, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, 32000, (4, 1), device=device, dtype=torch.int64)

        out_original = topk_log_softmax_original(logits, topk_ids, 1)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, 1)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_small_vocab(self, device):
        """Vocab smaller than BLOCK_SIZE."""
        torch.manual_seed(42)
        logits = torch.randn(4, 64, device=device, dtype=torch.float32)
        topk_ids = torch.randint(0, 64, (4, 5), device=device, dtype=torch.int64)

        out_original = topk_log_softmax_original(logits, topk_ids, 5)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, 5)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_single_request(self, device):
        """Single request with large vocab."""
        torch.manual_seed(42)
        logits = torch.randn(1, 32000, device=device, dtype=torch.float16)
        topk_ids = torch.randint(0, 32000, (1, 10), device=device, dtype=torch.int64)

        out_original = topk_log_softmax_original(logits, topk_ids, 10)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, 10)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_vocab_not_divisible_by_block(self, device):
        """Vocab size not a multiple of BLOCK_SIZE."""
        torch.manual_seed(42)
        logits = torch.randn(8, 32001, device=device, dtype=torch.float32)
        topk_ids = torch.randint(0, 32001, (8, 5), device=device, dtype=torch.int64)

        out_original = topk_log_softmax_original(logits, topk_ids, 5)
        out_spyre = topk_log_softmax_spyre(logits, topk_ids, 5)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)
