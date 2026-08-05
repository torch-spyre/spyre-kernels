# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for embedding kernel.

Tests that the block-pointer kernel produces identical results
to the original raw-pointer kernel across various shapes and dtypes.

Run: pytest tests/triton/test_embedding.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.vllm.embedding.wrapper import embedding
from kernels.vllm.embedding.original import embedding_forward_kernel
from kernels.vllm.embedding.block_ptr import embedding_forward_kernel_block_ptr


# ─── Test Parameters ───────────────────────────────────────────────

VOCAB_SIZES = [128, 1024, 32000, 32001]
EMBEDDING_DIMS = [64, 128, 1024, 4096]
N_TOKENS = [1, 4, 32, 128]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestEmbeddingEquivalence:
    """Block-pointer kernel must match raw-pointer kernel exactly."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("embedding_dim", EMBEDDING_DIMS)
    @pytest.mark.parametrize("n_tokens", N_TOKENS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, vocab_size, embedding_dim, n_tokens, dtype):
        torch.manual_seed(42)
        table = torch.randn(vocab_size, embedding_dim, device=device, dtype=dtype)
        indices = torch.randint(0, vocab_size, (n_tokens,), device=device, dtype=torch.int64)

        out_raw = embedding(table, indices)
        out_blk = embedding(table, indices, kernel_fn=embedding_forward_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, dtype):
        """Both kernels should match F.embedding reference."""
        torch.manual_seed(42)
        vocab_size, embedding_dim, n_tokens = 1024, 256, 64
        table = torch.randn(vocab_size, embedding_dim, device=device, dtype=dtype)
        indices = torch.randint(0, vocab_size, (n_tokens,), device=device, dtype=torch.int64)

        ref = torch.nn.functional.embedding(indices, table)
        out_raw = embedding(table, indices)
        out_blk = embedding(table, indices, kernel_fn=embedding_forward_kernel_block_ptr)

        torch.testing.assert_close(out_raw, ref, atol=0, rtol=0)
        torch.testing.assert_close(out_blk, ref, atol=0, rtol=0)

    def test_multidim_indices(self, device):
        """Indices tensor with shape [B, S] should return [B, S, D]."""
        torch.manual_seed(42)
        table = torch.randn(1024, 256, device=device, dtype=torch.float16)
        indices = torch.randint(0, 1024, (4, 16), device=device, dtype=torch.int64)

        out_blk = embedding(table, indices, kernel_fn=embedding_forward_kernel_block_ptr)

        assert out_blk.shape == (4, 16, 256)
        ref = torch.nn.functional.embedding(indices, table)
        torch.testing.assert_close(out_blk, ref, atol=0, rtol=0)

    def test_single_token(self, device):
        table = torch.randn(32000, 768, device=device, dtype=torch.bfloat16)
        indices = torch.tensor([12345], device=device, dtype=torch.int64)

        out_raw = embedding(table, indices)
        out_blk = embedding(table, indices, kernel_fn=embedding_forward_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    def test_repeated_indices(self, device):
        """Same index repeated — all output rows should be identical."""
        torch.manual_seed(42)
        table = torch.randn(512, 128, device=device, dtype=torch.float16)
        indices = torch.full((8,), 42, device=device, dtype=torch.int64)

        out_blk = embedding(table, indices, kernel_fn=embedding_forward_kernel_block_ptr)

        for i in range(8):
            torch.testing.assert_close(out_blk[i], table[42], atol=0, rtol=0)


class TestEmbeddingPerformance:
    """Block-pointer kernel should not regress significantly vs raw-pointer."""

    def test_no_major_regression(self, device):
        import triton.testing

        vocab_size, embedding_dim, n_tokens = 32000, 4096, 512
        table = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.bfloat16)
        indices = torch.randint(0, vocab_size, (n_tokens,), device=device, dtype=torch.int64)

        ms_raw = triton.testing.do_bench(lambda: embedding(table, indices))
        ms_blk = triton.testing.do_bench(
            lambda: embedding(table, indices, kernel_fn=embedding_forward_kernel_block_ptr)
        )

        slowdown = ms_blk / ms_raw
        print(f"  raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms, slowdown={slowdown:.2f}x")

        assert slowdown < 2.0, (
            f"Block-pointer kernel is {slowdown:.2f}x slower than raw-pointer "
            f"(raw={ms_raw:.3f}ms, block_ptr={ms_blk:.3f}ms)"
        )
