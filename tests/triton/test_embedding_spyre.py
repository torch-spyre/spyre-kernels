# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the Spyre-aware embedding kernel.

Compares embedding_forward_kernel_spyre output against the original kernel
across various shapes and core counts.

Run: pytest tests/triton/test_embedding_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.embedding.spyre import embedding_forward_kernel_spyre
from kernels.embedding.wrapper import embedding as embedding_original


# ─── Helpers ──────────────────────────────────────────────────────


def embedding_spyre(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    num_cores: int = 32,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
) -> torch.Tensor:
    """Launch the Spyre embedding kernel with a fixed grid."""
    assert embeddings.is_contiguous()
    assert indices.is_contiguous()
    assert embeddings.dim() == 2

    ori_shape = indices.shape
    indices_flat = indices.view(-1)
    n_elements = indices_flat.numel()
    vocab_size = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]

    BLOCK_SIZE_M = triton.next_power_of_2(min(BLOCK_SIZE_M, embedding_dim))
    BLOCK_SIZE_N = triton.next_power_of_2(min(BLOCK_SIZE_N, embedding_dim))

    output = torch.empty(
        n_elements, embedding_dim,
        device=indices.device, dtype=embeddings.dtype,
    )

    grid = (num_cores,)
    embedding_forward_kernel_spyre[grid](
        embeddings,
        indices_flat,
        output,
        n_elements,
        vocab_size,
        embedding_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output.view(*ori_shape, embedding_dim)


# ─── Test Parameters ───────────────────────────────────────────────

VOCAB_SIZES = [128, 1024, 32000, 32001]
EMBEDDING_DIMS = [64, 128, 1024, 4096]
N_TOKENS = [1, 4, 32, 128]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

CORE_COUNTS = [1, 4, 16, 32]


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────


class TestEmbeddingSpyreCorrectness:
    """Spyre kernel must match original kernel."""

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("embedding_dim", EMBEDDING_DIMS)
    @pytest.mark.parametrize("n_tokens", N_TOKENS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, vocab_size, embedding_dim, n_tokens, dtype):
        torch.manual_seed(42)
        table = torch.randn(vocab_size, embedding_dim, device=device, dtype=dtype)
        indices = torch.randint(0, vocab_size, (n_tokens,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, dtype):
        """Spyre kernel should match F.embedding reference."""
        torch.manual_seed(42)
        vocab_size, embedding_dim, n_tokens = 1024, 256, 64
        table = torch.randn(vocab_size, embedding_dim, device=device, dtype=dtype)
        indices = torch.randint(0, vocab_size, (n_tokens,), device=device, dtype=torch.int64)

        ref = torch.nn.functional.embedding(indices, table)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, ref, atol=0, rtol=0)

    def test_multidim_indices(self, device):
        """Indices tensor with shape [B, S] should return [B, S, D]."""
        torch.manual_seed(42)
        table = torch.randn(1024, 256, device=device, dtype=torch.float16)
        indices = torch.randint(0, 1024, (4, 16), device=device, dtype=torch.int64)

        out_spyre = embedding_spyre(table, indices)

        assert out_spyre.shape == (4, 16, 256)
        ref = torch.nn.functional.embedding(indices, table)
        torch.testing.assert_close(out_spyre, ref, atol=0, rtol=0)


class TestEmbeddingSpyreDistribution:
    """Verify correctness across different core counts."""

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_core_count_invariance(self, device, num_cores, dtype):
        """Result must be identical regardless of core count."""
        torch.manual_seed(42)
        vocab_size, embedding_dim, n_tokens = 1024, 512, 64
        table = torch.randn(vocab_size, embedding_dim, device=device, dtype=dtype)
        indices = torch.randint(0, vocab_size, (n_tokens,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_more_cores_than_work(self, device):
        """When cores exceed work items, extra cores should idle gracefully."""
        torch.manual_seed(42)
        table = torch.randn(256, 64, device=device, dtype=torch.float32)
        indices = torch.randint(0, 256, (2,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices, num_cores=32)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_single_core(self, device):
        """Single core must process all work sequentially."""
        torch.manual_seed(42)
        table = torch.randn(32000, 1024, device=device, dtype=torch.bfloat16)
        indices = torch.randint(0, 32000, (128,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices, num_cores=1)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)


class TestEmbeddingSpyreEdgeCases:
    """Edge cases for the Spyre kernel."""

    def test_single_token(self, device):
        """Minimum batch: single token lookup."""
        torch.manual_seed(42)
        table = torch.randn(32000, 768, device=device, dtype=torch.bfloat16)
        indices = torch.tensor([12345], device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_non_power_of_two_embedding_dim(self, device):
        """Embedding dim not a multiple of BLOCK_SIZE."""
        torch.manual_seed(42)
        table = torch.randn(512, 300, device=device, dtype=torch.float32)
        indices = torch.randint(0, 512, (16,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_non_divisible_n_tokens(self, device):
        """n_tokens not a multiple of BLOCK_SIZE_M."""
        torch.manual_seed(42)
        table = torch.randn(1024, 256, device=device, dtype=torch.float16)
        indices = torch.randint(0, 1024, (7,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_small_vocab(self, device):
        """Vocab smaller than typical BLOCK sizes."""
        torch.manual_seed(42)
        table = torch.randn(10, 64, device=device, dtype=torch.float32)
        indices = torch.randint(0, 10, (32,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_asymmetric_large_vocab_small_dim(self, device):
        """Many vocab entries with small embedding dimension."""
        torch.manual_seed(42)
        table = torch.randn(50000, 32, device=device, dtype=torch.bfloat16)
        indices = torch.randint(0, 50000, (64,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_asymmetric_small_vocab_large_dim(self, device):
        """Few vocab entries with large embedding dimension."""
        torch.manual_seed(42)
        table = torch.randn(16, 4096, device=device, dtype=torch.float32)
        indices = torch.randint(0, 16, (8,), device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)

    def test_repeated_indices(self, device):
        """Same index repeated — all output rows should be identical."""
        torch.manual_seed(42)
        table = torch.randn(512, 128, device=device, dtype=torch.float16)
        indices = torch.full((8,), 42, device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)
        for i in range(8):
            torch.testing.assert_close(out_spyre[i], table[42], atol=0, rtol=0)

    def test_last_vocab_entry(self, device):
        """Accessing the very last row of the embedding table."""
        torch.manual_seed(42)
        vocab_size = 32001
        table = torch.randn(vocab_size, 256, device=device, dtype=torch.float32)
        indices = torch.tensor([vocab_size - 1], device=device, dtype=torch.int64)

        out_original = embedding_original(table, indices)
        out_spyre = embedding_spyre(table, indices)

        torch.testing.assert_close(out_spyre, out_original, atol=0, rtol=0)
