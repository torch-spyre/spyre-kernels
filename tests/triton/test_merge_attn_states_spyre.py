# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the Spyre-aware merge_attn_states kernel.

Compares merge_attn_states_kernel_spyre output against the original kernel
across various shapes and core counts.

Run: pytest tests/triton/test_merge_attn_states_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.merge_attn_states.spyre import merge_attn_states_kernel_spyre
from kernels.merge_attn_states.wrapper import merge_attn_states as merge_attn_states_original


# ─── Helpers ──────────────────────────────────────────────────────


def merge_attn_states_spyre(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    num_cores: int = 32,
    BLOCK_HEAD: int = 128,
) -> torch.Tensor:
    """Launch the Spyre kernel with a fixed grid."""
    assert prefix_output.is_contiguous()
    assert suffix_output.is_contiguous()
    assert prefix_lse.is_contiguous()
    assert suffix_lse.is_contiguous()

    num_tokens, num_heads, head_size = prefix_output.shape
    BLOCK_HEAD = triton.next_power_of_2(min(BLOCK_HEAD, head_size))

    output = torch.empty_like(prefix_output)

    grid = (num_cores,)
    merge_attn_states_kernel_spyre[grid](
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        num_tokens,
        num_heads,
        head_size,
        BLOCK_HEAD=BLOCK_HEAD,
    )
    return output


# ─── Test Parameters ───────────────────────────────────────────────

HEAD_SIZES = [32, 64, 128]
NUM_TOKENS_VALUES = [1, 4, 16, 32]
NUM_HEADS_VALUES = [1, 4, 8]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

CORE_COUNTS = [1, 4, 16, 32]


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────


class TestMergeAttnStatesSpyreCorrectness:
    """Spyre kernel must match original kernel."""

    @pytest.mark.parametrize("head_size", HEAD_SIZES)
    @pytest.mark.parametrize("num_tokens", NUM_TOKENS_VALUES)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_VALUES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, num_tokens, num_heads, head_size, dtype):
        torch.manual_seed(42)
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        if dtype == torch.float32:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    @pytest.mark.parametrize("head_size", [64, 128])
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_non_power_of_two_heads(self, device, head_size, dtype):
        """Non-power-of-two num_heads to verify distribution loop."""
        torch.manual_seed(42)
        num_tokens, num_heads = 8, 3
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        if dtype == torch.float32:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)


class TestMergeAttnStatesSpyreDistribution:
    """Verify correctness across different core counts."""

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_core_count_invariance(self, device, num_cores, dtype):
        """Result must match original regardless of core count."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 16, 8, 128
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(
            prefix_output, prefix_lse, suffix_output, suffix_lse, num_cores=num_cores
        )

        if dtype == torch.float32:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    def test_more_cores_than_work(self, device):
        """When cores exceed (tokens * heads) pairs, extra cores should idle gracefully."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 2, 2, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(
            prefix_output, prefix_lse, suffix_output, suffix_lse, num_cores=32
        )

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_single_core(self, device):
        """Single core must process all work sequentially."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 32, 8, 128
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(
            prefix_output, prefix_lse, suffix_output, suffix_lse, num_cores=1
        )

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)


class TestMergeAttnStatesSpyreEdgeCases:
    """Edge cases for the Spyre kernel."""

    def test_single_token_single_head(self, device):
        """Minimum size: one token, one head."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 1, 1, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_non_power_of_two_head_size(self, device):
        """Head size not a power of two (not divisible by BLOCK_HEAD)."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 8, 4, 80
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_equal_lse(self, device):
        """When both LSEs are equal, output should be average of prefix and suffix."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 4, 4, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_spyre = merge_attn_states_spyre(prefix_output, lse, suffix_output, lse)

        expected = (prefix_output + suffix_output) / 2.0
        torch.testing.assert_close(out_spyre, expected, atol=1e-5, rtol=1e-5)

    def test_dominant_prefix(self, device):
        """When prefix_lse >> suffix_lse, output should be ~prefix_output."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 4, 4, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.full((num_heads, num_tokens), 100.0, device=device)
        suffix_lse = torch.full((num_heads, num_tokens), -100.0, device=device)

        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        torch.testing.assert_close(out_spyre, prefix_output, atol=1e-4, rtol=1e-4)

    def test_dominant_suffix(self, device):
        """When suffix_lse >> prefix_lse, output should be ~suffix_output."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 4, 4, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.full((num_heads, num_tokens), -100.0, device=device)
        suffix_lse = torch.full((num_heads, num_tokens), 100.0, device=device)

        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        torch.testing.assert_close(out_spyre, suffix_output, atol=1e-4, rtol=1e-4)

    def test_inf_lse_handling(self, device):
        """FA2 inf LSE should be converted to -inf (making that side contribute 0)."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 4, 4, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.full((num_heads, num_tokens), float("inf"), device=device)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_large_head_size(self, device):
        """Large head_size requiring multiple BLOCK_HEAD tiles."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 4, 2, 512
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(
            prefix_output, prefix_lse, suffix_output, suffix_lse, BLOCK_HEAD=128
        )

        torch.testing.assert_close(out_spyre, out_original, atol=1e-2, rtol=0)

    def test_many_tokens_few_heads(self, device):
        """Asymmetric: many tokens, few heads."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 128, 1, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)

    def test_few_tokens_many_heads(self, device):
        """Asymmetric: few tokens, many heads."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 2, 32, 128
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_original = merge_attn_states_original(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_spyre = merge_attn_states_spyre(prefix_output, prefix_lse, suffix_output, suffix_lse)

        torch.testing.assert_close(out_spyre, out_original, atol=1e-5, rtol=1e-5)
