# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for merge_attn_states kernel.

Run: pytest kernels/merge_attn_states/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

import kernels.merge_attn_states  # noqa: F401

from kernels.merge_attn_states.wrapper import merge_attn_states
from kernels.merge_attn_states.original import merge_attn_states_kernel
from kernels.merge_attn_states.block_ptr import _merge_attn_states_kernel_block_ptr


def merge_attn_states_reference(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch merge — the ground truth."""
    num_tokens, num_heads, head_size = prefix_output.shape
    p_lse = prefix_lse.float()  # [num_heads, num_tokens]
    s_lse = suffix_lse.float()

    # Handle inf -> -inf (FA2 compat)
    p_lse = torch.where(p_lse == float("inf"), torch.tensor(float("-inf"), device=p_lse.device), p_lse)
    s_lse = torch.where(s_lse == float("inf"), torch.tensor(float("-inf"), device=s_lse.device), s_lse)

    max_lse = torch.maximum(p_lse, s_lse)
    p_se = torch.exp(p_lse - max_lse)
    s_se = torch.exp(s_lse - max_lse)
    out_se = p_se + s_se  # [num_heads, num_tokens]

    # Reshape for broadcasting: [num_tokens, num_heads, 1]
    p_scale = (p_se / out_se).T.unsqueeze(-1)
    s_scale = (s_se / out_se).T.unsqueeze(-1)

    output = prefix_output.float() * p_scale + suffix_output.float() * s_scale
    return output.to(prefix_output.dtype)


# ─── Test Parameters ───────────────────────────────────────────────

HEAD_SIZES = [32, 64, 128]
NUM_TOKENS_VALUES = [1, 4, 16, 32]
NUM_HEADS_VALUES = [1, 4, 8]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestMergeAttnStatesEquivalence:

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

        out_raw = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_blk = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse, kernel_fn=_merge_attn_states_kernel_block_ptr)

        if dtype == torch.float32:
            torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(out_raw, out_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("head_size", [64, 128])
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_correctness_vs_pytorch(self, device, head_size, dtype):
        torch.manual_seed(42)
        num_tokens, num_heads = 8, 4
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
        prefix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)
        suffix_lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        ref = merge_attn_states_reference(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_raw = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_blk = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse, kernel_fn=_merge_attn_states_kernel_block_ptr)

        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 1e-2, 1e-2
        torch.testing.assert_close(out_raw, ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(out_blk, ref, atol=atol, rtol=rtol)

    def test_equal_lse(self, device):
        """When both LSEs are equal, output should be average of prefix and suffix."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 4, 4, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        lse = torch.randn(num_heads, num_tokens, device=device, dtype=torch.float32)

        out_raw = merge_attn_states(prefix_output, lse, suffix_output, lse)
        out_blk = merge_attn_states(prefix_output, lse, suffix_output, lse, kernel_fn=_merge_attn_states_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)
        expected = (prefix_output + suffix_output) / 2.0
        torch.testing.assert_close(out_blk, expected, atol=1e-5, rtol=1e-5)

    def test_dominant_prefix(self, device):
        """When prefix_lse >> suffix_lse, output should be ~prefix_output."""
        torch.manual_seed(42)
        num_tokens, num_heads, head_size = 4, 4, 64
        prefix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        suffix_output = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        prefix_lse = torch.full((num_heads, num_tokens), 100.0, device=device)
        suffix_lse = torch.full((num_heads, num_tokens), -100.0, device=device)

        out_raw = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse)
        out_blk = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse, kernel_fn=_merge_attn_states_kernel_block_ptr)

        torch.testing.assert_close(out_raw, out_blk, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_blk, prefix_output, atol=1e-4, rtol=1e-4)
