# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for reshape_and_cache kernel.

Run: pytest kernels/reshape_and_cache/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.reshape_and_cache.wrapper import reshape_and_cache
from kernels.reshape_and_cache.original import reshape_and_cache_kernel_flash
from kernels.reshape_and_cache.block_ptr import _reshape_and_cache_kernel_block_ptr


def reshape_and_cache_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Pure PyTorch reshape_and_cache — the ground truth."""
    block_size = key_cache.shape[1]
    for i in range(key.shape[0]):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        key_cache[block_idx, block_offset] = key[i]
        value_cache[block_idx, block_offset] = value[i]


# ─── Test Parameters ───────────────────────────────────────────────

NUM_TOKENS_VALUES = [1, 4, 16, 32]
NUM_HEADS_VALUES = [4, 8]
HEAD_SIZE_VALUES = [64, 128]
BLOCK_SIZE_VALUES = [16]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_inputs(num_tokens, num_heads, head_size, block_size, num_blocks, device, dtype):
    torch.manual_seed(42)
    key = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    value = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device=device, dtype=dtype)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, device=device, dtype=dtype)
    max_slots = num_blocks * block_size
    slot_mapping = torch.randint(0, max_slots, (num_tokens,), device=device, dtype=torch.int64)
    # Ensure unique slots
    slot_mapping = torch.unique(slot_mapping)[:num_tokens]
    if len(slot_mapping) < num_tokens:
        slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    return key, value, key_cache, value_cache, slot_mapping


class TestReshapeAndCacheEquivalence:

    @pytest.mark.parametrize("num_tokens", NUM_TOKENS_VALUES)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_VALUES)
    @pytest.mark.parametrize("head_size", HEAD_SIZE_VALUES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_numerical_equivalence(self, device, num_tokens, num_heads, head_size, dtype):
        block_size = 16
        num_blocks = 8
        key, value, kc_raw, vc_raw, slot_mapping = _make_inputs(
            num_tokens, num_heads, head_size, block_size, num_blocks, device, dtype
        )
        kc_blk = kc_raw.clone()
        vc_blk = vc_raw.clone()

        reshape_and_cache(key, value, kc_raw, vc_raw, slot_mapping)
        reshape_and_cache(key, value, kc_blk, vc_blk, slot_mapping, kernel_fn=_reshape_and_cache_kernel_block_ptr)

        torch.testing.assert_close(kc_raw, kc_blk, atol=0, rtol=0)
        torch.testing.assert_close(vc_raw, vc_blk, atol=0, rtol=0)

    @pytest.mark.parametrize("head_size", [64, 128])
    def test_correctness_vs_pytorch(self, device, head_size):
        num_tokens, num_heads, block_size, num_blocks = 16, 4, 16, 8
        dtype = torch.bfloat16
        key, value, kc_ref, vc_ref, slot_mapping = _make_inputs(
            num_tokens, num_heads, head_size, block_size, num_blocks, device, dtype
        )
        kc_raw = kc_ref.clone()
        vc_raw = vc_ref.clone()
        kc_blk = kc_ref.clone()
        vc_blk = vc_ref.clone()

        reshape_and_cache_reference(key, value, kc_ref, vc_ref, slot_mapping)
        reshape_and_cache(key, value, kc_raw, vc_raw, slot_mapping)
        reshape_and_cache(key, value, kc_blk, vc_blk, slot_mapping, kernel_fn=_reshape_and_cache_kernel_block_ptr)

        torch.testing.assert_close(kc_raw, kc_ref, atol=0, rtol=0)
        torch.testing.assert_close(kc_blk, kc_ref, atol=0, rtol=0)
        torch.testing.assert_close(vc_raw, vc_ref, atol=0, rtol=0)
        torch.testing.assert_close(vc_blk, vc_ref, atol=0, rtol=0)

    def test_negative_slots_skipped(self, device):
        """Tokens with slot_mapping < 0 should not be written."""
        num_tokens, num_heads, head_size, block_size, num_blocks = 8, 4, 64, 16, 4
        key, value, kc_raw, vc_raw, slot_mapping = _make_inputs(
            num_tokens, num_heads, head_size, block_size, num_blocks, device, torch.bfloat16
        )
        # Set some slots to -1
        slot_mapping[0] = -1
        slot_mapping[3] = -1

        kc_blk = kc_raw.clone()
        vc_blk = vc_raw.clone()

        reshape_and_cache(key, value, kc_raw, vc_raw, slot_mapping)
        reshape_and_cache(key, value, kc_blk, vc_blk, slot_mapping, kernel_fn=_reshape_and_cache_kernel_block_ptr)

        torch.testing.assert_close(kc_raw, kc_blk, atol=0, rtol=0)
        torch.testing.assert_close(vc_raw, vc_blk, atol=0, rtol=0)

    def test_single_token(self, device):
        """Edge case: single token."""
        key, value, kc_raw, vc_raw, slot_mapping = _make_inputs(
            1, 8, 64, 16, 4, device, torch.float32
        )
        kc_blk = kc_raw.clone()
        vc_blk = vc_raw.clone()

        reshape_and_cache(key, value, kc_raw, vc_raw, slot_mapping)
        reshape_and_cache(key, value, kc_blk, vc_blk, slot_mapping, kernel_fn=_reshape_and_cache_kernel_block_ptr)

        torch.testing.assert_close(kc_raw, kc_blk, atol=0, rtol=0)
        torch.testing.assert_close(vc_raw, vc_blk, atol=0, rtol=0)
