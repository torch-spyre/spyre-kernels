# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 Validation: GPU equivalence tests for decode_softmax_reducev kernel.

Tests that the block-pointer stage 2 kernel produces numerically identical
results to the original raw-pointer kernel.

Run: pytest kernels/vllm/decode_softmax_reducev/test_equivalence.py -v
Requires: GPU with triton support
"""

import pytest
import torch

from kernels.vllm.decode_softmax_reducev.wrapper import decode_softmax_reducev
from kernels.vllm.decode_softmax_reducev.original import _fwd_kernel_stage2
from kernels.vllm.decode_softmax_reducev.block_ptr import _fwd_kernel_stage2_block_ptr


def decode_softmax_reducev_reference(
    mid_o: torch.Tensor, b_seq_len: torch.Tensor, num_kv_splits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch online softmax merge — the ground truth."""
    batch, heads, splits, lv_plus_1 = mid_o.shape
    Lv = lv_plus_1 - 1

    o = torch.empty(batch, heads, Lv, device=mid_o.device, dtype=mid_o.dtype)
    lse = torch.empty(batch, heads, device=mid_o.device, dtype=torch.float32)

    for b in range(batch):
        seq_len = b_seq_len[b].item()
        for h in range(heads):
            e_max = float("-inf")
            e_sum = 0.0
            acc = torch.zeros(Lv, dtype=torch.float32, device=mid_o.device)

            for s in range(num_kv_splits):
                kv_len_per_split = (seq_len + num_kv_splits - 1) // num_kv_splits
                split_start = kv_len_per_split * s
                split_end = min(split_start + kv_len_per_split, seq_len)
                if split_end <= split_start:
                    continue

                tv = mid_o[b, h, s, :Lv].float()
                tlogic = mid_o[b, h, s, Lv].float().item()

                n_e_max = max(tlogic, e_max)
                old_scale = torch.exp(torch.tensor(e_max - n_e_max))
                acc = acc * old_scale
                exp_logic = torch.exp(torch.tensor(tlogic - n_e_max))
                acc = acc + exp_logic * tv
                e_sum = e_sum * old_scale.item() + exp_logic.item()
                e_max = n_e_max

            o[b, h] = (acc / e_sum).to(mid_o.dtype)
            lse[b, h] = e_max + torch.log(torch.tensor(e_sum))

    return o, lse


# ─── Test Parameters ───────────────────────────────────────────────

LV_VALUES = [32, 64, 128]
BATCH_SIZES = [1, 4, 16]
HEAD_COUNTS = [1, 4, 8]
NUM_KV_SPLITS_VALUES = [2, 4, 8]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device):
    """Create a realistic Mid_O tensor with partial V outputs and LSE values."""
    mid_o = torch.randn(batch, heads, num_kv_splits, Lv + 1, device=device, dtype=torch.float32)
    # Make the LSE values (last element) realistic: log of small positive numbers
    mid_o[:, :, :, Lv] = torch.randn(batch, heads, num_kv_splits, device=device) * 2.0
    return mid_o


# ─── Tests ─────────────────────────────────────────────────────────

class TestDecodeSoftmaxReducevEquivalence:
    """Block-pointer kernel must match raw-pointer kernel."""

    @pytest.mark.parametrize("Lv", LV_VALUES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    @pytest.mark.parametrize("num_kv_splits", NUM_KV_SPLITS_VALUES)
    def test_numerical_equivalence(self, device, batch_size, heads, Lv, num_kv_splits):
        torch.manual_seed(42)
        seq_lens = torch.randint(num_kv_splits, 512, (batch_size,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch_size, heads, num_kv_splits, Lv, seq_lens, device)

        o_raw, lse_raw = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_blk, lse_blk = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits, kernel_fn=_fwd_kernel_stage2_block_ptr)

        torch.testing.assert_close(o_raw, o_blk, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_raw, lse_blk, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("Lv", [64, 128])
    def test_correctness_vs_pytorch(self, device, Lv):
        """Both kernels should match PyTorch reference."""
        torch.manual_seed(42)
        batch_size, heads, num_kv_splits = 4, 4, 4
        seq_lens = torch.randint(num_kv_splits, 256, (batch_size,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch_size, heads, num_kv_splits, Lv, seq_lens, device)

        ref_o, ref_lse = decode_softmax_reducev_reference(mid_o, seq_lens, num_kv_splits)
        o_raw, lse_raw = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_blk, lse_blk = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits, kernel_fn=_fwd_kernel_stage2_block_ptr)

        torch.testing.assert_close(o_raw, ref_o, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(o_blk, ref_o, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lse_raw, ref_lse, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lse_blk, ref_lse, atol=1e-4, rtol=1e-4)

    def test_single_split(self, device):
        """Edge case: num_kv_splits=1 — just pass through."""
        torch.manual_seed(42)
        batch, heads, Lv = 4, 4, 64
        seq_lens = torch.full((batch,), 128, device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, 1, Lv, seq_lens, device)

        o_raw, lse_raw = decode_softmax_reducev(mid_o, seq_lens, 1)
        o_blk, lse_blk = decode_softmax_reducev(mid_o, seq_lens, 1, kernel_fn=_fwd_kernel_stage2_block_ptr)

        torch.testing.assert_close(o_raw, o_blk, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_raw, lse_blk, atol=1e-5, rtol=1e-5)

    def test_short_sequences(self, device):
        """Some splits may be empty when seq_len < num_kv_splits."""
        torch.manual_seed(42)
        batch, heads, Lv, num_kv_splits = 4, 4, 64, 8
        seq_lens = torch.randint(1, num_kv_splits, (batch,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device)

        o_raw, lse_raw = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_blk, lse_blk = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits, kernel_fn=_fwd_kernel_stage2_block_ptr)

        torch.testing.assert_close(o_raw, o_blk, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_raw, lse_blk, atol=1e-5, rtol=1e-5)
