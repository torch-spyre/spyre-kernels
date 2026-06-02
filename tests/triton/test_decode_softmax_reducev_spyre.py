# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the Spyre-aware decode_softmax_reducev kernel.

Compares _fwd_kernel_stage2_spyre output against the original kernel
across various shapes and core counts.

Run: pytest tests/triton/test_decode_softmax_reducev_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.decode_softmax_reducev.spyre import _fwd_kernel_stage2_spyre
from kernels.decode_softmax_reducev.wrapper import decode_softmax_reducev


# ─── Helpers ──────────────────────────────────────────────────────

def decode_softmax_reducev_spyre(
    mid_o: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
    num_cores: int = 32,
    tile_size_dv: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Spyre decode_softmax_reducev kernel with a fixed grid."""
    batch, heads, _splits, lv_plus_1 = mid_o.shape
    Lv = lv_plus_1 - 1

    o = torch.empty(batch, heads, Lv, device=mid_o.device, dtype=mid_o.dtype)
    lse = torch.empty(batch, heads, device=mid_o.device, dtype=torch.float32)

    grid = (num_cores,)
    _fwd_kernel_stage2_spyre[grid](
        mid_o,
        o,
        lse,
        b_seq_len,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        batch,
        heads,
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_SIZE=tile_size_dv,
        Lv=Lv,
    )
    return o, lse


# ─── Test Parameters ───────────────────────────────────────────────

LV_VALUES = [32, 64, 128]
BATCH_SIZES = [1, 4, 16]
HEAD_COUNTS = [1, 4, 8]
NUM_KV_SPLITS_VALUES = [2, 4, 8]
CORE_COUNTS = [1, 4, 16, 32]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device):
    """Create a realistic Mid_O tensor with partial V outputs and LSE values."""
    mid_o = torch.randn(batch, heads, num_kv_splits, Lv + 1, device=device, dtype=torch.float32)
    mid_o[:, :, :, Lv] = torch.randn(batch, heads, num_kv_splits, device=device) * 2.0
    return mid_o


# ─── Tests ─────────────────────────────────────────────────────────

class TestDecodeSoftmaxReducevSpyreCorrectness:
    """Spyre kernel must match original kernel."""

    @pytest.mark.parametrize("Lv", LV_VALUES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    @pytest.mark.parametrize("num_kv_splits", NUM_KV_SPLITS_VALUES)
    def test_numerical_equivalence(self, device, batch_size, heads, Lv, num_kv_splits):
        torch.manual_seed(42)
        seq_lens = torch.randint(num_kv_splits, 512, (batch_size,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch_size, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("Lv", [57, 100, 121])
    def test_non_power_of_two_lv(self, device, Lv):
        """Lv not a power of 2 — BLOCK_DV > Lv, tests OOB handling."""
        torch.manual_seed(42)
        batch_size, heads, num_kv_splits = 4, 4, 4
        seq_lens = torch.randint(num_kv_splits, 256, (batch_size,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch_size, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)


class TestDecodeSoftmaxReducevSpyreDistribution:
    """Verify correctness across different core counts."""

    @pytest.mark.parametrize("num_cores", CORE_COUNTS)
    def test_core_count_invariance(self, device, num_cores):
        """Result must be identical regardless of work distribution."""
        torch.manual_seed(42)
        batch_size, heads, Lv, num_kv_splits = 8, 8, 128, 4
        seq_lens = torch.randint(num_kv_splits, 256, (batch_size,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch_size, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits, num_cores=num_cores)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)

    def test_more_cores_than_work(self, device):
        """When num_cores > batch*heads, some cores have no work."""
        torch.manual_seed(42)
        batch, heads, Lv, num_kv_splits = 1, 2, 64, 4
        seq_lens = torch.full((batch,), 128, device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits, num_cores=32)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)

    def test_single_core(self, device):
        """Single core processes all (batch, head) pairs sequentially."""
        torch.manual_seed(42)
        batch, heads, Lv, num_kv_splits = 4, 8, 128, 4
        seq_lens = torch.randint(num_kv_splits, 256, (batch,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits, num_cores=1)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)


class TestDecodeSoftmaxReducevSpyreEdgeCases:
    """Edge cases for the Spyre kernel."""

    def test_single_split(self, device):
        """num_kv_splits=1 — just pass through."""
        torch.manual_seed(42)
        batch, heads, Lv = 4, 4, 64
        seq_lens = torch.full((batch,), 128, device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, 1, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, 1)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, 1)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)

    def test_short_sequences(self, device):
        """Some splits may be empty when seq_len < num_kv_splits."""
        torch.manual_seed(42)
        batch, heads, Lv, num_kv_splits = 4, 4, 64, 8
        seq_lens = torch.randint(1, num_kv_splits, (batch,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)

    def test_large_batch_heads(self, device):
        """Large total work count — verifies distribution with many items."""
        torch.manual_seed(42)
        batch, heads, Lv, num_kv_splits = 32, 32, 128, 4
        seq_lens = torch.randint(num_kv_splits, 512, (batch,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)

    def test_small_lv(self, device):
        """Lv=4 — minimum head dimension (descriptor requires >=16 bytes in last dim)."""
        torch.manual_seed(42)
        batch, heads, Lv, num_kv_splits = 4, 4, 4, 4
        seq_lens = torch.randint(num_kv_splits, 128, (batch,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device)

        o_orig, lse_orig = decode_softmax_reducev(mid_o, seq_lens, num_kv_splits)
        o_spyre, lse_spyre = decode_softmax_reducev_spyre(mid_o, seq_lens, num_kv_splits)

        torch.testing.assert_close(o_spyre, o_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_spyre, lse_orig, atol=1e-5, rtol=1e-5)
