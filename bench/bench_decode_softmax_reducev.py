# SPDX-License-Identifier: Apache-2.0
"""Benchmark Decode Softmax+ReduceV Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.decode_softmax_reducev.block_ptr import _fwd_kernel_stage2_block_ptr
from kernels.decode_softmax_reducev.wrapper import decode_softmax_reducev


def _make_mid_o(batch, heads, num_kv_splits, Lv, seq_lens, device):
    """Create a realistic Mid_O tensor with partial V outputs and LSE values."""
    mid_o = torch.randn(batch, heads, num_kv_splits, Lv + 1, device=device, dtype=torch.float32)
    mid_o[:, :, :, Lv] = torch.randn(batch, heads, num_kv_splits, device=device) * 2.0
    return mid_o


def bench_decode_softmax_reducev():
    """Benchmark raw-pointer vs block-pointer Decode Softmax+ReduceV kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    Lv_values = [64, 128]
    batch_size = 4
    heads = 4
    num_kv_splits = 4

    print("Decode Softmax+ReduceV Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for Lv in Lv_values:
        seq_lens = torch.randint(num_kv_splits, 256, (batch_size,), device=device, dtype=torch.int32)
        mid_o = _make_mid_o(batch_size, heads, num_kv_splits, Lv, seq_lens, device)

        ms_raw = stable_bench(lambda: decode_softmax_reducev(mid_o.clone(), seq_lens, num_kv_splits))
        ms_blk = stable_bench(lambda: decode_softmax_reducev(mid_o.clone(), seq_lens, num_kv_splits, kernel_fn=_fwd_kernel_stage2_block_ptr))

        print(format_result(Lv, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_decode_softmax_reducev()
