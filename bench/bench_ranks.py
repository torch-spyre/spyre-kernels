# SPDX-License-Identifier: Apache-2.0
"""Benchmark Ranks Triton kernels."""

import torch
import triton.testing

from kernels.ranks.wrapper import ranks
from kernels.ranks.block_ptr import _ranks_kernel_block_ptr


def bench_ranks():
    """Benchmark raw-pointer vs block-pointer Ranks kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    vocab_sizes = [32000]
    batch_size = 64

    print("Ranks Benchmark")
    print("=" * 60)
    print(f"{'Vocab Size':<12} {'Raw (ms)':<12} {'BlockPtr (ms)':<14} {'Slowdown':<10}")
    print("-" * 60)

    for vocab_size in vocab_sizes:
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)

        ms_raw = triton.testing.do_bench(lambda: ranks(logits, token_ids))
        ms_blk = triton.testing.do_bench(lambda: ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"{vocab_size:<12} {ms_raw:<12.3f} {ms_blk:<14.3f} {slowdown:<10.2f}x")


if __name__ == "__main__":
    bench_ranks()
