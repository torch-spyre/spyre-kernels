# SPDX-License-Identifier: Apache-2.0
"""Benchmark Top-K Log-Softmax Triton kernels."""

import torch
import triton.testing

from kernels.log_softmax.wrapper import topk_log_softmax
from kernels.log_softmax.block_ptr import _topk_log_softmax_kernel_block_ptr


def bench_log_softmax():
    """Benchmark raw-pointer vs block-pointer Log-Softmax kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    vocab_sizes = [32000]
    batch_size = 32
    topk = 10

    print("Log-Softmax Benchmark")
    print("=" * 60)
    print(f"{'Vocab Size':<12} {'Raw (ms)':<12} {'BlockPtr (ms)':<14} {'Slowdown':<10}")
    print("-" * 60)

    for vocab_size in vocab_sizes:
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64)

        ms_raw = triton.testing.do_bench(lambda: topk_log_softmax(logits, topk_ids, topk))
        ms_blk = triton.testing.do_bench(lambda: topk_log_softmax(logits, topk_ids, topk, kernel_fn=_topk_log_softmax_kernel_block_ptr))

        slowdown = ms_blk / ms_raw
        print(f"{vocab_size:<12} {ms_raw:<12.3f} {ms_blk:<14.3f} {slowdown:<10.2f}x")


if __name__ == "__main__":
    bench_log_softmax()
