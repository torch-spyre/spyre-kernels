# SPDX-License-Identifier: Apache-2.0
"""Benchmark Top-K Log-Softmax Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.log_softmax.block_ptr import _topk_log_softmax_kernel_block_ptr
from kernels.log_softmax.wrapper import topk_log_softmax


def bench_log_softmax():
    """Benchmark raw-pointer vs block-pointer Log-Softmax kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    vocab_sizes = [32000]
    batch_size = 32
    topk = 10

    print("Log-Softmax Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for vocab_size in vocab_sizes:
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, vocab_size, (batch_size, topk), device=device, dtype=torch.int64)

        ms_raw = stable_bench(lambda: topk_log_softmax(logits, topk_ids, topk))
        ms_blk = stable_bench(lambda: topk_log_softmax(logits, topk_ids, topk, kernel_fn=_topk_log_softmax_kernel_block_ptr))

        print(format_result(vocab_size, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_log_softmax()
