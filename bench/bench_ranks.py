# SPDX-License-Identifier: Apache-2.0
"""Benchmark Ranks Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.vllm.ranks.block_ptr import _ranks_kernel_block_ptr
from kernels.vllm.ranks.wrapper import ranks


def bench_ranks():
    """Benchmark raw-pointer vs block-pointer Ranks kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    vocab_sizes = [32000]
    batch_size = 64

    print("Ranks Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for vocab_size in vocab_sizes:
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)

        ms_raw = stable_bench(lambda: ranks(logits, token_ids))
        ms_blk = stable_bench(lambda: ranks(logits, token_ids, kernel_fn=_ranks_kernel_block_ptr))

        print(format_result(vocab_size, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_ranks()
