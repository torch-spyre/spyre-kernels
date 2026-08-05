# SPDX-License-Identifier: Apache-2.0
"""Benchmark Embedding Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.vllm.embedding.block_ptr import embedding_forward_kernel_block_ptr
from kernels.vllm.embedding.wrapper import embedding


def bench_embedding():
    """Benchmark raw-pointer vs block-pointer Embedding kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    vocab_size = 32000
    # (embedding_dim, n_tokens) — decode (single token) and prefill batches
    configs = [
        (4096, 1),
        (4096, 128),
        (4096, 512),
        (4096, 2048),
    ]

    print("Embedding Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for embedding_dim, n_tokens in configs:
        table = torch.randn(vocab_size, embedding_dim, device=device, dtype=torch.bfloat16)
        indices = torch.randint(0, vocab_size, (n_tokens,), device=device, dtype=torch.int64)

        ms_raw = stable_bench(lambda: embedding(table, indices))
        ms_blk = stable_bench(
            lambda: embedding(table, indices, kernel_fn=embedding_forward_kernel_block_ptr)
        )

        label = f"D={embedding_dim},N={n_tokens}"
        print(format_result(label, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_embedding()
