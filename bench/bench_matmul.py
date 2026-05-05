# SPDX-License-Identifier: MIT
"""Benchmark matmul Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.matmul.block_ptr import matmul_kernel_block_ptr
from kernels.matmul.wrapper import matmul


def bench_matmul():
    """Benchmark raw-pointer vs block-pointer matmul kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1, 4096, 4096),       # single-token decode projection
        (128, 4096, 4096),     # prefill projection
    ]

    print("Matmul Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for M, N, K in sizes:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        ms_raw = stable_bench(lambda: matmul(a, b))
        ms_blk = stable_bench(lambda: matmul(a, b, kernel_fn=matmul_kernel_block_ptr))

        label = f"{M}x{N}x{K}"
        print(format_result(label, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_matmul()
