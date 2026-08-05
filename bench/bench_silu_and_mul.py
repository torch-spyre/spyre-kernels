# SPDX-License-Identifier: Apache-2.0
"""Benchmark SwiGLU (silu_and_mul) Triton kernels."""

import torch

from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.vllm.silu_and_mul.block_ptr import _swiglustep_and_mul_kernel_block_ptr
from kernels.vllm.silu_and_mul.wrapper import silu_and_mul


def bench_silu_and_mul():
    """Benchmark raw-pointer vs block-pointer SwiGLU kernels."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    device = torch.device("cuda")
    half_hidden_sizes = [4096, 5120]
    batch_size = 128

    print("SwiGLU (silu_and_mul) Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    for d in half_hidden_sizes:
        x = torch.randn(batch_size, 2 * d, device=device, dtype=torch.bfloat16)

        ms_raw = stable_bench(lambda: silu_and_mul(x))
        ms_blk = stable_bench(lambda: silu_and_mul(x, kernel_fn=_swiglustep_and_mul_kernel_block_ptr))

        print(format_result(d, ms_raw, ms_blk))


if __name__ == "__main__":
    bench_silu_and_mul()
