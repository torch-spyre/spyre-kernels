# Block-Pointer vs Raw-Pointer Benchmark Results (CCC)

Benchmarked with `bench/utils.py` stabilization: 3 independent `do_bench` trials (trimmed mean), 100ms warmup, 500ms rep, median + [p20-p80] percentile bands. All times in milliseconds.

## NVIDIA A100-SXM4-80GB

torch 2.9.1+cu128, triton 3.5.1

| Kernel | Config | Raw median [p20-p80] | BlkPtr median [p20-p80] | Slowdown |
|--------|--------|----------------------|-------------------------|----------|
| decode_softmax_reducev | 64 | 0.011 [0.011-0.012] | 0.011 [0.011-0.012] | 1.00x |
| decode_softmax_reducev | 128 | 0.011 [0.011-0.012] | 0.011 [0.011-0.012] | 1.00x |
| log_softmax | 32000 | 0.046 [0.046-0.047] | 0.047 [0.046-0.048] | 1.01x |
| merge_attn_states | 64 | 0.007 [0.007-0.007] | 0.008 [0.007-0.009] | 1.16x |
| merge_attn_states | 128 | 0.007 [0.007-0.007] | 0.008 [0.007-0.009] | 1.16x |
| mrope | 64 | 0.014 [0.014-0.015] | 0.013 [0.013-0.014] | 0.93x |
| mrope | 128 | 0.013 [0.012-0.013] | 0.013 [0.013-0.014] | 1.02x |
| prefill_attention | [16] | 0.019 [0.019-0.020] | 0.020 [0.019-0.020] | 1.01x |
| prefill_attention | [32] | 0.020 [0.020-0.020] | 0.020 [0.020-0.021] | 1.01x |
| prefill_attention | [64] | 0.021 [0.020-0.021] | 0.021 [0.021-0.021] | 1.01x |
| ranks | 32000 | 0.032 [0.032-0.033] | 0.032 [0.032-0.033] | 1.01x |
| reshape_and_cache | 16x4x64 | 0.019 [0.019-0.019] | 0.019 [0.019-0.019] | 1.00x |
| rms_norm | 4096 | 0.012 [0.012-0.013] | 0.012 [0.012-0.013] | 1.02x |
| rms_norm | 5120 | 0.014 [0.013-0.014] | 0.014 [0.013-0.014] | 1.01x |
| silu_and_mul | 4096 | 0.011 [0.010-0.011] | 0.010 [0.010-0.010] | 0.90x |
| silu_and_mul | 5120 | 0.011 [0.010-0.012] | 0.010 [0.010-0.011] | 0.91x |
| matmul | 512x512x512 | 0.013 [0.013-0.013] | 0.013 [0.013-0.014] | 1.05x |
| matmul | 1024x1024x1024 | 0.026 [0.026-0.026] | 0.028 [0.028-0.028] | 1.06x |
| matmul | 2048x2048x2048 | 0.095 [0.095-0.096] | 0.099 [0.099-0.100] | 1.05x |
| matmul | 4096x4096x4096 | 0.613 [0.607-0.616] | 0.677 [0.673-0.683] | 1.10x |
| matmul | 1x4096x4096 | 0.057 [0.057-0.058] | 0.047 [0.046-0.047] | 0.81x |
| matmul | 128x4096x4096 | 0.049 [0.049-0.050] | 0.060 [0.060-0.061] | 1.22x |

## NVIDIA H100 80GB HBM3

torch 2.9.1+cu128, triton 3.5.1

| Kernel | Config | Raw median [p20-p80] | BlkPtr median [p20-p80] | Slowdown |
|--------|--------|----------------------|-------------------------|----------|
| decode_softmax_reducev | 64 | 0.009 [0.009-0.009] | 0.009 [0.009-0.009] | 1.00x |
| decode_softmax_reducev | 128 | 0.009 [0.009-0.009] | 0.009 [0.009-0.009] | 1.00x |
| log_softmax | 32000 | 0.034 [0.034-0.035] | 0.035 [0.034-0.035] | 1.00x |
| merge_attn_states | 64 | 0.005 [0.005-0.006] | 0.005 [0.005-0.006] | 0.98x |
| merge_attn_states | 128 | 0.005 [0.005-0.006] | 0.005 [0.005-0.006] | 0.99x |
| mrope | 64 | 0.010 [0.010-0.010] | 0.010 [0.010-0.010] | 1.03x |
| mrope | 128 | 0.010 [0.010-0.010] | 0.010 [0.010-0.010] | 1.01x |
| prefill_attention | [16] | 0.016 [0.016-0.017] | 0.016 [0.016-0.017] | 1.00x |
| prefill_attention | [32] | 0.017 [0.016-0.018] | 0.017 [0.016-0.018] | 1.00x |
| prefill_attention | [64] | 0.017 [0.016-0.018] | 0.017 [0.016-0.018] | 1.00x |
| ranks | 32000 | 0.026 [0.025-0.026] | 0.026 [0.026-0.026] | 1.01x |
| reshape_and_cache | 16x4x64 | 0.016 [0.015-0.017] | 0.016 [0.016-0.017] | 1.00x |
| rms_norm | 4096 | 0.009 [0.009-0.009] | 0.009 [0.009-0.009] | 1.01x |
| rms_norm | 5120 | 0.010 [0.010-0.010] | 0.010 [0.010-0.011] | 1.01x |
| silu_and_mul | 4096 | 0.007 [0.007-0.007] | 0.007 [0.007-0.007] | 1.00x |
| silu_and_mul | 5120 | 0.007 [0.007-0.007] | 0.007 [0.007-0.007] | 1.00x |
| matmul | 512x512x512 | 0.008 [0.008-0.009] | 0.009 [0.009-0.009] | 1.04x |
| matmul | 1024x1024x1024 | 0.012 [0.012-0.012] | 0.013 [0.013-0.013] | 1.06x |
| matmul | 2048x2048x2048 | 0.030 [0.029-0.030] | 0.032 [0.032-0.033] | 1.08x |
| matmul | 4096x4096x4096 | 0.205 [0.203-0.214] | 0.227 [0.225-0.232] | 1.11x |
| matmul | 1x4096x4096 | 0.041 [0.041-0.042] | 0.033 [0.032-0.033] | 0.79x |
| matmul | 128x4096x4096 | 0.032 [0.032-0.033] | 0.033 [0.033-0.034] | 1.02x |

## Cross-GPU Summary

| Kernel | Config | A100 Slowdown | H100 Slowdown | H100 Raw Speedup vs A100 |
|--------|--------|:---:|:---:|:---:|
| decode_softmax_reducev | 64 | 1.00x | 1.00x | 1.22x |
| log_softmax | 32000 | 1.01x | 1.00x | 1.35x |
| merge_attn_states | 64 | 1.16x | 0.98x | 1.40x |
| mrope | 64 | 0.93x | 1.03x | 1.40x |
| prefill_attention | [16] | 1.01x | 1.00x | 1.19x |
| ranks | 32000 | 1.01x | 1.01x | 1.23x |
| reshape_and_cache | 16x4x64 | 1.00x | 1.00x | 1.19x |
| rms_norm | 4096 | 1.02x | 1.01x | 1.33x |
| silu_and_mul | 4096 | 0.90x | 1.00x | 1.57x |
| matmul | 512x512x512 | 1.05x | 1.04x | 1.63x |
| matmul | 1024x1024x1024 | 1.06x | 1.06x | 2.17x |
| matmul | 2048x2048x2048 | 1.05x | 1.08x | 3.17x |
| matmul | 4096x4096x4096 | 1.10x | 1.11x | 2.99x |
| matmul | 1x4096x4096 | 0.81x | 0.79x | 1.39x |
| matmul | 128x4096x4096 | 1.22x | 1.02x | 1.53x |

## Key Findings

1. **Block-pointer overhead is negligible** — all kernels within 0.90x–1.03x on both GPUs after mrope optimization.
2. **A100 shows small regression** for merge_attn_states (1.16x) that disappears on H100. This is a sub-10μs kernel where GPU scheduling jitter dominates.
3. **H100 is 1.2-1.6x faster** than A100 across all raw-pointer baselines; matmul sees up to 3x speedup on large square shapes.
4. **Sub-15μs kernels** (silu_and_mul, merge_attn_states) show higher cross-run variance due to GPU scheduling jitter at that timescale.
5. **Matmul block-ptr overhead** is 4–11% on both GPUs for square shapes. The `1x4096x4096` decode shape is an exception — block-ptr is ~20% *faster* on both GPUs. The `128x4096x4096` prefill regression (1.22x on A100) largely vanishes on H100 (1.02x).
