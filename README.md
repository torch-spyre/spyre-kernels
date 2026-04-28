# Tritokti

Triton-to-KTIR kernel porting for vLLM on IBM Spyre/AIU accelerators.

## Overview

This project ports Triton kernels from vLLM's dense transformer inference path to work on IBM Spyre/AIU hardware. Spyre uses a tiled "stick" memory layout (128-byte aligned), so raw pointer arithmetic from GPU Triton kernels doesn't map directly. Block pointers abstract memory access into structured operations that can be lowered to Spyre's model. The conversion follows a three-phase pipeline:

```
Raw-pointer Triton → Block-pointer Triton → KTIR (MLIR dialect for Spyre)
```

**Target models:** Granite4 8B, Mistral3/Ministral 8B (dense decoder-only transformers with GQA attention, RMSNorm, SwiGLU MLP, RoPE)

## Kernel Inventory

| # | Kernel | Component |
|---|--------|-----------|
| 1 | `rms_norm` | RMSNorm normalization |
| 2 | `silu_and_mul` | SwiGLU activation |
| 3 | `ranks` | Logprob ranks |
| 4 | `log_softmax` | Top-K log-softmax |
| 5 | `decode_softmax_reducev` | Decode attention merge |
| 6 | `merge_attn_states` | Attention state merge |
| 7 | `mrope` | Multi-RoPE embeddings |
| 8 | `reshape_and_cache` | KV cache reshape |
| 9 | `prefill_attention` | Prefill SDPA |

All kernels are extracted verbatim from vLLM commit [`cde8d2471026`](https://github.com/vllm-project/vllm/commit/cde8d2471026).

## Project Structure

```
tritokti/
├── kernels/                    # Kernel implementations
│   ├── <name>/
│   │   ├── original.py        # Raw-pointer Triton (from vLLM)
│   │   ├── block_ptr.py       # Block-pointer Triton
│   │   ├── wrapper.py         # Python launcher
│   │   └── kernel.ktir.mlir   # KTIR output
│
├── external/
│   └── ktir_cpu/              # KTIR CPU interpreter (cloned separately)
│
├── tests/
│   ├── triton/                # GPU equivalence tests
│   │   └── test_<name>.py
│   └── ktir/                  # CPU validation tests
│       └── test_<name>.py
│
├── bench/
│   ├── utils.py               # Shared benchmarking utilities
│   ├── bench_<name>.py        # Performance benchmarks
│   └── run_all.py             # Run all benchmarks
│
├── scripts/
│   └── fetch_originals.py     # Extract kernels from vLLM
│
├── ccc_results.md             # A100/H100 benchmark results
└── kernels.json               # Kernel registry (vLLM source mapping)
```

## Quick Start

### Prerequisites

```bash
# Activate the virtual environment
source .venv/bin/activate

# Or create a new environment with uv
uv venv
source .venv/bin/activate
uv pip install -e .[test]
```

### Running Tests

**GPU tests** verify that block-pointer kernels produce numerically identical results to the original raw-pointer vLLM kernels:
```bash
# All Triton tests (requires GPU)
pytest tests/triton/ -v

# Single kernel test
pytest tests/triton/test_rms_norm.py -v
```

**CPU tests** validate KTIR output against NumPy reference implementations using the `ktir_cpu` interpreter:
```bash
# Clone the ktir_cpu interpreter into external/ (one-time setup)
git clone https://github.com/torch-spyre/ktir-cpu external/ktir_cpu

# Run from the ktir_cpu directory (its pyproject.toml provides the interpreter dependency)
cd external/ktir_cpu
uv run python ../../tests/ktir/test_rms_norm.py
uv run python ../../tests/ktir/test_silu_and_mul.py
# ... etc
```

### Running Benchmarks

Benchmarks measure the GPU latency overhead of block-pointer conversion by comparing original vs. block-pointer kernels side-by-side:

```bash
# All benchmarks (uses stable_bench: 3-trial trimmed mean, median + [p20-p80] bands)
python bench/run_all.py

# With repeated trials for cross-run consistency checks
python bench/run_all.py --trials 3

# Lock GPU clocks for maximum stability (requires sudo)
python bench/run_all.py --lock-clocks

# Single benchmark
python bench/bench_rms_norm.py

# Control trial count via environment variable (default: 5)
BENCH_TRIALS=3 python bench/bench_mrope.py
```

Benchmarks use `bench/utils.py` for consistent methodology:
- GPU warmup before measurements
- `triton.testing.do_bench` with `warmup=100ms`, `rep=500ms`, `quantiles=[0.5, 0.2, 0.8]`
- Multiple independent trials with trimmed mean (drop min/max)
- Reports median + [p20-p80] percentile bands to show measurement noise

See [ccc_results.md](ccc_results.md) for A100/H100 benchmark results.

## Verification

Check kernel sync with vLLM:

```bash
python scripts/fetch_originals.py --diff
```

## Adding a New Kernel

### 1. Add to Registry

Edit `kernels.json`:

```json
{
  "repo": "vllm-project/vllm",
  "commit": "cde8d2471026",
  "kernels": {
    "your_kernel": {
      "vllm_file": "path/to/kernel.py",
      "kernel_function": "_your_kernel_fn"
    }
  }
}
```

### 2. Extract from vLLM

```bash
python scripts/fetch_originals.py
```

This creates `kernels/your_kernel/original.py` with the verbatim vLLM kernel.

### 3. Create Block-Pointer Version

Create `kernels/your_kernel/block_ptr.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _your_kernel_block_ptr(
    # ... arguments ...
):
    # Convert raw-pointer loads/stores to block pointers:
    # Before: x = tl.load(ptr + offset, mask=mask)
    # After:
    x_block = tl.make_block_ptr(
        base=ptr, shape=(N,), strides=(1,),
        offsets=(pid * BLOCK,), block_shape=(BLOCK,), order=(0,)
    )
    x = tl.load(x_block, boundary_check=(0,))
```

### 4. Add Wrapper

Create `kernels/your_kernel/wrapper.py`:

```python
import torch
from .original import _your_kernel
from .block_ptr import _your_kernel_block_ptr

def your_kernel(x, kernel_fn=None):
    if kernel_fn is None:
        kernel_fn = _your_kernel
    
    # Launch kernel
    grid = (...)
    kernel_fn[grid](x, ...)
    return x
```

### 5. Add Tests

**GPU test** (`tests/triton/test_your_kernel.py`):
```python
import pytest
import torch
from kernels.your_kernel.wrapper import your_kernel
from kernels.your_kernel.block_ptr import _your_kernel_block_ptr

def test_numerical_equivalence(device):
    x = torch.randn(...)
    out_raw = your_kernel(x)
    out_blk = your_kernel(x, kernel_fn=_your_kernel_block_ptr)
    torch.testing.assert_close(out_raw, out_blk)
```

**KTIR test** (`tests/ktir/test_your_kernel.py`):
```python
import numpy as np
from ktir_cpu import KTIRInterpreter

def test_your_kernel_ktir():
    interp = KTIRInterpreter()
    interp.load("kernels/your_kernel/kernel.ktir.mlir")
    # ... execute and compare to NumPy reference ...
```

### 6. Add Benchmark

Create `bench/bench_your_kernel.py`:

```python
import torch
from bench.utils import BENCH_HEADER, BENCH_SEP, format_result, gpu_warmup, stable_bench
from kernels.your_kernel.wrapper import your_kernel
from kernels.your_kernel.block_ptr import _your_kernel_block_ptr

def bench_your_kernel():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    gpu_warmup()

    print("Your Kernel Benchmark")
    print("=" * 80)
    print(BENCH_HEADER)
    print(BENCH_SEP)

    x = torch.randn(...)
    ms_raw = stable_bench(lambda: your_kernel(x))
    ms_blk = stable_bench(lambda: your_kernel(x, kernel_fn=_your_kernel_block_ptr))
    print(format_result("config", ms_raw, ms_blk))
```

### 7. Run Validation

```bash
# Verify equivalence
pytest tests/triton/test_your_kernel.py -v

# Run benchmark
python bench/bench_your_kernel.py

# Add to benchmark suite
# bench/run_all.py auto-discovers bench_*.py files
```

## Further Reading

See [plan.md](plan.md) for project roadmap and next steps.

See [status.md](status.md) for per-kernel conversion details and takeaways.
