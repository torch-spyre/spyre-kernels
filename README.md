# Tritokti

Triton-to-KTIR kernel porting for vLLM on IBM Spyre/AIU accelerators.

## Overview

This project ports Triton kernels from vLLM's dense transformer inference path to work on IBM Spyre/AIU hardware. Spyre uses a tiled "stick" memory layout (128-byte aligned), so raw pointer arithmetic from GPU Triton kernels doesn't map directly. Block pointers abstract memory access into structured operations that can be lowered to Spyre's model. The conversion follows a three-phase pipeline:

```
Raw-pointer Triton → Block-pointer Triton → KTIR (MLIR dialect for Spyre)
```

**Target models:** Granite4 8B, Mistral3/Ministral 8B (dense decoder-only transformers with GQA attention, RMSNorm, SwiGLU MLP, RoPE)

## Kernel Inventory

| # | Kernel | Source | Component |
|---|--------|--------|-----------|
| 1 | `rms_norm` | vLLM | RMSNorm normalization |
| 2 | `silu_and_mul` | vLLM | SwiGLU activation |
| 3 | `ranks` | vLLM | Logprob ranks |
| 4 | `log_softmax` | vLLM | Top-K log-softmax |
| 5 | `decode_softmax_reducev` | vLLM | Decode attention merge |
| 6 | `merge_attn_states` | vLLM | Attention state merge |
| 7 | `mrope` | vLLM | Multi-RoPE embeddings |
| 8 | `reshape_and_cache` | vLLM | KV cache reshape |
| 9 | `prefill_attention` | vLLM | Prefill SDPA |
| 10 | `matmul` | Triton | Dense matrix multiplication (GEMM) |
| 11 | `embedding` | Liger-Kernel | Embedding table lookup |

Kernels are extracted from vLLM commit [`cde8d2471026`](https://github.com/vllm-project/vllm/commit/cde8d2471026), Triton commit [`933cefce4`](https://github.com/triton-lang/triton/commit/933cefce4ecbb1600bac10e975d1e6fad166b587), and Liger-Kernel commit [`c4b16d43`](https://github.com/linkedin/Liger-Kernel/commit/c4b16d43f9d8f69068e6a15bd879dfc6a63b2449).

## Project Structure

```
tritokti/
├── kernels/                   # Kernel implementations
│   ├── <name>/
│   │   ├── original.py        # Raw-pointer Triton (from vLLM)
│   │   ├── block_ptr.py       # Block-pointer Triton
│   │   ├── wrapper.py         # Python launcher
│   │   └── kernel.ktir        # KTIR output
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
# Install with test dependencies (includes ktir-cpu interpreter)
uv sync --extra test
```

### Running Tests

**GPU tests** verify that block-pointer kernels produce numerically identical results to the original raw-pointer vLLM kernels:
```bash
# All GPU equivalence tests (requires GPU)
pytest tests/triton/ -v

# Single kernel test
pytest tests/triton/test_rms_norm.py -v
```

**KTIR tests** validate KTIR output against the original vLLM kernels using the `ktir_cpu` interpreter:
```bash
# All KTIR tests (requires GPU + ktir-cpu, installed via uv sync --extra test)
uv run pytest tests/ktir/ -v

# Single kernel test
uv run pytest tests/ktir/test_rms_norm.py -v
```

**All tests** at once:
```bash
pytest tests/ -v
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

Edit `kernels.json`. The registry supports multiple sources (vLLM, Triton, etc.):

```json
{
  "sources": {
    "vllm": {
      "repo": "vllm-project/vllm",
      "commit": "cde8d2471026",
      "license_header": "# SPDX-License-Identifier: Apache-2.0\n# ..."
    },
    "triton": {
      "repo": "triton-lang/triton",
      "commit": "933cefce4...",
      "license_header": "# SPDX-License-Identifier: MIT\n# ..."
    }
  },
  "kernels": {
    "your_kernel": {
      "source": "vllm",
      "file": "path/to/kernel.py",
      "kernel_function": "_your_kernel_fn",
      "helpers": ["optional_helper_fn"]
    }
  }
}
```

Each kernel specifies:
- `source`: which source repo to fetch from
- `file`: path within that repo
- `kernel_function`: the `@triton.jit` function to extract
- `helpers` (optional): additional functions to extract (placed before the kernel)

### 2. Extract from Source

```bash
python scripts/fetch_originals.py your_kernel  # fetch single kernel
python scripts/fetch_originals.py              # fetch all
```

This creates `kernels/your_kernel/original.py` with the verbatim kernel and attribution header.

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
import torch
from ktir_cpu import KTIRInterpreter
from kernels.your_kernel.wrapper import your_kernel

def vllm_reference(x_np):
    """Run vLLM kernel on GPU and return result as numpy."""
    x = torch.from_numpy(x_np).cuda()
    out = your_kernel(x)
    return out.cpu().numpy()

def test_your_kernel_ktir():
    interp = KTIRInterpreter()
    interp.load("kernels/your_kernel/kernel.ktir")
    # ... execute and compare to vLLM reference ...
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
