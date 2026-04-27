# Tritokti

Triton-to-KTIR kernel porting for vLLM on IBM Spyre/AIU accelerators.

## Overview

This project ports Triton kernels from vLLM's dense transformer inference path to work on IBM Spyre/AIU hardware. The conversion follows a three-phase pipeline:

```
Raw-pointer Triton → Block-pointer Triton → KTIR (MLIR dialect for Spyre)
```

**Target models:** Granite4 8B, Mistral3/Ministral 8B (dense decoder-only transformers with GQA attention, RMSNorm, SwiGLU MLP, RoPE)

## Kernel Inventory

| # | Kernel | Component | Status |
|---|--------|-----------|--------|
| 1 | `rms_norm` | RMSNorm normalization | Converted |
| 2 | `silu_and_mul` | SwiGLU activation | Converted |
| 3 | `ranks` | Logprob ranks | Converted |
| 4 | `log_softmax` | Top-K log-softmax | Converted |
| 5 | `decode_softmax_reducev` | Decode attention merge | Converted |
| 6 | `merge_attn_states` | Attention state merge | Converted |
| 7 | `mrope` | Multi-RoPE embeddings | Converted |
| 8 | `reshape_and_cache` | KV cache reshape | Converted |
| 9 | `prefill_attention` | Prefill SDPA | Converted |

All kernels are extracted verbatim from vLLM commit [`cde8d2471026`](https://github.com/vllm-project/vllm/commit/cde8d2471026).

## Quick Start

### Prerequisites

```bash
# Activate the virtual environment
source .venv-office/bin/activate

# Or create a new environment with uv
uv venv
source .venv/bin/activate
uv pip install -e .[test]
```

### Running Tests

**GPU tests (Triton kernel equivalence):**
```bash
# All Triton tests (requires GPU)
pytest tests/triton/ -v

# Single kernel test
pytest tests/triton/test_rms_norm.py -v
```

**CPU tests (KTIR validation):**
```bash
# All KTIR tests (CPU only, requires ktir_cpu interpreter)
cd external/ktir_cpu
uv run python ../tests/ktir/test_rms_norm.py
uv run python ../tests/ktir/test_silu_and_mul.py
# ... etc
```

### Running Benchmarks

```bash
# All benchmarks
python bench/run_all.py

# Single benchmark
python bench/bench_rms_norm.py
python bench/bench_silu_and_mul.py
```

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
├── tests/
│   ├── triton/                # GPU equivalence tests
│   │   └── test_<name>.py
│   └── ktir/                  # CPU validation tests
│       └── test_<name>.py
│
├── bench/
│   ├── bench_<name>.py        # Performance benchmarks
│   └── run_all.py             # Run all benchmarks
│
├── scripts/
│   └── fetch_originals.py     # Extract kernels from vLLM
│
└── kernels.json               # Kernel registry (vLLM source mapping)
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
import triton.testing
from kernels.your_kernel.wrapper import your_kernel

def bench_your_kernel():
    x = torch.randn(...)
    ms_raw = triton.testing.do_bench(lambda: your_kernel(x))
    ms_blk = triton.testing.do_bench(lambda: your_kernel(x, kernel_fn=...))
    print(f"Slowdown: {ms_blk/ms_raw:.2f}x")
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

## Verification

Check kernel sync with vLLM:

```bash
python scripts/fetch_originals.py --diff
```

## Resources

- [Experiment Design](status.md) - Detailed conversion status and methodology
- [Status Report](status.md) - Per-kernel conversion details
- [Project Plan](plan.md) - Roadmap and milestones
