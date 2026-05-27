# Spyre Kernel Test Skill

Write numerical equivalence tests for a Spyre-aware Triton kernel by comparing its output against the original kernel using random inputs.

## Trigger

Use when the user asks to write tests, add tests, or create a test file for a Spyre kernel (`spyre.py`).

## Inputs

- **kernel_name**: Name of the kernel directory under `kernels/` (e.g., `matmul`, `log_softmax`)
- The Spyre kernel is at `kernels/<name>/spyre.py`
- The original kernel is at `kernels/<name>/original.py`
- The existing wrapper is at `kernels/<name>/wrapper.py`
- The output test file goes at `tests/triton/test_<name>_spyre.py`

## Pre-flight: Consult the Spyre Knowledge Base

Before writing tests, query the `spyre-kb` MCP server for relevant context:

1. **Search for kernel-specific constraints** — call `mcp__spyre-kb__search(query="<kernel_type>")` (e.g., `"softmax"`, `"matmul"`, `"attention"`) to check for known precision issues, hardware-specific edge cases, or documented test patterns.
2. **Search for precision/dtype guidance** — call `mcp__spyre-kb__search(query="precision DL16 BF16")` to understand dtype behavior on Spyre hardware. This informs tolerance choices — DL16 (AIU 1.0) has different mantissa/exponent than IEEE fp16.
3. **Read relevant pages** — if search surfaces pages about tolerance, numerical stability, or test patterns for this kernel type, call `mcp__spyre-kb__read(path="<path>")` for full details.

Use knowledge base results to:
- Adjust tolerances if the KB documents known precision differences for this kernel type
- Add extra edge cases if the KB documents known failure modes
- Validate that test shapes cover hardware-relevant boundaries (e.g., stick-aligned sizes at 64 elements for 16-bit)

## Pre-flight: Read Source Files

1. Read `kernels/<name>/spyre.py` to understand the kernel signature (pointers, runtime args, constexprs)
2. Read `kernels/<name>/wrapper.py` to understand how the original kernel is launched (shapes, dtypes, output allocation)
3. Read `tests/triton/test_<name>.py` (if it exists) to match the existing test style and parameter choices

## Test File Structure

```python
# SPDX-License-Identifier: Apache-2.0
"""
Numerical tests for the Spyre-aware <name> kernel.

Compares <kernel_function>_spyre output against the original kernel
across various shapes and core counts.

Run: pytest tests/triton/test_<name>_spyre.py -v
Requires: GPU with triton support (tensor descriptor support)
"""

import pytest
import torch
import triton

from kernels.<name>.spyre import <kernel_function>_spyre
from kernels.<name>.wrapper import <wrapper_function> as <wrapper_function>_original


# ─── Helpers ──────────────────────────────────────────────────────

def <wrapper_function>_spyre(
    <args matching original wrapper>,
    num_cores: int = 32,
    <BLOCK constexprs with defaults>,
) -> torch.Tensor:
    """Launch the Spyre kernel with a fixed grid."""
    ...
    grid = (num_cores,)
    <kernel_function>_spyre[grid](...)
    return output


# ─── Test Parameters ───────────────────────────────────────────────

<SHAPES, SIZES, etc. — include divisible and non-divisible values>

CORE_COUNTS = [1, 4, 16, 32]


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Tests ─────────────────────────────────────────────────────────

class Test<Name>SpyreCorrectness:
    """Spyre kernel must match original kernel."""
    ...

class Test<Name>SpyreDistribution:
    """Verify correctness across different core counts."""
    ...

class Test<Name>SpyreEdgeCases:
    """Edge cases for the Spyre kernel."""
    ...
```

## Test Categories

### 1. Correctness (compare against original)

- Parametrize over representative shapes from the existing test file
- Use `torch.manual_seed(42)` for reproducibility
- Generate random inputs with appropriate dtypes
- Call both the original wrapper and the Spyre wrapper
- Compare with `torch.testing.assert_close(out_spyre, out_original, atol=..., rtol=...)`

**Tolerance guidelines:**
- f32 output from f32 inputs: `atol=1e-5, rtol=1e-5`
- f32 output from f16/bf16 inputs: `atol=1e-5, rtol=1e-5` (reductions accumulate in f32)
- f16 output from f16 inputs: `atol=1e-2, rtol=0`

### 2. Distribution invariance (varying core counts)

- Run the Spyre kernel with `num_cores` in `[1, 4, 16, 32]`
- Compare each result against the original kernel
- This verifies the distribution loop works for any partitioning

Also test:
- More cores than work items (some cores idle)
- Single core (sequential execution)

### 3. Edge cases

Always include:
- **Non-divisible shapes**: dimensions not multiples of BLOCK sizes
- **Minimum size**: inputs smaller than one tile
- **Asymmetric shapes**: one dimension much larger than another
- **Kernel-specific edges**: e.g., topk=1, single batch, vocab < BLOCK_SIZE

## Spyre Wrapper Conventions

The Spyre wrapper differs from the original wrapper:
- **Grid**: fixed `(num_cores,)` instead of derived from problem size
- **No stride args**: Spyre kernels assume contiguous layout (assert it)
- **Explicit constexprs**: pass BLOCK sizes directly (no autotune)
- **Extra runtime args**: `num_requests`, `num_elements`, etc. that replace grid-size-equals-problem-size patterns

## Checklist

1. [ ] Test file at `tests/triton/test_<name>_spyre.py`
2. [ ] Helper wrapper launches Spyre kernel with configurable `num_cores`
3. [ ] Correctness tests parametrized over shapes matching existing tests
4. [ ] Distribution tests with `CORE_COUNTS = [1, 4, 16, 32]`
5. [ ] Edge cases: non-divisible, minimum size, asymmetric
6. [ ] All comparisons use `torch.testing.assert_close` against original kernel
7. [ ] `torch.manual_seed(42)` in every test for reproducibility
8. [ ] Random inputs (not handcrafted) — catches more bugs
9. [ ] No torch.matmul or PyTorch reference — compare only against original kernel
