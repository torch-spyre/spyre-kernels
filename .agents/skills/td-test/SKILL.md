---
name: td-test
description: "Write numerical equivalence tests for a tensor-descriptor kernel (tensor_descriptor.py), comparing it against the original kernel. Produces tests/triton/test_<name>_td.py. Use when asked to test a _td kernel."
---

# Tensor-Descriptor Test Skill

Write numerical-equivalence tests for a `_td` kernel against the original, using
random inputs. Output goes to `tests/triton/test_<name>_td.py`.

This is for the **descriptor-API** form.

## Trigger

Use when asked to write/add tests for a `tensor_descriptor.py` (`_td`) kernel.

## Pre-flight

1. Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md)
   (precision notes can shift tolerances).
2. Read `kernels/<name>/tensor_descriptor.py` for the signature.
3. Read `kernels/<name>/wrapper.py` for launch shapes/dtypes.
4. Read `tests/triton/test_<name>.py` if present, to match style.

## Launching the kernel — reuse `wrapper.py`

**Use the existing `kernels/<name>/wrapper.py` to launch both kernels.** It
already dispatches to the `_td` kernel (typically via a `kernel_fn=` argument)
and owns the grid, allocator, and signature — call it for the `_td` form and the
original form alike. Do **not** copy its launch logic into the test as a private
helper: a forked launch path drifts from production and has to be updated twice
(exactly the duplication that bites when the kernel signature changes).

If the wrapper does not expose a parameter a test needs (e.g. a tile-batching
constexpr, or a stride you want to vary), **extend `wrapper.py` to accept it**
so the test and the production path stay on the same code — only fall back to a
direct in-test launch when changing the wrapper is genuinely not appropriate.

```python
# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the tensor-descriptor <name> kernel."""

import pytest
import torch

from kernels.<name>.tensor_descriptor import _<name>_kernel_td
from kernels.<name>.wrapper import <wrapper_fn>


# Reference vs _td both go through the wrapper:
#   out_ref = <wrapper_fn>(x, ...)                              # original kernel
#   out_td  = <wrapper_fn>(x, ..., kernel_fn=_<name>_kernel_td) # _td kernel


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")
```

## Test categories

### 1. Correctness vs original

Parametrize over representative shapes (include non-divisible), seed with
`torch.manual_seed(42)`, random inputs, compare with
`torch.testing.assert_close`.

**Tolerances — size to the output dtype's ULP, not a round number.** When the
`_td` kernel and the original differ *only* in float op order (a reduction
accumulated across tiles vs reduced per-tile), output differs by ≤ ~1 ULP of the
output dtype. Define a per-dtype `TOL` dict once:

```python
# Kernels differ only in f32 reduction order → per-element error <= 1 ULP.
TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),   # f32 reductions stay near-exact
    torch.float16: dict(atol=1e-3, rtol=1e-3),   # 10-bit mantissa
    torch.bfloat16: dict(atol=8e-3, rtol=8e-3),  # 7-bit mantissa
}
...
torch.testing.assert_close(out_td, out_original, **TOL[dtype])
```

Loosen *up* only with a stated reason in a comment — e.g. the divergence is not
pure reordering (a precomputed `inv_rms` that multiplies, a `tl.dot` whose K
accumulates in a different order than the reference), or the KB documents a
precision difference. **f32 outputs should be near-exact** (reductions
accumulate in f32) — keep `1e-5`.

**`atol=0` is for bytes the kernel *moved*, `TOL[dtype]` is for numbers it
*computed*.** Reduction order is not the only source of divergence: the original
and the `_td` kernel lower their loads differently (raw pointer + mask vs
descriptor + `where`), so the compiler may contract a multiply-add into an FMA
on one path and not the other — ~1 ULP, elementwise, no reduction involved. So a
single-tile / no-reduction case is *not* automatically bitwise-identical.

- Output the kernel computes arithmetically (any multiply-then-add/sub) → FMA
  hazard → use `TOL[dtype]`, even single-tile.
- Output the kernel only copies/passes through untouched → genuinely
  bitwise-identical → assert `atol=0, rtol=0`. A loose tolerance there hides
  regressions.

### 2. Edge cases

- **Non-divisible shapes** (dimensions not a multiple of BLOCK).
- **Minimum size** (smaller than one tile).
- **Asymmetric shapes**, **zero input** (eps handling), **large values**.
- Any kernel-specific tile-batching parameter (one controlling how many work
  items a program processes per tile) — parametrize over its valid values,
  including ones that do not divide the work-item count, to exercise OOB
  padding. Respect any power-of-2 constraint the descriptor imposes on a block
  dimension.

## Checklist

1. [ ] Test at `tests/triton/test_<name>_td.py`
2. [ ] Both kernels launched via `wrapper.py` (no forked launch path; extend
       the wrapper if a needed param is missing); correctness parametrized over shapes
3. [ ] Edge cases: non-divisible, minimum size, asymmetric, zero, large
4. [ ] Compare with `torch.testing.assert_close` against the original only
5. [ ] Per-dtype `TOL` dict; loosening has a stated reason
6. [ ] Single-tile cases assert `atol=0, rtol=0`
7. [ ] `torch.manual_seed(42)` everywhere; random (not handcrafted) inputs
