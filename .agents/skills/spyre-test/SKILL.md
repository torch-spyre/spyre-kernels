---
name: spyre-test
description: "Write numerical equivalence tests for a Spyre-aware kernel (spyre.py) vs the original, including distribution invariance across core counts. Produces tests/triton/test_<name>_spyre.py. Use when asked to test a Spyre kernel. For plain _td kernels, use td-test."
---

# Spyre Kernel Test Skill

Write numerical-equivalence tests for a Spyre-aware kernel against the original,
including **distribution invariance** (same result for any core count). Output:
`tests/triton/test_<name>_spyre.py`. For a `_td` kernel (no distribution loop),
use `td-test`.

## Trigger

Use when asked to write/add tests for a `spyre.py` kernel.

## Pre-flight

1. Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md)
   (precision notes for Spyre DL16 can shift tolerances).
2. Read `kernels/<name>/spyre.py` (signature), `kernels/<name>/wrapper.py`
   (launch), and `tests/triton/test_<name>.py` if present (style).

## Execution path (intended vs interim)

The right Spyre verification path is **not** "GPU triton":

1. Lower Triton → TTIR → KTIR via the mlir-frontend (the Triton repo's spyre
   backend; `dump_round_trip.py` does this end-to-end).
2. Execute the KTIR on **ktir-cpu** (the `test_ktir_examples.py` flow).
3. Compare against a NumPy/PyTorch reference.

The fixture framework in
https://github.com/torch-spyre/triton/tree/main/third_party/spyre/test is the
canonical reference, and includes IR-level assertion helpers (`assert_present`,
`assert_result_type`, …) usable from the review side.

> **Interim:** until that path is wired into this repo's test harness, tests run
> the kernel on GPU/CUDA (the `@pytest.fixture device` + `torch.cuda` pattern
> below) and compare against the original kernel. Write tests against the
> original-kernel reference so they port cleanly to the ktir-cpu path later.

## Launching the kernel — reuse `wrapper.py`

**Launch both kernels through the existing `kernels/<name>/wrapper.py`** (which
dispatches to the Spyre kernel, typically via a `kernel_fn=` argument) — do not
copy its launch logic into the test. A forked launch path drifts from
production and must be updated twice when the signature changes.

Distribution-invariance tests need to vary `num_cores`, and the wrapper should
**accept `num_cores` as a parameter** so the test drives it through the same
launch path the production code uses. If the wrapper does not yet expose
`num_cores` (or another parameter a test needs), **extend `wrapper.py`** rather
than building a parallel launch helper in the test.

## Test categories

### 1. Correctness vs original

Parametrize over representative shapes (include non-divisible), seed with
`torch.manual_seed(42)`, random inputs, `torch.testing.assert_close`.

**Tolerances — size to the output dtype's ULP**, via a per-dtype `TOL` dict
(same rationale as `td-test`):

```python
TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=8e-3, rtol=8e-3),
}
```

Loosen *up* only with a stated reason. **Important — matmul / `tl.dot`
reductions:** `tl.dot` and a NumPy `@` reference accumulate the K dimension in
different orders, so a `1e-5` rtol is unrealistic — the project's
`matmul/default` fixture uses **`rtol≈1e-2`**. Align matmul-style tolerances with
that fixture, not the ULP table. **Single-tile cases** (`size <= BLOCK_SIZE`)
have no reduction-order divergence — assert `atol=0, rtol=0`.

### 2. Distribution invariance

Run the Spyre kernel with `num_cores` in `[1, 4, 16, 32]`; compare each against
the original. Also test: **more cores than work items** (idle cores) and
**single core** (sequential). This verifies the distribution loop for any
partitioning.

### 3. Edge cases

Non-divisible shapes; minimum size (< one tile); asymmetric shapes; zero input
(eps); large values; kernel-specific edges (topk=1, single batch, vocab <
BLOCK_SIZE).

## Spyre wrapper conventions

- **Grid:** fixed `(num_cores,)`, not derived from problem size.
- **No stride args** if the kernel assumes contiguous layout (assert it).
- **Explicit constexprs** (no autotune).
- **Extra runtime args** (`num_requests`, `num_elements`) that replace
  grid-size-equals-problem-size patterns.

## Checklist

1. [ ] Test at `tests/triton/test_<name>_spyre.py`
2. [ ] Both kernels launched via `wrapper.py` with configurable `num_cores`
       (extend the wrapper if it doesn't expose it — no forked launch path)
3. [ ] Correctness parametrized over shapes; distribution tests over `[1,4,16,32]`
4. [ ] Edge cases: non-divisible, min size, asymmetric, more-cores-than-work,
       single-core
5. [ ] Compare with `torch.testing.assert_close` against the original only
6. [ ] Per-dtype `TOL`; matmul-style `tl.dot` uses fixture rtol (~1e-2), not 1e-5
7. [ ] Single-tile cases assert `atol=0, rtol=0`
8. [ ] `torch.manual_seed(42)` everywhere; random inputs
