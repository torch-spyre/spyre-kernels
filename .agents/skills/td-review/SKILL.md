---
name: td-review
description: "Review a tensor-descriptor Triton kernel (tensor_descriptor.py) for correct tl.make_tensor_descriptor usage — deprecated APIs removed, valid shape/strides/block_shape, no redundant masking, 16-byte rule, correctness vs original. Backend-agnostic; does not check Spyre invariants. Use spyre-review for the full Spyre compliance review."
---

# Tensor-Descriptor Review Skill

Review a kernel converted by `td-convert` for correct use of the
`tl.make_tensor_descriptor` API and equivalence to the original.
## Trigger

Use when asked to review/verify a `tensor_descriptor.py` (`_td`) kernel.

## Pre-flight

Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md).

## Inputs

- **kernel_name**: directory under `kernels/`
- Review: `kernels/<name>/tensor_descriptor.py`
- Compare against: `kernels/<name>/original.py`

## Step 0 — Portable or not?

If `tensor_descriptor.py` is **absent**, td-convert declared the kernel
non-portable. There is no code to review — review the *verdict* instead. Open
`conversion-notes.md` and check the `## Tensor-descriptor conversion` section
claims non-portability and names a blocking access. Then verify the wall is
real against `original.py` and `descriptor-rules.md` §4–§5:

- **§4 wall** — an un-widenable last dim. Confirm the named access is a genuine
  scalar on a 1D tensor (or otherwise cannot reach ≥ 16 bytes by batching or
  layout change). If a second dim *could* widen it, the verdict is **wrong** —
  the kernel is convertible. FAIL the verdict.
- **§5 wall** — multiplicative mixing, dimension fusion, non-delinearizable, or
  ≥ 2 indirect dims with no single-`gather` form. Confirm the index genuinely
  cannot be expressed as `desc.gather`/`desc.scatter` on a multi-dim descriptor.
  A per-row/per-program index into a structured tensor is portable — a verdict
  citing that is **wrong**. FAIL the verdict.

Report **VERDICT: non-portable CONFIRMED** (wall real, named, no convertible
form) or **VERDICT: WRONG** (a descriptor form exists — name it). Then stop;
skip Steps 1–4 (no code), still run Step 5 on the notes.

If `tensor_descriptor.py` is **present**, proceed to Step 1.

## Review procedure

### Step 1 — Descriptor API usage

Against [`../_shared/descriptor-rules.md`](../_shared/descriptor-rules.md) §1–3,
check:

1. **No `tl.make_block_ptr`**, no `tl.advance`, no `order` / `boundary_check` /
   `padding_option` (block_ptr patterns).
2. **No raw pointer arithmetic** feeding loads/stores.
3. Every descriptor has valid `shape`, `strides`, `block_shape`.
4. `shape` uses runtime args for problem dims; `block_shape` uses constexprs.
5. Load/store offsets are multiples of block dims: `[i * BLOCK_M, j * BLOCK_N]`.

### Step 2 — Redundant tail masking

The descriptor `shape` already carries the boundary. A kernel that *also* keeps
the original's `mask` / `tl.where` / `other=` is re-zeroing already-zero lanes —
a common conversion artifact. Flag as **WARN** (cleanup) and recommend deleting.

```python
# WARN — redundant: descriptor load already zero-filled lanes >= N
x = in_desc.load([tile * BLOCK])
offs = tile * BLOCK + tl.arange(0, BLOCK)
acc += tl.sum(tl.where(offs < N, x, 0.0))

# CLEAN
x = in_desc.load([tile * BLOCK])
acc += tl.sum(x)
```

**Exception — non-identity fills.** Zero-fill is the identity only for additive
reductions. If the tail needs `1.0` (product) or `-inf` (max), an explicit
fix-up IS required — do not flag it; instead verify it uses the correct
identity. Dropping the mask *and* needing a non-zero identity is a correctness
**FAIL** — check against the original's `other=`.

### Step 3 — 16-byte last-dim minimum

Per `descriptor-rules.md` §4, a scalar descriptor (`block_shape=[1]` or
`[..., 1]`) will not trace and must be resolved, not annotated. Verify the
conversion widened the last dim to ≥ 16 bytes — or, where that was impossible,
declared the kernel non-portable. A surviving scalar descriptor, or a stale
`# [gap] scalar descriptor` annotation, is a **FAIL**.

### Step 4 — Correctness vs original

Same math, same accumulator dtype (usually f32), same output dtype conversion,
same reduction structure, no dropped operations (activations, clamps,
normalization), all original tensors handled.

### Step 5 — Conversion notes (WARN, not FAIL)

`td-convert` requires `kernels/<name>/conversion-notes.md` with a
`## Tensor-descriptor conversion` section recording what changed and why. Verify
it exists and reflects the kernel as reviewed (signature changes, dropped masks,
scalar-load resolution). Flag a **WARN** if the file is missing, lacks the TD
section, or is stale relative to the code — documentation hygiene, not a
correctness defect, so it does not by itself make the kernel non-compliant.

## Report format

Non-portable verdict (no `tensor_descriptor.py`):

```
## Tensor-Descriptor Review: <kernel_name>

### Portability verdict
**VERDICT: non-portable CONFIRMED / WRONG**
- Blocking access: <named access + §4/§5 rule>
- Real wall / convertible via <desc.gather|batched layout|...>

### Conversion notes
**Status:** OK / WARN
- <present and current / missing / stale>

### Overall: COMPLIANT (correctly non-portable) / NON-COMPLIANT (convertible)
```

Converted kernel (`tensor_descriptor.py` present):

```
## Tensor-Descriptor Review: <kernel_name>

### Descriptor API usage
**Status:** PASS / WARN / FAIL
- block_ptr / advance / order: <absent / found at ...>
- Raw pointer loads: <count>
- Redundant tail masking: <none / WARN at <location>>
- 16-byte last dim: <ok / FAIL: scalar descriptor at ... — resolve or declare non-portable>

### Correctness vs original
**Status:** PASS / FAIL
- <discrepancies>

### Conversion notes
**Status:** OK / WARN
- <present and current / missing / stale — what's absent>

### Overall: COMPLIANT / NON-COMPLIANT
```
