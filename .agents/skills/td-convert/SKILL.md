---
name: td-convert
description: "Convert a raw-pointer or block_ptr Triton kernel to the tl.make_tensor_descriptor API, producing a tensor_descriptor.py. Backend-agnostic — no Spyre invariants. Use when asked to port a kernel to tensor descriptors or modernize block_ptr/raw-pointer loads."
---

# Tensor-Descriptor Conversion Skill

Modernize a GPU-shaped Triton kernel that uses raw pointer arithmetic or the
deprecated `tl.make_block_ptr` API into one that uses `tl.make_tensor_descriptor`.

This is a **backend-agnostic** transformation: it changes *how memory is
addressed*, not *how work is distributed*.

## Trigger

Use when the user asks to convert a kernel to tensor descriptors, port
`block_ptr`/raw-pointer loads to `make_tensor_descriptor`, or produce a
`tensor_descriptor.py` from an `original.py` / `block_ptr.py`.

## Pre-flight

Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md).

## Inputs

- **kernel_name**: directory under `kernels/` (e.g. `rms_norm`, `matmul`)
- Source: `kernels/<name>/original.py` (and `block_ptr.py` if present)
- Output: `kernels/<name>/tensor_descriptor.py`
- Kernel function named `_<name>_kernel_td`

## Procedure

0. **Scan every access and pick its descriptor form.** Before writing anything,
   list the original's loads/stores and map each to a descriptor op — most are
   not the plain `desc.load`/`desc.store` (see `descriptor-rules.md`):
   - affine/contiguous tile → `desc.load([off...])` / `desc.store(...)` (§1)
   - **data-dependent gather/scatter** — `tl.load(ptr + idx)` where `idx` was
     loaded at runtime (e.g. `token_id = tl.load(...)` then
     `tl.load(row_ptr + token_id)`) → `desc.gather(indices, offset)` /
     `desc.scatter(value, indices, offset)` on a multi-dim descriptor (§5).
     This IS portable — do not mistake it for non-portable.

   A kernel is **non-portable** only if an access hits a genuine wall (§4–§5):
   a last dim that cannot reach ≥ 16 bytes, or an indirect index that mixes
   multiplicatively / fuses dims / is non-delinearizable (hash, `f(loaded)`).
   In that case say so, name the blocking access, and stop — do NOT ship a hybrid
   that leaves a raw `tl.load`/`tl.store` in and annotates it (violates
   checklist item 1). But reach for `desc.gather`/`desc.scatter` before declaring
   defeat: a per-row/per-program index into a structured tensor is portable.

1. **Replace all pointer arithmetic / `block_ptr` with descriptors.** Follow
   [`../_shared/descriptor-rules.md`](../_shared/descriptor-rules.md) §1 — one
   descriptor per tensor, any rank, strides from the signature or computed for
   row-major.
2. **Drop redundant tail masks.** The descriptor `shape` is the boundary; do
   not port the original's `mask` / `tl.where` / `other=` machinery. See
   `descriptor-rules.md` §2 — including the non-identity-fill caveat (a product
   reduction with `other=1.0`, a max with `-inf` etc. still needs a tail fix-up).
3. **Respect the 16-byte last-dim minimum** (`descriptor-rules.md` §4). Resolve
   any scalar descriptor per that rule — widen the last dim to ≥ 16 bytes, or
   declare the kernel non-portable. Do not emit a `[..., 1]` descriptor.
4. **Preserve the kernel signature when possible** so the wrapper can call both
   forms with minimal branching. If the descriptor form needs a value the
   original derived from the grid (e.g. a row count), recover it from existing
   args first; add a new parameter only as a last resort and update the wrapper.
5. **Keep the grid as-is.** TD conversion does *not* cap or reshape the grid —
   one program per work item (e.g. `grid = (n_rows,)`) is fine.

## Output file structure

```python
# SPDX-License-Identifier: <same as original>
# SPDX-FileCopyrightText: <same as original>
#
# Tensor-descriptor conversion of <kernel_function_name>.
# Original: <source path>
# Changes summarized in kernels/<name>/conversion-notes.md.

import triton
import triton.language as tl


@triton.jit
def _<name>_kernel_td(..., BLOCK_...: tl.constexpr):
    """<Brief description.>"""
    ...
```

## Document the conversion

After writing the kernel, record what changed in
`kernels/<name>/conversion-notes.md` — a single file shared with `spyre-convert`
(this skill owns the tensor-descriptor section; spyre-convert appends its own).
Create the file if it does not exist:

```markdown
# <name> conversion notes

## Tensor-descriptor conversion

- Source: original.py / block_ptr.py → tensor_descriptor.py
- Pointer arithmetic / block_ptr replaced with tl.make_tensor_descriptor
- Tail masks dropped (descriptor shape carries the boundary)
- <any signature change, scalar-load resolution (batching / layout), or non-obvious decision>
```

Keep it short — the *why* behind non-obvious choices, not a line-by-line diff.
If the file already exists (e.g. a prior run, or spyre-convert ran first), edit
the `## Tensor-descriptor conversion` section in place and leave any
`## Spyre-aware conversion` section untouched.

## Checklist

0. [ ] Every access mapped to a descriptor op — affine → `desc.load/store`,
       data-dependent index → `desc.gather/scatter` (§5). Kernel declared
       non-portable + STOPPED only on a genuine §4/§5 wall (un-widenable scalar,
       multiplicative/fused/non-delinearizable index); never a raw-op hybrid.
1. [ ] All memory accesses use descriptors — no raw `tl.load`/`tl.store`, no
       `tl.make_block_ptr`, no `tl.advance`, no `order`/`boundary_check`
2. [ ] Redundant tail masks dropped (non-identity fills handled explicitly)
3. [ ] No scalar descriptors — last dim ≥ 16 bytes (batched/restructured), or
       kernel declared non-portable
4. [ ] Problem sizes are runtime args; tile sizes are `tl.constexpr`
5. [ ] Non-divisible shapes handled (`tl.cdiv` / `tl.minimum`)
6. [ ] Signature preserved where possible; wrapper updated if changed
7. [ ] Output at `kernels/<name>/tensor_descriptor.py`, kernel `_<name>_kernel_td`
8. [ ] `kernels/<name>/conversion-notes.md` updated with the TD section
