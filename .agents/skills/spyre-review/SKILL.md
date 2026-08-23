---
name: spyre-review
description: "Review a Spyre-aware kernel (spyre.py) for compliance with the authoring invariants (see _shared/invariants/), scratchpad utilization, Spyre-compiler descriptor patterns, and correctness vs original. Use when asked to review/verify a Spyre kernel. For a plain descriptor-API review (no invariants), use td-review."
---

# Spyre Kernel Review Skill

Review a Spyre-aware kernel for the authoring invariants, scratchpad
utilization, Spyre-compiler descriptor patterns, and equivalence to the
original. For a descriptor-API-only review (no invariants), use `td-review`.

## Trigger

Use when asked to review/verify/validate a `spyre.py`.

## Pre-flight

Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md). If
the KB documents constraints newer than this procedure, flag the discrepancy in
the report.

## Inputs

- **kernel_name**: directory under `kernels/`
- Review: `kernels/<name>/spyre.py`; compare against `kernels/<name>/original.py`

## Review procedure

### Step 1 — Invariants

Read [`../_shared/invariants/`](../_shared/invariants/) and check the kernel
against **every** file in it. Do not work from a fixed list — the set may grow
or shrink, so enumerate the directory each time. Each invariant file states its
own rule, verification steps, and red flags; apply them as written and report
PASS/FAIL per invariant.

### Step 1b — Scratchpad utilization (WARN, not FAIL)

After the hard constraint, assess under-use per
[`../_shared/spyre/scratchpad-batching.md`](../_shared/spyre/scratchpad-batching.md).
If peak live bytes are below ~10% of 2 MB and a loop axis processes one item per
iteration with an independent reduction, recommend batching `BLOCK_ITEMS`, give a
value, and note:
- the **16-byte escape** side benefit (cross-reference any unresolved scalar
  descriptor from Step 2 — batching is one way to resolve it, per
  `descriptor-rules.md` §4);
- the **divergent-control-flow** trade-off (feasible-with-masking, not a
  blocker);
- the **distribution-granularity** trade-off (`cdiv(total_work, BLOCK_ITEMS) ≥ 32`).

### Step 2 — Descriptor API usage

Per [`../_shared/descriptor-rules.md`](../_shared/descriptor-rules.md): no
`make_block_ptr`/`advance`/`order`/`boundary_check`; valid shape/strides/
block_shape; runtime `shape`, constexpr `block_shape`; offsets are block
multiples. Flag **redundant tail masking** as WARN (with the non-identity-fill
exception — a missing non-zero identity is a correctness FAIL; check the
original's `other=`). Per `descriptor-rules.md` §4 a scalar descriptor must be
resolved (last dim ≥ 16 bytes) or the kernel declared non-portable — a surviving
scalar descriptor or stale gap annotation is a FAIL.

### Step 2b — Physical layout markers

For stick-tiled tensors, verify the `tl.spyre_tensor_layout` markers (see
[`../_shared/spyre/tensor-layout-marker.md`](../_shared/spyre/tensor-layout-marker.md)):

- **Present** on each stick-tiled descriptor; **inline literal** or a `constexpr`
  arg (a list bound to a plain local is a compile error — FAIL). A `constexpr`
  guard should read `if X_LAYOUT is not None:`.
- **Well-formed**: one entry per physical dim; `stick-on-X` form
  `[(X,"floordiv",S), other, (X,"mod",S)]`; `src` indices in range for the
  logical rank; a repeated logical dim appears **only** as exactly one `floordiv`
  + one `mod` (two identities, two floordivs, or a three-way split are all
  rejected by the op verifier).
- **Stick divisor** `S = 128 // dtype_bytes` (64 fp16/bf16, 32 fp32, 128 fp8) —
  a hard-coded `64` on a non-2-byte dtype is a FAIL.
- **Stickified extents are multiples of `S`** — this one is a *silent* failure
  (an out-of-bounds physical view, no diagnostic), so check it by hand rather
  than trusting the compiler. A stick dim whose *block* extent is smaller than
  `S` is rejected (`a stick dim cannot be sub-stick`).
- **Axis matches intent**: the marked axis is the one actually stick-tiled.
  Stick-tiling a parallel/output axis synthesizes a scatter loop over its sticks;
  stick-tiling the contraction axis synthesizes a K-reduction loop.
- **Operands of the tiled op only** are marked. Both sides of a stickified
  contraction axis must be marked with the same stick size (marking one side is
  rejected). Logical intermediates (a softmax result, a scratchpad) and
  elementwise addends (a bias / additive mask) must be **unmarked** — a marked
  addend fails as an `arith.addf` type mismatch.
- **A transposed operand SHOULD be marked.** The pass absorbs an operand-side
  transpose (`retypeChain` reinterprets roles via `dimRoles` and erases it), so
  marker + `tl.trans` is the correct way to write `Q·Kᵀ` — *not* a defect.
  Chained transposes compose. A transpose *between the contraction and a marked
  store* is preserved deliberately; do not flag it as redundant.
- **Multi-stick parallel output is supported** (the pass tiles the accumulator
  with an outer scatter loop), so a parallel dim spanning several sticks is fine.
  But all marked operands must agree on which output axis is scattered —
  `operands disagree on the parallel multi-stick scatter` is the diagnostic.
- **Batched matmul**: for an N-D descriptor whose leading axis is a batch/head
  dim, that axis is an **identity dim** (a bare `src` int), and only the inner
  matmul axis is stick-tiled — a stick entry on the batch axis is a FAIL. More
  than one leading batch dim is not dispatched.
- **Enclosing loops in block units**: a loop whose IV feeds a marked stick dim is
  rescaled by the pass (bounds *and* step). A kernel that pre-multiplies by the
  stick count itself double-scales — FAIL.
- **Gather**: the indirect (row) dim of a gather must not be stick-split
  (`stick-splitting the indirect (gather) row dim is not supported`).
- **Dynamic extent**: a runtime-sized axis appears in `shape` only; `strides` and
  `block_shape` for that descriptor stay compile-time constant — a runtime value
  in `block_shape`/`strides` is a FAIL.
- **Output descriptor** carries a marker when the store must scatter into sticks.
- Because `RewriteDescriptorLayout` closes the layout gap, a *missing* marker
  where one is needed is a **FAIL**, not a `# [gap]` (reconcile with Step 3 /
  `gap-handling.md`).

### Step 3 — Spyre-compiler descriptor patterns


The set of compiler descriptor gaps changes as the lowering evolves, so this
step does not enumerate them — see
[`../_shared/spyre/gap-handling.md`](../_shared/spyre/gap-handling.md)
for the contract and the live sources (KB, `torch-spyre/triton` issues/docs)
where the current gaps are tracked.

What to verify is the **handling**, not a fixed list: every site that hits a
compiler gap stays in clean descriptor-first form with a `# [gap]` annotation,
**not** a baked-in raw-pointer workaround or invented layout. A workaround
shipped in place of the clean form is the FAIL.

### Step 4 — Removed GPU patterns

- [ ] No `@triton.autotune`, `tl.assume`, `tl.multiple_of`, CUDA/HIP config
- [ ] No `order=`, `boundary_check=`, `padding_option=`

### Step 5 — Correctness vs original

Same math, accumulator dtype, output dtype conversion, reduction structure; no
dropped operations; all original tensors handled.

### Step 6 — Conversion notes (WARN, not FAIL)

`spyre-convert` requires `kernels/<name>/conversion-notes.md` with a
`## Spyre-aware conversion` section recording the conversion decisions
(invariant handling, scratchpad batching, gap sites, signature changes). Verify
it exists and reflects the kernel as reviewed. Flag a **WARN** if the file is
missing, lacks the Spyre-aware section, or is stale relative to the code —
documentation hygiene, not a correctness defect, so it does not by itself make
the kernel non-compliant.

## Report format

```
## Spyre Compliance Review: <kernel_name>

### Invariants
- <one line per invariant file in `_shared/invariants/`>: PASS/WARN/FAIL — <evidence, verdict>

### Scratchpad utilization
**Status:** OK / WARN
- Peak live bytes / 2MB: <%>; batchable axis: <name/none>;
  recommended BLOCK_ITEMS: <value>; side benefits/caveats: <...>

### Descriptor API usage
**Status:** PASS / WARN / FAIL — <block_ptr, redundant masking, unresolved scalar descriptors>

### Physical layout markers
**Status:** PASS / WARN / FAIL / N/A — <presence, well-formedness, stick divisor,
stick-multiple extents, marked set (operands only), batch/transpose handling>

### Spyre-compiler descriptor patterns
**Status:** PASS / FAIL — <gaps in clean form + `# [gap]` annotation, or baked-in workaround>

### Correctness vs original
**Status:** PASS / FAIL — <discrepancies>

### Conversion notes
**Status:** OK / WARN — <present and current / missing / stale>

### Overall: COMPLIANT / NON-COMPLIANT
```

Conversion is `spyre-convert`; numerical tests are `spyre-test`.
