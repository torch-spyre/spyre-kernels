# `tl.spyre_tensor_layout` — physical stick-layout marker

> **Spyre-family only.** `tl.spyre_tensor_layout` is a Triton builtin from
> `torch-spyre/triton`; it is not in stock PyPI Triton, so KTIR generation must
> use the spyre build — see **Dependency** at the bottom.

## What it does

The descriptor stays **logical** (`shape`/`strides`/`block_shape` in math
dimensions). The marker declares how that tensor is **physically stick-tiled** in
device memory. The compiler's `RewriteDescriptorLayout` pass reads the marker and
synthesizes the physical loops (slice sticks, run the per-tile op, accumulate).
`tl.dot` and the descriptor are otherwise untouched — there is **no** reshape
glue (contrast the physical-descriptor variant).

## Syntax

```python
@triton.jit
def kernel(..., LAYOUT: tl.constexpr):
    tl.spyre_tensor_layout(desc, LAYOUT)

# Host/lowering config: LAYOUT = [<one entry per physical dim>]
```

Entry forms (the OpSpec `device_coordinates` map):

| Entry | Meaning |
|-------|---------|
| `src` (bare int) | identity — this physical dim = logical dim `src`, unchanged |
| `(src, "floordiv", S)` | stick **index**: `logical[src] // S` |
| `(src, "mod", S)` | within-stick **lane**: `logical[src] % S` |

- **`src` is the logical dimension index** being addressed. The same-looking
  marker names a different axis on different operands: K is dim 1 of `A[M,K]` but
  dim 0 of `B[K,N]`.
- **`S = 128 // dtype_bytes`** — the DataStick is 128 bytes, so `S` is
  **64** (fp16/bf16), **32** (fp32), **128** (fp8). Never hard-code 64. Upstream
  has a canonical helper, `test/utils.py::sticksize(signature, key)`
  (`STICK_BYTES // itemsize`); fixtures bind it once as
  `_SS = functools.partial(sticksize, _SIG_SPYRE)`.

In the IR the marker lowers to three **parallel array attributes** —
`phys_src`, `phys_op`, `phys_arg` — one entry per physical dim, where
`phys_op` is `0`=identity, `1`=floordiv, `2`=mod. Worth knowing because that is
the form the verifier's diagnostics talk about:

```mlir
tt.spyre_tensor_layout %desc {phys_src = array<i64: 1, 0, 1>,
                              phys_op  = array<i64: 1, 0, 2>,
                              phys_arg = array<i64: 64, 0, 64>}
```

## Layout convention

`stick-on-X` factors dimension `X` into `(X // S, X % S)`. The **conventional**
physical dim order is `[X//S, other, X%S]` — stick index first, lane last:

```python
# stick-on-X  ->  [(X, "floordiv", S), other_logical, (X, "mod", S)]
[(0, "floordiv", 64), 1, (0, "mod", 64)]   # A[M,K] stick-on-M
[(1, "floordiv", 64), 0, (1, "mod", 64)]   # B[K,N] stick-on-N
[(1, "floordiv", 64), 0, (1, "mod", 64)]   # C[M,N] stick-on-N
```

**That order is a convention, not a requirement.** `classify()` locates the lane
by *scanning for the `Mod` entry*, not by position, so identity dims may lead and
the lane need not be last. The real Spyre activation/weight layout relies on this
— `bmm_spyre_stick_activation` puts an identity dim first:

```python
# A[B,M,K] stick-on-K: phys [M, K/S, B, K%S]  — M leading, batch sandwiched
"A_LAYOUT": [1, (2, "floordiv", S), 0, (2, "mod", S)]
# B[B,K,N] stick-on-N: phys [N/S, B, K, N%S]
"B_LAYOUT": [(2, "floordiv", S), 0, 1, (2, "mod", S)]
```

Use the conventional order unless you are matching a specific hardware layout.

## Stickified extents must be stick-multiples; blocks must be ≥ one stick

Every extent you stick-tile must stay a multiple of `S`. A ragged
(non-divisible) split **silently produces an out-of-bounds physical view** — it
is not diagnosed. Pad on the host instead. (Upstream fixture comment: sweeping
onto a non-multiple "would encode broken behavior rather than test it".)

Separately, the `block_shape` extent on a stick dim must be **at least `S`** — so
a stickified dim's `BLOCK_*` may not be sub-stick:

```
spyre_tensor_layout: block extent of stick dim (32) is smaller than the
stick size (64); a stick dim cannot be sub-stick
```

## Dispatch model

The pass assigns a **role to every physical dim** and synthesizes loops from the
collection of roles. From `Classify.h`:

> Assign a role to each physical dim of an operand. `>= 0` : parallel dim, maps
> to output axis `[value]`; `-1` : reduction dim.

`ClassifiedDims` (`Types.h`) then splits those into `floorDims` (parallel
stick-index dims), `reduceDims`, `opInnerDim` (the rightmost reduce dim),
`loopDims`, and `opTileDims`, with a per-dim `SliceKind` of `StickIndex`,
`StickifiedBlock`, or `WholeBlock`.

The useful author-facing summary:

- Stick-tiling a **parallel** (output) axis → a **parallel scatter loop** over
  that axis's sticks.
- Stick-tiling the **contraction** axis → a **reduction loop** over K-sticks.
- Both at once is supported (they index independently), subject to the
  single-parallel-axis rule below.

You still do not pick the loop structure; it follows from which axes you mark.

## Multi-stick parallel output

A marked operand whose **parallel** dim spans more than one stick is supported:
the pass tiles the accumulator, wrapping the reduction in an outer scatter loop
that `extract_slice`s the parallel slab and `insert_slice`s the result back
(`rewrite-descriptor-layout-parallel-multistick.mlir`). Two consequences:

- An extent-1 parallel floor dim costs nothing: "the scatter loop inlines at
  trip <= 1", emitting exactly what the reduction-only path emits.
- **All marked operands must agree on which output axis is scattered.** One
  `(factor, axis)` pair is carried, so markers that want scatter loops on two
  different output axes are rejected:

```
spyre_tensor_layout: operands disagree on the parallel multi-stick scatter
```

## Enclosing loops are rescaled from block units to stick units

If your own loop's induction variable feeds the stick (floor) dim of a marked
descriptor, the pass consumes that IV **directly as the physical stick index**
and rewrites all three bounds together — `lower`, `upper` **and `step`**, each
multiplied by `factor` = sticks per block
(`rescaleEnclosingLoop`, `rewrite-descriptor-layout-loop-rescale.mlir`):

> `N=768, BLOCK_N=128, stick=64 => 2 sticks per block => factor 2.`
> Loop `for n = 2 to 6 step 1` covers blocks `{2,3,4,5}` = sticks `[4, 12)`.
> Rescaled loop must be `for = 4 to 12 step 2`.

So **write the loop in block units** and let the pass rescale — do not
pre-multiply by the stick count yourself. `factor == 1` is a no-op. Non-zero
lower bounds and non-unit steps are handled.

## Transposes

- **On an operand, a transpose is absorbed.** Mark the operand like any other and
  transpose it with `tl.trans` — the operand is a load "whose physical shape
  already encodes where each logical axis lives, so `retypeChain` can reinterpret
  roles via `dimRoles` and **erase the transpose**". This is the normal way to
  write `Q·Kᵀ`: mark `K` *and* `tl.trans` it. (Only the trailing two dims are
  transposed; a batch axis stays put: `tl.trans(k, (0, 2, 1))` for
  `[batch, N, K]`.)
- **Chained transposes compose.** Several `tl.trans` ops on one operand chain —
  even with elementwise ops between them — compose into a single net permutation
  (`rewrite-descriptor-layout-chained-transpose.mlir`); two `[1,0]`s cancel to
  identity.
- **After the contraction, a transpose survives verbatim.** A `tl.trans` between
  the matmul and a marked store is neither erased nor folded into the sink's own
  permutation: "After a contraction there is no physical layout left to
  reinterpret… the user transpose is the ONLY thing producing the requested
  order. That is why `retypeChain` stops at `isContractionOp`."

## Mark memory operands of the tiled op — leave everything else logical

The pass physicalizes a marked descriptor into stick tiles and orients it into
canonical form for the consuming op. Rules:

- **Both operands of a *multi-stick* stickified contraction axis must be marked**,
  with the same stick size. The check fires only when the annotated operand's
  reduction axis actually spans more than one stick; the diagnostic is
  `operands share a stickified contraction axis but not all are annotated — any
  two operands sharing a stickified contraction axis must both carry a
  tt.spyre_tensor_layout marker with the same stick size on that axis`. If the
  shared contraction axis is single-stick or unstickified, an **unmarked logical
  scratchpad operand is legal** — that is exactly the attention `P·V` case
  (`@attn_pv`: "P is unannotated (a logical scratchpad), which is legal here
  because the shared contraction axis (K) is not stickified").
- **Logical intermediates stay unmarked.** A value produced inside the kernel
  (a softmax result `P` feeding `P·V`, or the `bc` scratchpad in upstream's
  `chained_matmul_kernel` — "the only logical intermediate (pure register value,
  no descriptor)") has no descriptor and is not marked. The pass has a dedicated
  `classifyScratchpad` path for "a scratchpad operand (no marker, logical
  shape)".
- **Elementwise addends are not tiled operands — do not mark them.** A bias or
  additive mask added to the result is not stick-tiled the same way; marking it
  physicalizes the load to a stick tile that fails to add to the logical result
  (`arith.addf` type mismatch). Leave it logical.
- Marking an operand whose parallel dim is single-stick while leaving the other
  operands unmarked is legitimate (see the multistick tests, which mark only `A`).

### The real elementwise rule: all operands of an elementwise op must agree

`retypeChain` walks forward along **operand 0 only**. So when a marked tensor
feeds an elementwise op, any *sibling* operand whose producer the walk never
reaches keeps its logical rank and the op fails to verify. The rule that actually
holds is: **every operand of an elementwise op must physicalize identically.**

- ✅ `vector_add__2d_spyre_stick` marks **all three** of `x`/`y`/`out` stick-on-N;
  the add stays pure elementwise on rank-3 tiles.
- ❌ A reduce-then-broadcast against a marked tensor breaks — this is why
  **`softmax/` has no layout variant at all**. Upstream records it: the loaded row
  physicalizes to rank 3 but "the broadcast of `row_max` stays rank 2, so
  `row - row_max` fails with `'arith.subf' op requires the same type for all
  operands and results`".

Treat a softmax-shaped pattern (a reduction whose broadcast result feeds an
elementwise op against a marked tensor) as an **unsupported compiler pattern**,
not an authoring mistake.

## `tl.inter_tile` and layout markers do not compose

This limitation is tracked by
[`torch-spyre/triton#87`](https://github.com/torch-spyre/triton/issues/87).
Use `tl.inter_tile` and `tl.spyre_tensor_layout` in separate kernel variants.
A marked value flowing through `tl.inter_tile` does not lower correctly:
`RewriteDescriptorLayout` physicalizes the partial and the `tt.inter_tile`
result, but the op's sibling `identities` operand stays at logical rank.
`LowerInterTile` forwards that identity while deriving result types from the
physicalized partial, and verification fails:

```text
'ktdp.inter_tile_reduce' op failed to verify that identity types must
match result types
```

Therefore:

- do not pass a marked/physicalized value to `tl.inter_tile`;
- represent layout-marker and inter-tile implementations as separate variants;
- lower and test each variant independently; and
- classify a kernel requiring both operations on the same value as unsupported.

## `tl.reduce` support

Markers are not matmul-specific. A marked input to a reduction gets the same
treatment: `A[64,128]` stick-on-N with `N=128, S=64` yields 2 N-sticks and the
pass synthesizes "`scf.for` over 2 sticks, `extract_slice` + `linalg.reduce`
inside" (`rewrite-descriptor-layout-reduce.mlir`). Stick-tiling the reduced axis
gives a reduction loop exactly as it does for a contraction.

## Batched matmul — an identity batch dim (N-D descriptors)

A descriptor may carry a **leading batch axis** that the matmul iterates over
independently. Make the batch axis an **identity dim** (a bare `src` int) so it
passes through untouched, and stick-tile only the inner axis:

```python
# Host/lowering config for A[BATCH, M, K] stick-on-K; dim 0 is the batch.
A_LAYOUT = [(2, "floordiv", S), 0, 1, (2, "mod", S)]
# Kernel body: tl.spyre_tensor_layout(a_desc, A_LAYOUT)
```

`tl.dot` over the **trailing two dims** lowers to `linalg.batch_matmul`. The bare
`0` keeps the batch axis out of the stick factoring.

- Both operands must agree on every shared parallel extent, including batch.
  The pass does not check this itself; a mismatch surfaces as a downstream
  `linalg.batch_matmul` verifier error.
- Only rank-3 logical dot operands are dispatched to `linalg.batch_matmul`.
  Fold multiple logical batch axes into one leading dimension on the host.
  Physical rank is independent and may be higher (rank-5 layouts are covered).

## Gather / scatter

A gather's **indirect (row) dim cannot be stick-split**:

```
spyre_tensor_layout: stick-splitting the indirect (gather) row dim is
not supported
```

Mark the non-indirect dims only, or keep a gathered operand logical.

## Output descriptor drives the store sink

Marking the **output** descriptor triggers the store **sink stage** — the logical
result is scattered into the physical stick buffer via `tensor.insert_slice`.
Leave the output unmarked and the store stays logical (sink is a no-op).

## Dynamic descriptor extent and strides

An extent may be a runtime `i32` arg rather than a `constexpr`, so one lowered
kernel serves any size along that axis (e.g. per-request sequence length):

```python
q_desc = tl.make_tensor_descriptor(
    Q, shape=[HEADS, SEQ, Lk], strides=[stride_h, stride_s, 1],  # SEQ runtime i32
    block_shape=[1, BLOCK_M, DMODEL],                            # block_shape constexpr
)
```

The runtime axis lowers to a `?` (kDynamic) memref dim; a floordiv dim over it
becomes an `arith.ceildivsi`. A loop bounded by the runtime size lowers to an
`scf.for` with a runtime trip count.

- **`block_shape` must be compile-time constant.** The coord map is applied to the
  *block* shape and must yield all-static extents, else:
  `spyre_tensor_layout: cannot derive static block_shape`.
- **`strides` may be dynamic.** A runtime `i64` stride works in both modes —
  device mode emits a `muli` chain, host mode passes it through
  (`@dynamic_strides_device` in `rewrite-descriptor-layout-advanced.mlir`).

## Pass the layout as a `tl.constexpr` arg — NOT an inline literal

**A layout containing stick-split entries must arrive as a `tl.constexpr` kernel
argument.** An inline literal at the call site raises.

```python
@triton.jit
def k(..., A_LAYOUT: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(...)
    if A_LAYOUT is not None:                    # ✅ the working form
        tl.spyre_tensor_layout(a_desc, A_LAYOUT)
```
```python
    # ❌ inline literal -> TypeError: int() argument must be ... not 'tuple'
    tl.spyre_tensor_layout(a_desc, [(0,"floordiv",64), 1, (0,"mod",64)])
    # ❌ bound to a local -> CompilationError: cannot convert floordiv of
    #    type <class 'str'> to tensor   (visit_Assign tries to_tensor on the str)
    lay = [(0,"floordiv",64), 1, (0,"mod",64)]
    tl.spyre_tensor_layout(a_desc, lay)
```

Why: `Semantic._parse_coord_entry` gates on `isinstance(entry, (tuple, list))`
using the **Python builtins** (`semantic.py` does `from . import core as tl` and
never shadows those names). But the JIT frontend's `visit_List` / `visit_Tuple`
both return `language.tuple(...)`, and `core.tuple(base_value)` is *not* a
subclass of the builtin. So a nested entry from an inline literal fails the
isinstance test, falls through to the bare-int branch, and `int()` raises on it.
A `constexpr` arg is never walked by the frontend, so it stays a plain Python
list of plain tuples and parses correctly.

A layout of **only bare ints** (all-identity, no stick split) does survive an
inline literal — the entries are `constexpr` scalars, not tuples — but there is no
reason to rely on that; use the constexpr-arg form uniformly.

Host side, build the value with the `sticksize` helper:

```python
_SS = functools.partial(sticksize, _SIG_SPYRE)
"A_LAYOUT": [(1, "floordiv", _SS("a_ptr")), 0, (1, "mod", _SS("a_ptr"))]
```

Two additive entry spellings also work: `(src, "identity")` (a 2-tuple,
equivalent to a bare int) and the raw op code `0`/`1`/`2` in place of the keyword
string.

## The op has a verifier — malformed markers are diagnosed

Malformed markers produce diagnostics
(`test/Triton/spyre-tensor-layout-invalid.mlir`). Quotable messages:

| Mistake | Diagnostic |
|---|---|
| arrays of unequal length | `phys_src, phys_op and phys_arg must have the same number of entries` |
| no entries | `must describe at least one physical dim` |
| bad op code | `phys_op[0] must be 0 (identity), 1 (floordiv) or 2 (mod), got 3` |
| `src` out of range / negative | `phys_src[0] must be in [0, 2), got 2` |
| logical dim repeated wrongly | `logical dim 0 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry), got 2 identity, 0 floordiv, 0 mod` |
| zero divisor / modulus | `phys_arg[0] must be > 0 for a floordiv/mod dim, got 0` |

So a repeated logical dim is legal **only** as exactly one `floordiv` + one `mod`
pair — never two identities, two floordivs, or a three-way split.

## ⚠ A marked kernel that compiles is not thereby correct

The pass has **no diagnostic for a consumer it cannot physicalize** (tracked by
`torch-spyre/triton#95`). When Phase 2 has no pattern for an op, that op is left
holding a physical-rank operand against a logical-rank signature and **the pass
still reports success**. This produces either a downstream verifier error that
does not mention layouts or incorrect numerical output.

A `linalg.generic` hand-written as a matmul is a representative trap: it declares
no `ContractionOpInterface`, so `isContractionOp` misses it, and it falls through
the elementwise retype path where its result type gets overwritten with operand
0's shape.

So **never treat "it lowered" as verification.** Always pair a marked kernel with:

1. structural assertions on the generated KTIR — no surviving
   `tt.spyre_tensor_layout`, the expected `linalg.matmul` / `linalg.batch_matmul`
   present; and
2. a numerical run on ktir-cpu against a reference.

## Post-conditions worth asserting in a test

- **No surviving markers.** `RewriteDescriptorLayout` "consumes and erases"
  every `tt.spyre_tensor_layout`, so asserting absence in the generated KTIR is
  valid. (`spyre-tensor-layout-survives.mlir` only checks the marker survives the
  passes that run *before* `RewriteDescriptorLayout` — it is not a
  counter-example.) Upstream fixtures assert
  `assert_absent("tt.spyre_tensor_layout")`.
- The expected op is present — `linalg.matmul`, or `linalg.batch_matmul` when a
  batch axis should batch, plus `scf.for` / `tensor.insert_slice` where a
  reduction or sink is expected.
- A dynamic extent shows as a `?` memref dim.

## Lowering options the pass reads

`SpyreOptions` (`third_party/spyre/backend/compiler.py`) carries two fields that
change marked-kernel output:

- **`data_layout`** — `"device"` (row-major strides over the *physical* shape) or
  `"host"` (physical strides derived from the *logical* strides through the coord
  map). Default is **`"device"`**, but **every upstream layout fixture sets
  `"host"`** — that is what makes a host-side row-major NumPy buffer match, i.e.
  what makes numerical ktir-cpu execution come out right. For a kernel validated
  against a NumPy reference, **`"host"` is almost certainly what you want.**
  Anything other than the two values raises. Worked example of the difference, for
  `[128,128]` stick-on-N with `S=64` → physical `[2,128,64]` and logical strides
  `[128,1]`: host gives strides `[64, 128, 1]`, device gives `[8192, 64, 1]`
  (`rewrite-descriptor-layout-advanced.mlir`). Via `spyre-triton-opt` the flag
  spelling is `--rewrite-descriptor-layout=data-layout=host`.
- **`required_fixes`** — optional correctness patches spliced into the TTIR→KTIR
  pipeline as `{fix pass: core pass it runs after}`, e.g.
  `{"convert_elementwise_to_linalg": "lower_compute_ops"}`. The anchor must be
  one of the core passes (`lower_descriptor_memory`, `lower_scalar_load`,
  `lower_compute_ops`, `rewrite_descriptor_layout`, `lower_inter_tile`,
  `convert_functions`). **A bad anchor is silently ignored and the fix never
  runs** — a missing pass *binding* raises, a bad *anchor* does not.

  Available fix passes: `convert_elementwise_to_linalg` (anchor
  `lower_compute_ops`), `unalias_linalg_outs`, `drop_reduction_init_fill`
  (anchor `lower_compute_ops`), and `materialize_base_addresses` (anchor
  `convert_functions`; its `base_addresses` are **element indices, not bytes**).
  `unalias_linalg_outs` is documented with anchor
  `convert_elementwise_to_linalg`, which is not a core pass, while the composition
  loop only honours core-pass anchors; do not rely on this pairing without an
  explicit lowering test.

> **This repo's lowering path:** `scripts/_spyre/round_trip.py`'s
> `make_ktir_mod` forwards only `grid`. It does not accept `data_layout` or
> `required_fixes`; marked kernels lowered through `scripts/gen_ktir.py` therefore
> use `data_layout="device"` and no fix passes. Kernels requiring host-layout
> interpretation or a fix pass are unsupported by this path.

## Known compiler gaps

- `tt.addptr` into `tl.make_tensor_descriptor` is not lowered by
  `LowerDescriptorMemory` (upstream `bmm_addptr` fixtures are `disabled`, pinned
  by `test_lower_desc_memory.py::TestAddptrIntoDescriptor`). Build per-batch
  descriptors from a marked N-D descriptor with an identity batch dim instead.

## Dependency

`tl.spyre_tensor_layout` exists only in the `torch-spyre/triton` build, not in
stock PyPI Triton. This repo pins the spyre-Triton rev in
`.github/workflows/ci.yaml` (`SPYRE_TRITON`), `README.md`, and
`scripts/_spyre/round_trip.py` at `0ddc67b8`. This revision provides the builtin,
`RewriteDescriptorLayout` (`Classify.cpp`, `ContractionSynthesis.cpp`,
`PermutationUtils.h`), the op verifier, multi-stick parallel scatter, and loop
rescaling. The pin is **coupled** to the `ktir-cpu` rev in `pyproject.toml` via
`ktir-mlir-frontend` — see the README note. The base tier uses PyPI Triton and
cannot compile the marker; marker tests use `gen_ktir.py` plus ktir-cpu.

Source of truth: `torch-spyre/triton`
`third_party/spyre/lib/Dialect/KTDP/Transforms/RewriteDescriptorLayout*`, the
`test/Conversion/rewrite-descriptor-layout-*.mlir` lit tests, and
`test/fixtures/matmul/`.
