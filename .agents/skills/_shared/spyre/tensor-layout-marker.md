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
tl.spyre_tensor_layout(desc, [ <one entry per physical dim> ])
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

`stick-on-X` factors dimension `X` into `(X // S, X % S)` and the physical dim
order is **`[X//S, other, X%S]`** — stick index first, lane last:

```python
# stick-on-X  ->  [(X, "floordiv", S), other_logical, (X, "mod", S)]
tl.spyre_tensor_layout(a_desc, [(0, "floordiv", 64), 1, (0, "mod", 64)])  # A[M,K] stick-on-M
tl.spyre_tensor_layout(b_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])  # B[K,N] stick-on-N
tl.spyre_tensor_layout(c_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])  # C[M,N] stick-on-N
```

## Stickified extents must be stick-multiples

Every extent you stick-tile must stay a multiple of `S`. A ragged
(non-divisible) split **silently produces an out-of-bounds physical view** — it
is not diagnosed. Pad on the host instead. (Upstream fixture comment: sweeping
onto a non-multiple "would encode broken behavior rather than test it".)

Separately, the *block* extent of a stick dim may not be sub-stick:

```
spyre_tensor_layout: block extent of stick dim (32) is smaller than the
stick size (64); a stick dim cannot be sub-stick
```

## The dispatch model — roles per dim, not two fixed "cases"

Older notes framed this as a choice between "Case 1 parallel sticks" and
"Case 2 split-K reduction". That is no longer the whole picture: the pass
assigns a **role to every physical dim** and synthesizes loops from the
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

## Multi-stick parallel output — now supported

A marked operand whose **parallel** dim spans more than one stick works: the
pass tiles the accumulator, wrapping the reduction in an outer scatter loop that
`extract_slice`s the parallel slab and `insert_slice`s the result back
(`rewrite-descriptor-layout-parallel-multistick.mlir`). Earlier fixtures noted
"multi-output-stick scatter is not yet implemented" — that limitation is gone.
Two consequences:

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

- **Both operands of a stickified contraction axis must be marked**, with the
  same stick size. When a `tl.dot`'s contraction axis is stick-tiled, the pass
  windows *each* operand per stick and pairs the slices; it has no physical
  coordinate map for an unmarked operand, so the per-stick slice of the marked
  side no longer matches. Marking only one side is rejected with a diagnostic.
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

## Not only `tl.dot` — `tl.reduce` too

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
# A[BATCH, M, K] stick-on-K (contraction axis = logical dim 2); dim 0 is the batch
tl.spyre_tensor_layout(a_desc, [(2, "floordiv", S), 0, 1, (2, "mod", S)])
```

`tl.dot` over the **trailing two dims** lowers to `linalg.batch_matmul`. The bare
`0` keeps the batch axis out of the stick factoring.

- The batch extents of two operands **need not match**: a grouped case (GQA,
  where several query heads share a KV head) resolves head-sharing at the
  descriptor **load index**, not at the matmul batch dim.
- More than one leading batch dim is **not** dispatched — the pass carries one
  batch dim. Fold extra batch axes into a single leading dim on the host.

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

## Dynamic descriptor extent — a runtime size in `shape`

An extent may be a runtime `i32` arg rather than a `constexpr`, so one lowered
kernel serves any size along that axis (e.g. per-request sequence length):

```python
q_desc = tl.make_tensor_descriptor(
    Q, shape=[HEADS, SEQ, Lk], strides=[stride_h, stride_s, 1],  # SEQ runtime i32
    block_shape=[1, BLOCK_M, DMODEL],                            # block_shape constexpr
)
```

The runtime axis lowers to a `?` (kDynamic) memref dim. **`strides` and
`block_shape` must stay compile-time constant** — only full extents may be
dynamic. A loop bounded by the runtime size lowers to an `scf.for` with a runtime
trip count.

## Inline-only constraint

The layout list must be an **inline literal** at the call site. Binding it to a
plain local makes the `@triton.jit` code generator try to tensor-convert the
keyword strings → `CompilationError`:

```python
lay = [(0, "floordiv", 64), 1, (0, "mod", 64)]
tl.spyre_tensor_layout(a_desc, lay)                                   # ❌ raises
tl.spyre_tensor_layout(a_desc, [(0,"floordiv",64), 1, (0,"mod",64)])  # ✅ inline
```

Passing the layout as a **`tl.constexpr` kernel argument** is also fine (a
constexpr is not a runtime local) — this is how upstream fixtures parametrize
`A_LAYOUT`/`B_LAYOUT`/`C_LAYOUT`, applying them with:

```python
if A_LAYOUT is not None:
    tl.spyre_tensor_layout(a_desc, A_LAYOUT)
```

(Older notes used `if A_LAYOUT != 0:`; upstream now uses `is not None`.)

## The op has a verifier — malformed markers are diagnosed

A bad marker used to crash the pass; it now produces a diagnostic
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

- **`data_layout`** — `"device"` (stickified row-major physical strides) or
  `"host"` (strides derived from logical strides via the coordinate map).
  Default is **`"device"`**, but every upstream `spyre_stick_*` fixture sets
  **`"host"`**. Anything other than these two values raises.
- **`required_fixes`** — optional correctness patches spliced into the TTIR→KTIR
  pipeline as `{fix pass: core pass it runs after}`, e.g.
  `{"convert_elementwise_to_linalg": "lower_compute_ops"}`. The anchor must be
  one of the core passes (`lower_descriptor_memory`, `lower_scalar_load`,
  `lower_compute_ops`, `rewrite_descriptor_layout`, `lower_inter_tile`,
  `convert_functions`). **A bad anchor is silently ignored and the fix never
  runs** — a missing pass *binding* raises, a bad *anchor* does not.

> **This repo's gap:** `scripts/_spyre/round_trip.py`'s `make_ktir_mod` forwards
> only `grid`. Neither `data_layout` nor `required_fixes` is plumbed through, so a
> marked kernel lowered by `scripts/gen_ktir.py` gets `data_layout="device"` and
> no fixes. Plumb them through `lower.py`'s `VARIANTS` before relying on them.

## Known compiler gaps

- `tt.addptr` into `tl.make_tensor_descriptor` is not lowered by
  `LowerDescriptorMemory` (upstream `bmm_addptr` fixtures are `disabled`, pinned
  by `test_lower_desc_memory.py::TestAddptrIntoDescriptor`). Build per-batch
  descriptors from a marked N-D descriptor with an identity batch dim instead.

## Dependency

`tl.spyre_tensor_layout` exists only in the `torch-spyre/triton` build, not in
stock PyPI Triton. This repo pins the spyre-Triton rev in
`.github/workflows/ci.yaml` (`SPYRE_TRITON`), `README.md`, and
`scripts/_spyre/round_trip.py` — currently `0ddc67b8`, which provides the
builtin, the refactored `RewriteDescriptorLayout` (`Classify.cpp`,
`ContractionSynthesis.cpp`, `PermutationUtils.h`), the op verifier, multi-stick
parallel scatter, and loop rescaling. That pin is **coupled** to the `ktir-cpu`
rev in `pyproject.toml` via `ktir-mlir-frontend` — see the README note. The base
tier (`pyproject.toml` `triton>=3.7.0`, PyPI) is unchanged; markers only matter
on the spyre lowering, so they are exercised via `gen_ktir.py` + the ktir-cpu
tests, not on a GPU.

Source of truth: `torch-spyre/triton`
`third_party/spyre/lib/Dialect/KTDP/Transforms/RewriteDescriptorLayout*`, the
`test/Conversion/rewrite-descriptor-layout-*.mlir` lit tests, and
`test/fixtures/matmul/`.
