---
name: spyre-review
description: "Reviews a Spyre-aware Triton kernel for compliance with the three authoring invariants and correct use of the tl.make_tensor_descriptor API. Use when the user asks to review, verify, validate, or check a Spyre kernel for compliance, or when a spyre.py file has just been created and needs verification."
---

# Spyre Kernel Review Skill

Review a Spyre-aware Triton kernel for compliance with the three authoring invariants and correct use of the `tl.make_tensor_descriptor` API.

## Trigger

Use when the user asks to review, verify, validate, or check a Spyre kernel for compliance — or when a `spyre.py` file has just been created and needs verification.

## Pre-flight: Consult the Spyre Knowledge Base

Before reviewing, query the `spyre-kb` MCP server for relevant context:

1. **Search for known constraints** — call `mcp__spyre-kb__search(query="<kernel domain>")` (e.g., `"attention"`, `"softmax"`, `"tensor descriptor limitations"`) to check for known pitfalls or hardware-specific notes relevant to this kernel type.
2. **Check for updated API docs** — call `mcp__spyre-kb__search(query="make_tensor_descriptor")` or `mcp__spyre-kb__search(query="distribution loop")` to verify that the review criteria align with the latest documented behavior.
3. **Read relevant pages** — if search surfaces pages with updated constraints or newly documented edge cases, call `mcp__spyre-kb__read(path="<path>")` to get full details.

If the knowledge base provides updated or more specific constraints than the review procedure below, flag discrepancies in the review report.

## Inputs

- **kernel_name**: Name of the kernel directory under `kernels/` (e.g., `matmul`, `rms_norm`)
- The file to review is `kernels/<name>/spyre.py`
- Compare against `kernels/<name>/original.py` for correctness

## Review Procedure

### Step 1: Check Invariant 1 — Tiles fit scratchpad (≤ 2MB)

For each descriptor in the kernel, identify `block_shape`. Compute the total concurrently-live tile bytes at the most memory-intensive program point:

```
tile_bytes = product(block_shape) * dtype_bytes
```

Sum all tiles that are alive simultaneously (loaded tiles in the same loop body + accumulator). Report the total and whether it's under 2MB.

**Common dtype sizes:** f16 = 2 bytes, f32 = 4 bytes, bf16 = 2 bytes, i32 = 4 bytes.

**Example (matmul inner loop):**
- A tile: BLOCK_M × BLOCK_K × 2 bytes (f16)
- B tile: BLOCK_K × BLOCK_N × 2 bytes (f16)
- Accumulator: BLOCK_M × BLOCK_N × 4 bytes (f32)
- Total = BM×BK×2 + BK×BN×2 + BM×BN×4

Flag if the formula could exceed 2MB for plausible constexpr values (e.g., BLOCK_M=128, BLOCK_N=128, BLOCK_K=64).

**Critical check**: Verify that tile constexprs are independent of problem dimensions. If a tile size is derived from a problem size (e.g., `BLOCK = next_power_of_2(N)` set in the wrapper to cover an entire dimension without looping), scratchpad usage grows with the problem and the invariant does NOT hold — even if the byte count is small for typical values. Flag this as FAIL and recommend introducing a fixed tile size with a loop over the dimension.

### Step 1b: Scratchpad utilization — detect under-use

After verifying the hard constraint (tiles fit), assess whether the kernel is **wasting scratchpad capacity** by processing less data per iteration than it could. Low utilization means the kernel is leaving performance on the table — the scratchpad exists to hold working data, and using 1 KB of 2 MB means the hardware's parallelism is underexploited.

**Procedure:**

1. Compute the peak live bytes (from Step 1). Compare against the 2 MB budget. If utilization is below ~10% (i.e., < 200 KB), flag for investigation.

2. Identify which loop axis processes **one element at a time** when it could batch multiple elements per iteration. The telltale pattern:
   ```python
   for work_idx in range(start, end):
       # ... processes exactly ONE item (one batch, one head, one token) per iteration
       acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # 1D accumulator
   ```

3. Ask: **could the kernel vectorize across that axis?** The answer is YES when:
   - The reduction (inner loop) is **independent** across items on that axis — i.e., each item has its own accumulator, max, sum, etc., with no cross-item dependency.
   - The scratchpad can fit `BLOCK_ITEMS × BLOCK_SIZE` tiles without exceeding 2 MB.
   - Example: a per-element reduction (softmax, norm, etc.) where each work item's state is fully independent of others on the batched axis.

4. If batching is possible, compute a recommended `BLOCK_ITEMS` value:
   ```
   available = 2 MB (or conservative 1 MB)
   per_item_bytes = (acc_tile + loaded_tiles) * dtype_bytes
   BLOCK_ITEMS = floor(available / per_item_bytes)
   ```
   Clamp to a power of 2 and a reasonable maximum (e.g., 16–64).

5. Note side benefits of batching:
   - Amortizes loop overhead and descriptor setup across multiple items
   - Increases compute density on inner multiply-add operations
   - May escape the 16-byte descriptor minimum: if a scalar load becomes a `[BLOCK_ITEMS]` vector load, `BLOCK_ITEMS × dtype_bytes ≥ 16` satisfies the constraint naturally (e.g., `BLOCK_ITEMS=4` × 4 bytes = 16 bytes)

6. **Cross-reference with GAP 1 findings**: If the kernel has scalar descriptors flagged in Step 4/5 (block_shape with < 16 bytes in the last dim), check whether batching along the work axis would widen that dimension to ≥ 16 bytes. If so, report that batching is not just a performance improvement but also a **gap resolution path** — it eliminates the need for the scalar descriptor entirely.

7. **Divergent control flow across batched items**: If items in a tile have different runtime parameters (e.g., different lengths that control which loop iterations are active), batching is still possible with per-lane masking. The pattern:
   - Load a vector of per-item parameters (one per lane in the batch axis)
   - Compute per-lane boolean masks for conditional logic
   - Use `tl.where` to zero contributions from inactive lanes
   - The reduction stays per-item (elementwise across the batch axis)
   
   Report this as feasible-with-masking rather than infeasible.

8. **Distribution granularity trade-off**: Batching reduces the number of distributable tiles. If `cdiv(total_work, BLOCK_ITEMS) < 32`, some cores will be idle. Report the trade-off: suggest a `BLOCK_ITEMS` value where `total_work / BLOCK_ITEMS ≥ 32` for typical problem sizes, or note that the kernel will under-utilize cores for small problems.

**Report this as WARN (not FAIL)** — scratchpad under-use is a performance issue, not a correctness/compliance issue. Include specific recommendations for which axis to batch and what `BLOCK_ITEMS` value to use.

**Caveat — when batching across an axis is NOT possible:**
- The items on that axis have **different control flow** (e.g., different sequence lengths determine which iterations are active) — batching requires either uniform control flow or masking
- The axis involves **cross-item dependencies** (e.g., a reduction that accumulates across the batched dimension)
- Layout constraints make multi-item loads non-contiguous in memory

In such cases, note the limitation in the report but still flag the under-use as a known performance gap.

### Step 2: Check Invariant 2 — Grid fits 32 cores

Verify:
1. The kernel uses `tl.program_id(axis)` and `tl.num_programs(axis)` to query the grid
2. There is an explicit distribution loop that partitions work across cores
3. The distribution pattern follows:
   ```python
   blocks_per_core = tl.cdiv(total_blocks, num_cores)
   start = pid * blocks_per_core
   end = tl.minimum(start + blocks_per_core, total_blocks)
   for i in range(start, end):
       ...
   ```
4. No assumption that `num_programs` equals the total number of tiles/blocks

**Red flags:**
- `pid` used directly as a tile index without a distribution loop
- Grid size computed from problem dimensions (e.g., `grid = cdiv(M, BLOCK_M)`)
- Missing `tl.num_programs()` call

### Step 3: Check Invariant 3 — Runtime-arg agnostic

Verify:
1. Problem sizes (`M`, `N`, `K`, `n_elements`, `seq_len`, `batch`, etc.) are **not** annotated with `tl.constexpr`
2. Tile sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `BLOCK_SIZE`) **are** annotated with `tl.constexpr`
3. Tail handling exists: `tl.cdiv` for block counts, `tl.minimum` for loop bounds
4. No integer division or modulo that assumes divisibility without guarding

**Red flags:**
- `M: tl.constexpr` (problem size as constexpr)
- `range(0, M // BLOCK_M)` without `tl.cdiv`
- Missing `tl.minimum` on distribution loop end bound

### Step 4: Verify `tl.make_tensor_descriptor` usage

Check that:
1. **No `tl.make_block_ptr`** appears anywhere — it's deprecated
2. **No raw pointer arithmetic** for regular (non-scalar, non-indirect) loads/stores
3. Every descriptor has valid `shape`, `strides`, and `block_shape` arguments
4. Descriptor `shape` uses runtime args (not constexprs) for problem dimensions
5. Descriptor `block_shape` uses constexprs for tile dimensions
6. Loads/stores use offset expressions that are multiples of block dimensions: `[i * BLOCK_M, j * BLOCK_N]`
7. No `tl.advance` calls (block_ptr pattern, not descriptor pattern)
8. **No redundant tail masking** layered on descriptor loads/stores (see below)

**No raw pointer usage is acceptable.** All accesses must use descriptors. When a descriptor hits a known compiler gap (≥16-byte last-dim minimum, rank-reduced loads), write the descriptor form anyway and annotate with the relevant gap comment.

**Redundant tail masking is dead weight.** The descriptor `shape=[...]` already encodes the tensor boundary: every `.load()` zero-fills out-of-range lanes and every `.store()` clamps writes at the boundary. A converted kernel that *also* carries the original's `mask`/`tl.where`/`other=` machinery is re-zeroing lanes that are already zero. This is a common conversion artifact — the author ports the raw-pointer mask over without realizing the descriptor subsumes it. Flag it as a **WARN** (cleanup, not a correctness failure): the manual mask is harmless but adds clutter and forces needless `tl.arange`/`[None, :]` broadcasts inside the loop (worse on multi-dim batched tiles). Recommend deleting it.

```python
# WARN — redundant: descriptor load already zero-filled lanes >= N
x = in_desc.load([tile * BLOCK])
offs = tile * BLOCK + tl.arange(0, BLOCK)
acc += tl.sum(tl.where(offs < N, x, 0.0))    # mask re-zeroes already-zero lanes

# CLEAN — let the descriptor's shape carry the boundary
x = in_desc.load([tile * BLOCK])
acc += tl.sum(x)                             # zero lanes are the sum identity
```

**Exception — non-identity fills.** Zero-fill is the reduction identity only for additive reductions (sum). If the tail must carry a *non-zero* identity — `1.0` for a product, `-inf` for a max — the descriptor's zero-fill is wrong and an explicit tail fix-up IS required. Do not flag masking in that case; instead verify the fix-up uses the correct identity. (A converted kernel that drops the mask *and* needs a non-zero identity is a correctness FAIL — check this against the original's `other=` value.)

### Step 5: Check descriptor memory pattern constraints

Verify the kernel does not use any rejected Spyre compiler patterns:

1. **No `addptr` as descriptor base**: The base pointer passed to `tl.make_tensor_descriptor` must be a raw pointer argument — NOT the result of pointer arithmetic (e.g., `ptr + offset * stride`). This rejects batched patterns that offset the base per batch.
   ```python
   # FAIL — addptr result as descriptor base
   base = a_ptr + batch_idx * stride_batch
   desc = tl.make_tensor_descriptor(base, ...)
   ```

2. **Rank-reduced loads/stores are a known gap**: Descriptors may be any rank (1D–4D), but loading from an ND descriptor and reshaping to drop leading singleton dims triggers a rank-reduced `tt.descriptor_load` that `LowerDescriptorMemory` cannot handle. Tracked: `msrivats/triton#99`. The kernel should write the reshape explicitly and annotate with `# [gap] rank-reduced load — msrivats/triton#99`. This is NOT a compliance failure — it is the target form pending compiler support.
   ```python
   # [gap] rank-reduced load — msrivats/triton#99
   desc = tl.make_tensor_descriptor(ptr, shape=[B, M, K], block_shape=[1, BM, BK])
   tile = desc.load([b, m, k]).reshape([BM, BK])
   ```

3. **Gather requires 2D descriptors**: `tl.descriptor_gather` only works with 2D block types. N-D gather (e.g., paged-attention style with 3D blocks) is rejected.

4. **Gather `x_offsets` must come from a descriptor load**: If the kernel uses `tl.descriptor_gather`, the index tensor (`x_offsets`) must be loaded from memory via another descriptor — it cannot be a tensor-typed kernel argument.
   ```python
   # FAIL — x_offsets is a kernel argument, not loaded via descriptor
   def kernel(x_offsets, ...):
       result = tl.descriptor_gather(desc, x_offsets, y_offset)
   
   # PASS — x_offsets loaded from a pointer via descriptor
   idx_desc = tl.make_tensor_descriptor(idx_ptr, ...)
   x_offsets = idx_desc.load([offset])
   result = tl.descriptor_gather(desc, x_offsets, y_offset)
   ```

5. **Descriptor placement is flexible**: Descriptors at function top level (preferred), inside loops, or inside conditionals are all valid. Note: top-level placement constructs the view once and reuses it, which is more efficient.

**Red flags:**
- Any arithmetic on a pointer before it becomes a descriptor base
- Any raw `tl.load`/`tl.store` without a gap annotation
- `tl.descriptor_gather` with a kernel argument as indices
- `tl.reshape` applied to a descriptor load result without a gap annotation

### Step 6: Correctness check against original

Compare the converted kernel against `original.py`:
1. Same mathematical operation is computed
2. Same accumulator dtype (usually f32 for reductions)
3. Same output dtype conversion (e.g., `.to(tl.float16)`)
4. Same reduction structure (loops in the right order)
5. No accidentally dropped operations (activations, clamping, normalization steps)
6. All input/output tensors from the original are handled

### Step 7: Check for removed patterns

Verify these GPU-specific patterns are absent:
- [ ] No `@triton.autotune` decorator
- [ ] No `tl.assume()` calls
- [ ] No `tl.multiple_of()` calls
- [ ] No CUDA/HIP config functions
- [ ] No `order=(1, 0)` parameter (block_ptr pattern)
- [ ] No `boundary_check` parameter (block_ptr pattern)
- [ ] No `padding_option` parameter (block_ptr pattern)

## Report Format

```
## Spyre Compliance Review: <kernel_name>

### Invariant 1 — Tiles fit scratchpad
**Status:** PASS / WARN / FAIL
- Concurrently-live tiles: <list with sizes>
- Total at peak: <formula> = <value for typical constexprs>
- Verdict: <explanation>

### Scratchpad utilization
**Status:** OK / WARN
- Peak live bytes: <value>
- Utilization: <peak / 2MB as percentage>
- Batchable axis: <axis name or "none">
- Recommended BLOCK_ITEMS: <value or N/A>
- Side benefits: <e.g., escapes 16-byte minimum>
- Caveats: <e.g., divergent control flow requires masking>

### Invariant 2 — Grid fits 32 cores
**Status:** PASS / FAIL
- Distribution pattern: <description>
- Distribution axis: <which dimension is distributed>
- <any issues>

### Invariant 3 — Runtime-arg agnostic
**Status:** PASS / FAIL
- Runtime args: <list>
- Constexpr args: <list>
- Tail handling: <present/missing where>

### Descriptor API usage
**Status:** PASS / WARN / FAIL
- Descriptors: <count> created
- Raw pointer loads: <count> (justified: <yes/no for each>)
- Redundant tail masking: <none / present at <location> — recommend removing>
- Issues: <any>

### Descriptor memory patterns
**Status:** PASS / FAIL
- Addptr as base: <present/absent>
- Descriptor rank: <all 2D / violations found>
- Gather usage: <correct / violations>
- Issues: <any>

### Correctness vs original
**Status:** PASS / FAIL
- <any discrepancies>

### Overall: COMPLIANT / NON-COMPLIANT
<summary of issues to fix, if any>
```

## References

- Three invariants defined in: https://gist.github.ibm.com/flim/6bc5edf67ed8b509d3e51abeb77a08d0
- Canonical Spyre fixtures: `msrivats/triton` repo, `third_party/spyre/test/fixtures/`
- Memory patterns reference: `msrivats/triton` repo, `third_party/spyre/docs/patterns/memory.md`
- Conversion skill: `skills/spyre-convert.md`
