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

**Acceptable raw pointer usage:**
- Single scalar loads: `tl.load(ptr + idx)` for one element
- Data-dependent/indirect loads: offset comes from a loaded index value
- Scalar stores: `tl.store(ptr + idx, scalar_value)`

### Step 5: Correctness check against original

Compare the converted kernel against `original.py`:
1. Same mathematical operation is computed
2. Same accumulator dtype (usually f32 for reductions)
3. Same output dtype conversion (e.g., `.to(tl.float16)`)
4. Same reduction structure (loops in the right order)
5. No accidentally dropped operations (activations, clamping, normalization steps)
6. All input/output tensors from the original are handled

### Step 6: Check for removed patterns

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
**Status:** PASS / FAIL
- Descriptors: <count> created
- Raw pointer loads: <count> (justified: <yes/no for each>)
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
- Conversion skill: `skills/spyre-convert.md`
