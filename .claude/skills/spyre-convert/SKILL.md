---
name: spyre-convert
description: "Converts a GPU-shaped Triton kernel (original.py) into a Spyre-aware kernel (spyre.py) that satisfies the three authoring invariants for IBM Spyre/AIU accelerators. Use when the user asks to convert a kernel to Spyre-aware form, port a kernel to Spyre, or produce a spyre.py file from an original.py."
---

# Spyre Kernel Conversion Skill

Convert a GPU-shaped Triton kernel (`original.py`) into a Spyre-aware kernel (`spyre.py`) that satisfies the three authoring invariants for IBM Spyre/AIU accelerators.

## Trigger

Use when the user asks to convert a kernel to Spyre-aware form, port a kernel to Spyre, or produce a `spyre.py` file from an `original.py`.

## Pre-flight: Consult the Spyre Knowledge Base

Before converting, query the `spyre-kb` MCP server for relevant context:

1. **Check for an existing skill** — call `mcp__spyre-kb__skill(name="<kernel_type>")` (e.g., `name="matmul"`) to see if a wiki-defined skill already covers this conversion pattern.
2. **Search for API guidance** — call `mcp__spyre-kb__search(query="<relevant topic>")` with queries like `"tensor descriptor"`, `"distribution loop"`, or the kernel's domain (e.g., `"attention"`, `"layernorm"`). Use results to inform descriptor layout, tile sizing, and distribution strategy.
3. **Read specific pages** — if search returns relevant wiki pages, call `mcp__spyre-kb__read(path="<path>")` to get full content (e.g., API reference pages, hardware constraint docs).

Use the knowledge base results to supplement (not override) the conversion rules below. If the KB provides more specific or updated guidance for a pattern, prefer it.

## Inputs

- **kernel_name**: Name of the kernel directory under `kernels/` (e.g., `matmul`, `rms_norm`)
- The source file is `kernels/<name>/original.py`
- The output file is `kernels/<name>/spyre.py`

## The Three Invariants

Every Spyre-aware kernel MUST satisfy these three invariants:

### Invariant 1 — Tiles fit scratchpad

Each Spyre core has **2MB of scratchpad**. The sum of concurrently-live tile bytes (all loaded tiles + accumulators + partial outputs alive at the same time) must be ≤ 2MB. Tile sizes are passed as `tl.constexpr` args; the kernel assumes they are valid.

**Critical**: Tile constexprs must be independent of problem dimensions. If a tile size is derived from a problem size (e.g., `BLOCK = next_power_of_2(N)` to cover an entire dimension in one load), scratchpad usage grows with the problem and the invariant does not hold. The fix is to introduce a fixed tile size and loop over the dimension in chunks.

### Invariant 2 — Grid fits 32 cores

Total cooperating program count ≤ 32. If more work exists than 32 programs, express it as an **explicit outer loop inside the kernel** (a distribution loop), not by launching more programs. Use `tl.num_programs(axis)` to query the grid size and compute per-core work bounds:

```python
pid = tl.program_id(0)
num_cores = tl.num_programs(0)
blocks_per_core = tl.cdiv(total_blocks, num_cores)
start = pid * blocks_per_core
end = tl.minimum(start + blocks_per_core, total_blocks)
for i in range(start, end):
    ...
```

### Invariant 3 — Runtime-arg agnostic

Problem-size args (`M`, `N`, `n_elements`, sequence length, batch) are runtime `i32`. Tile-size args (`BLOCK_M`, `BLOCK_SIZE`) are `tl.constexpr`. The kernel must produce correct output for **any valid combination** of runtime args given appropriately-sized constexprs — including non-divisible shapes. Use `tl.cdiv`, `tl.minimum` for tail handling. Never assume divisibility.

## Conversion Rules

### 1. Replace all pointer arithmetic with `tl.make_tensor_descriptor`

The `tl.make_block_ptr` API is deprecated. The `tl.make_tensor_descriptor` API is what Spyre targets.

**Before (raw pointer or block_ptr):**
```python
a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
```

or:
```python
p = tl.make_block_ptr(base, shape, strides, offsets, block_shape=(BM, BN), order=(1, 0))
x = tl.load(p, boundary_check=(0, 1))
p = tl.advance(p, (0, BLOCK_K))
```

**After (tensor descriptor):**
```python
a_desc = tl.make_tensor_descriptor(
    a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
)
# ...
a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
```

Key differences:
- `tl.advance` has no equivalent — just pass different offsets to each `.load()` / `.store()` call
- `boundary_check` and `padding_option` are implicit — the descriptor handles OOB
- `mask` is not used — OOB loads return zero by default
- `order` parameter does not exist — layout is determined by strides

### 2. Replace unbounded GPU grid with 32-core distribution loop

**Before (GPU-shaped):**
```python
pid = tl.program_id(axis=0)
# pid ranges over cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N) — potentially thousands
```

**After (Spyre-shaped):**
```python
pid = tl.program_id(0)
num_cores = tl.num_programs(0)
# Distribute work across exactly num_cores (≤ 32) programs
m_blocks = tl.cdiv(M, BLOCK_M)
m_blocks_per_core = tl.cdiv(m_blocks, num_cores)
m_start = pid * m_blocks_per_core
m_end = tl.minimum(m_start + m_blocks_per_core, m_blocks)
for m in range(m_start, m_end):
    ...
```

### 3. Strip `@triton.autotune`

Spyre picks constexprs explicitly — autotune is not supported. Remove the decorator entirely. Tile sizes become explicit `tl.constexpr` parameters.

### 4. Remove GPU-specific helpers

Remove `tl.assume`, `tl.multiple_of`, and any CUDA/HIP-specific config functions.

### 5. Preserve the kernel signature when possible

The spyre kernel should have the same signature as the original so the wrapper can call both with minimal branching. When the spyre kernel needs values not in the original signature (e.g., problem dimensions that the original derived from the grid), first try to recover them from existing args or grid dimensions (e.g., `tl.num_programs` on phantom axes). Only add new parameters as a last resort, and update the wrapper to handle both calling conventions.

### 6. Use descriptors for all memory accesses — no raw pointer arithmetic

Raw pointer `tl.load`/`tl.store` with `tt.addptr` are legacy TTIR operations. The Spyre backend's `LowerDescriptorMemory` pass only lowers tensor descriptor operations (`desc.load`/`desc.store`/`desc.gather`/`desc.scatter`) to KTIR — raw `tt.load`/`tt.store` have **no KTIR lowering path**.

**Goal**: Express every memory access via `tl.make_tensor_descriptor`. Use higher-rank descriptors (3D, 4D) matching the tensor's actual rank to avoid `addptr` in the base pointer. Pass strides from the kernel signature directly into the descriptor.

**Data-dependent / indirect accesses**: Use `desc.gather`/`desc.scatter` where the indirect pattern maps to those APIs (index tensor supplies one dimension's coordinate). If the pattern cannot be expressed structurally, it falls outside KTIR's expressiveness entirely — write the closest descriptor form and annotate with a gap comment.

**Known compiler gaps** (document these in the kernel header when encountered):

- **GAP: ≥16 bytes in last dimension** — `tl.make_tensor_descriptor` requires ≥16 bytes in the last dimension (a Triton frontend constraint). Scalar loads (single i32/f32) with `block_shape=[1]` or `block_shape=[..., 1]` are rejected at trace time. Write the descriptor form anyway and annotate with `# [gap] scalar descriptor — requires ≥16 bytes in last dim`.

- **GAP: Rank-reduced loads/stores** — A descriptor with leading singleton block dims (e.g. `block_shape=[1, 1, BLOCK]`) produces a result with those extra dims. Reshaping to drop them triggers `triton-combine` to fold it into a rank-reduced `tt.descriptor_load`, which `LowerDescriptorMemory` cannot handle (access tile rank ≠ result rank). Tracked: `msrivats/triton#99`. Until fixed, write the reshape explicitly and mark with `# [gap] rank-reduced load — msrivats/triton#99`.

**When gaps apply**: Write the kernel in descriptor form anyway (the target form for when gaps are resolved), and annotate each gap site with a comment. This makes it clear what will work once the compiler catches up, and avoids rewriting later.

**Utilize the scratchpad**: Each Spyre core has 2MB of scratchpad. If the distribution loop processes one work item per iteration with a small accumulator, most of that scratchpad is wasted. Batch multiple work items per iteration — widen the accumulator and load larger tiles — so that each core's scratchpad is filled with useful live data. This improves compute density and amortizes descriptor overhead.

### 6b. Batch work items to utilize scratchpad and escape descriptor gaps

After producing the initial single-item conversion, evaluate whether the distribution loop processes one work item per iteration (e.g., one row, one token, one (batch, head) pair). If so, consider batching `BLOCK_ITEMS` work items per iteration:

**When to batch:**
- The work items are **independent** — each has its own accumulator/reduction state with no cross-item dependencies.
- The scratchpad can fit `BLOCK_ITEMS` copies of the per-item live data within 2 MB.
- A scalar descriptor (`block_shape=[1]` or `block_shape=[..., 1]`) hits the 16-byte minimum gap — batching along that axis may widen the block to `[BLOCK_ITEMS]`, naturally satisfying ≥ 16 bytes.

**How to batch:**
1. Introduce a `BLOCK_ITEMS: tl.constexpr` tile parameter for the work axis.
2. Change the distribution granularity: distribute over `cdiv(total_work, BLOCK_ITEMS)` tiles instead of individual items.
3. Widen the accumulator: `[BLOCK_SIZE]` → `[BLOCK_ITEMS, BLOCK_SIZE]`.
4. Load `BLOCK_ITEMS` elements per descriptor access on the work axis.
5. If items in a tile may have divergent control flow (e.g., different sequence lengths determining which loop iterations are active), handle this with per-lane masking — compute a boolean mask per item and use `tl.where` to zero contributions from inactive lanes. The reduction math stays per-item (elementwise across the batch axis).

**Memory layout considerations:**
- If the work axis maps to a dimension with uniform stride between consecutive items (e.g., a contiguous or regularly-strided layout), the descriptor can use that stride directly with `block_shape=[BLOCK_ITEMS, ...]`.
- If consecutive work items cross a boundary in the original tensor (e.g., tiling over flattened `batch × heads` where `stride_batch = heads × stride_head`), verify that the stride between any two consecutive items is uniform. For standard contiguous layouts this holds. For non-contiguous layouts, restrict batching to within one higher-level dimension or restructure the wrapper to provide a uniformly-strided view.
- If a per-item auxiliary value (e.g., a per-batch scalar) is needed for each item in the tile, the wrapper can expand it to `[total_work_items]` via `repeat_interleave` so the kernel loads `BLOCK_ITEMS` values in one descriptor access.

**Choosing BLOCK_ITEMS:**
- Minimum: 4 (escapes the 16-byte gap for 4-byte dtypes).
- Balance scratchpad utilization against distribution granularity: `BLOCK_ITEMS` too large means fewer tiles to distribute across 32 cores (some may sit idle). Choose so that `cdiv(total_work, BLOCK_ITEMS) ≥ 32` for typical problem sizes.
- Power-of-2 values preferred for hardware efficiency.

### 6. Strides must be expressible at descriptor creation

Tensor descriptors require strides to be known when the descriptor is created. For row-major contiguous tensors, use computed strides like `[N, 1]` for a `[M, N]` tensor. If the original kernel receives strides as runtime arguments, the descriptor can still use them — pass them directly to `strides=[stride_row, stride_col]`.

### 7. Handle the activation / conditional logic

If the original kernel has optional activation functions (like `ACTIVATION: tl.constexpr`), keep that logic but apply it to the accumulator before storing via the descriptor.

### 8. Descriptor memory pattern constraints

The Spyre compiler only supports specific descriptor memory patterns. Observe these rules when converting:

#### Supported patterns

- **Static and dynamic shapes**: Both compile-time-known shapes and runtime argument shapes work. Runtime shapes produce `memref<?x...>` with runtime bounds.
  ```python
  # Static shape — OK
  desc = tl.make_tensor_descriptor(ptr, shape=[1024], strides=[1], block_shape=[BLOCK])
  # Dynamic shape — also OK (N is a runtime kernel argument)
  desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1], block_shape=[BLOCK])
  ```

- **Descriptor placement**: Descriptors can be created at function top level OR inside loops/conditionals. Placing at top level is preferred (view is constructed once and reused).
  ```python
  # Preferred: descriptor at function top, load in loop
  desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1], block_shape=[BLOCK])
  for off in range(0, N, BLOCK):
      tile = desc.load([off])
  ```

- **Gather (2D only)**: `tl.descriptor_gather` is supported for 2D descriptors with index buffers loaded via a descriptor (not passed as tensor-typed function args).
  ```python
  # OK: indices loaded via descriptor, then used in gather
  idx = idx_desc.load([offset_m])
  result = tl.descriptor_gather(data_desc, idx, y_offset)
  ```

#### Rejected patterns — do NOT use

- **`tt.addptr` result as descriptor base**: Cannot offset a pointer with arithmetic and then pass it to `tl.make_tensor_descriptor`. This means batched matmul patterns like `a_ptr + b_idx * stride_batch` as a descriptor base are NOT supported. Instead, use an extra dimension in the descriptor shape or restructure the access pattern.
  ```python
  # REJECTED — will fail to legalize
  base = a_ptr + batch_idx * stride_batch
  desc = tl.make_tensor_descriptor(base, shape=[M, K], strides=[K, 1], ...)
  ```

- **Rank-reduced loads (ND descriptor → lower-rank result)**: Loading from an ND descriptor and reshaping to drop leading singleton dims triggers `triton-combine` to fold the reshape into a rank-reduced `tt.descriptor_load`. `LowerDescriptorMemory` cannot handle the rank mismatch between the access tile and the result type. Tracked: `msrivats/triton#99`. Write the reshape explicitly and annotate as a gap.
  ```python
  # [gap] rank-reduced load — msrivats/triton#99
  desc = tl.make_tensor_descriptor(ptr, shape=[B, M, K], strides=[M*K, K, 1],
                                   block_shape=[1, BLOCK_M, BLOCK_K])
  tile = desc.load([b, m, k]).reshape([BLOCK_M, BLOCK_K])
  ```

- **N-D gather (rank > 2)**: `descriptor_gather` with a 3D block type is rejected. Paged-attention style N-D gather (block table driving an indirect dimension) is not yet supported.
  ```python
  # REJECTED — descriptor block must be a 2D tensor
  k_desc = tl.make_tensor_descriptor(k_cache_ptr, [...], block_shape=[1, block_size, dim])
  result = tl.descriptor_gather(k_desc, block_table, head_offset)
  ```

- **Gather with tensor-typed function argument as `x_offsets`**: The `x_offsets` for `descriptor_gather` must come from a `descriptor_load` (i.e., loaded from a `!tt.ptr<i32>` buffer), not passed as a tensor-typed kernel argument.
  ```python
  # REJECTED — x_offsets must come from a descriptor load, not a kernel arg
  @triton.jit
  def kernel(x_offsets, ...):  # x_offsets as tensor arg — NOT allowed
      result = tl.descriptor_gather(desc, x_offsets, y_offset)
  ```

## Output File Structure

```python
# SPDX-License-Identifier: <same as original>
# SPDX-FileCopyrightText: <same as original>
#
# Spyre-aware conversion of <kernel_function_name>.
# Original: <source path>
#
# Conversion from original:
#   - All pointer arithmetic replaced with tl.make_tensor_descriptor
#   - Grid capped at 32 cores with explicit distribution loop
#   - <other specific changes>

import triton
import triton.language as tl


@triton.jit
def <kernel_name>_spyre(
    # Pointers
    ...,
    # Problem dimensions (runtime i32)
    ...,
    # Tile sizes (constexpr)
    BLOCK_...: tl.constexpr,
):
    """<Brief description of what the kernel computes.>"""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # Create tensor descriptors
    ...

    # Distribution loop
    ...
```

## Worked Example: Matmul

**Input** (`kernels/matmul/original.py`): GPU-shaped matmul with autotune, pointer arithmetic, unbounded grid.

**Output** (`kernels/matmul/spyre.py`):

```python
@triton.jit
def matmul_kernel_spyre(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores)
    m_start = pid * m_blocks_per_core
    m_end = tl.minimum(m_start + m_blocks_per_core, m_blocks)

    for m in range(m_start, m_end):
        for n in range(n_blocks):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(k_tiles):
                a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
                b_tile = b_desc.load([k * BLOCK_K, n * BLOCK_N])
                acc = tl.dot(a_tile, b_tile, acc)
            c_desc.store([m * BLOCK_M, n * BLOCK_N], acc)
```

Key decisions in this conversion:
- M is the distributed axis (rows of C partitioned across cores)
- N and K are inner loops (every core does all N-blocks and all K-tiles)
- Strides assume row-major contiguous layout (`[K, 1]` for A, `[N, 1]` for B)
- No activation logic (stripped for simplicity; add back if needed)
- Accumulator is f32 (standard practice for numerical stability)

## Checklist Before Finishing

1. [ ] ALL memory accesses use descriptors — no raw `tl.load`/`tl.store` anywhere
2. [ ] Gap sites annotated: scalar descriptors with `# [gap] scalar descriptor`, rank-reduced with `# [gap] rank-reduced load/store — msrivats/triton#99`
3. [ ] `@triton.autotune` removed
4. [ ] Grid ≤ 32 with distribution loop using `tl.num_programs()`
5. [ ] All problem sizes are runtime args (no `tl.constexpr` on M, N, K, etc.)
6. [ ] All tile sizes are `tl.constexpr`
7. [ ] Non-divisible shapes handled via `tl.cdiv` + `tl.minimum`
8. [ ] No `tl.make_block_ptr`, no `tl.advance`
9. [ ] No `tl.assume`, no `tl.multiple_of`
10. [ ] No pointer arithmetic (`addptr`) feeding into descriptor base
11. [ ] Descriptors may be any rank (1D–4D); rank-reduced loads/stores are annotated as gap (msrivats/triton#99)
12. [ ] Gather indices (if used) come from descriptor loads, not kernel args
13. [ ] Kernel produces correct output for the same inputs as original
14. [ ] File placed at `kernels/<name>/spyre.py`

## References

- Spyre fixtures (canonical examples): `msrivats/triton` repo, `third_party/spyre/test/fixtures/`
- Memory patterns reference: `msrivats/triton` repo, `third_party/spyre/docs/patterns/memory.md`
- Issue #13: https://github.ibm.com/Ohad-Eytan1/tritokti/issues/13
- Flim's roadmap gist: https://gist.github.ibm.com/flim/6bc5edf67ed8b509d3e51abeb77a08d0
- Call for Spyre kernels: https://github.ibm.com/msrivats/triton/issues/79
