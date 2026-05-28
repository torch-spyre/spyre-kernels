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

### 5. Keep scalar/data-dependent loads as raw pointers

If a load is:
- A single scalar element (e.g., `tl.load(token_ids_ptr + req_idx)`)
- Data-dependent / indirect (the offset comes from a runtime-loaded index)

Then it **stays as a raw pointer load** — tensor descriptors cannot express indirect access (use `desc.gather`/`desc.scatter` only if the indirect pattern maps cleanly to those APIs).

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

- **Rank-reduced loads (3D descriptor → 2D result)**: Cannot fold `tl.reshape(desc.load(...))` where the descriptor is 3D and the result is 2D. Descriptors must be 2D with 2D block shapes.
  ```python
  # REJECTED — 3D block shape not supported
  desc = tl.make_tensor_descriptor(ptr, shape=[B, M, K], strides=[M*K, K, 1],
                                   block_shape=[1, BLOCK_M, BLOCK_K])
  tile = tl.reshape(desc.load([b, m, k]), [BLOCK_M, BLOCK_K])
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

1. [ ] All `tl.load`/`tl.store` with pointer arithmetic → descriptor `.load()`/`.store()`
2. [ ] Exception: scalar/indirect loads stay raw
3. [ ] `@triton.autotune` removed
4. [ ] Grid ≤ 32 with distribution loop using `tl.num_programs()`
5. [ ] All problem sizes are runtime args (no `tl.constexpr` on M, N, K, etc.)
6. [ ] All tile sizes are `tl.constexpr`
7. [ ] Non-divisible shapes handled via `tl.cdiv` + `tl.minimum`
8. [ ] No `tl.make_block_ptr`, no `tl.advance`
9. [ ] No `tl.assume`, no `tl.multiple_of`
10. [ ] No pointer arithmetic (`addptr`) feeding into descriptor base
11. [ ] All descriptors are 2D (no 3D block shapes or rank-reduced loads)
12. [ ] Gather indices (if used) come from descriptor loads, not kernel args
13. [ ] Kernel produces correct output for the same inputs as original
14. [ ] File placed at `kernels/<name>/spyre.py`

## References

- Spyre fixtures (canonical examples): `msrivats/triton` repo, `third_party/spyre/test/fixtures/`
- Memory patterns reference: `msrivats/triton` repo, `third_party/spyre/docs/patterns/memory.md`
- Issue #13: https://github.ibm.com/Ohad-Eytan1/tritokti/issues/13
- Flim's roadmap gist: https://gist.github.ibm.com/flim/6bc5edf67ed8b509d3e51abeb77a08d0
- Call for Spyre kernels: https://github.ibm.com/msrivats/triton/issues/79
