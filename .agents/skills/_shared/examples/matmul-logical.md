# Worked example — matmul (logical shape)

> **Logical shape + layout markers.** You write descriptors in the tensor's math
> shape and attach a `tl.spyre_tensor_layout` marker declaring the physical stick
> layout. The compiler's `RewriteDescriptorLayout` pass synthesizes the physical
> loops — you do not hand-write shape/strides for it, and `tl.dot` is unchanged.
> This file is the template you copy.

**Input** (`kernels/vllm/matmul/original.py`): GPU-shaped matmul with autotune,
pointer arithmetic, unbounded grid.

**Output** (`kernels/vllm/matmul/spyre.py`):

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
    # Physical stick layout via markers. Descriptors stay logical; tl.dot is
    # unchanged; no reshape glue. Convention (stick-on-X -> phys [X//S, other, X%S]):
    #   marker = [(X_logical, "floordiv", S), other_logical, (X_logical, "mod", S)]
    # S = 128 // dtype_bytes (64 for fp16/bf16); src is the *logical* dim index.
    # Every stickified extent must be a multiple of S — M and N here.
    tl.spyre_tensor_layout(a_desc, [(0, "floordiv", 64), 1, (0, "mod", 64)])  # A[M,K] stick-on-M
    tl.spyre_tensor_layout(b_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])  # B[K,N] stick-on-N
    tl.spyre_tensor_layout(c_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])  # C[M,N] stick-on-N (drives store sink)

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

Key decisions:
- **M** is the distributed axis (rows of C partitioned across cores).
- **N** and **K** are inner loops (every core does all N-blocks, all K-tiles).
- Strides assume row-major contiguous layout (`[K, 1]` for A, `[N, 1]` for B).
- Accumulator is f32 (numerical stability for the K reduction).
- **Layout markers** declare the stick tiling: `stick-on-X -> [(X,"floordiv",S),
  other, (X,"mod",S)]`, `S = 128//dtype_bytes`, inline literal (or a `constexpr`
  arg). Which loop structure results is a *consequence* of the axes you mark —
  stick-tiling a parallel axis gives a scatter loop over its sticks, stick-tiling
  the contraction axis gives a K-reduction loop. Marking the output descriptor is
  what drives the store sink.
- The `m`/`n` loops above are in **block units**; if a loop IV feeds a marked
  descriptor's stick dim the pass rescales its bounds *and step* to stick units
  for you — do not pre-multiply. See
  [`../spyre/tensor-layout-marker.md`](../spyre/tensor-layout-marker.md).
- No activation logic (stripped for brevity; add back if the original has it).
