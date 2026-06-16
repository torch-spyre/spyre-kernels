# Worked example — matmul (logical shape)

> **Logical shape.** This is the form you write: descriptors in the tensor's
> math shape. The physical device layout is derived by the compiler — you do
> not hand-write it. This file is the template you copy.

**Input** (`kernels/matmul/original.py`): GPU-shaped matmul with autotune,
pointer arithmetic, unbounded grid.

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

Key decisions:
- **M** is the distributed axis (rows of C partitioned across cores).
- **N** and **K** are inner loops (every core does all N-blocks, all K-tiles).
- Strides assume row-major contiguous layout (`[K, 1]` for A, `[N, 1]` for B).
- Accumulator is f32 (numerical stability for the K reduction).
- No activation logic (stripped for brevity; add back if the original has it).
