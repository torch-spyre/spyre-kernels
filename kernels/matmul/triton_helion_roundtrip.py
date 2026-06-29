from __future__ import annotations
import torch
import helion
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher
helion.runtime.set_triton_allocator()

@triton.jit
def _helion_matmul_helion(a, b, out, a_size_0, a_size_1, b_size_0, b_size_1, a_stride_0, b_stride_0, out_stride_0, out_stride_1, m, n, k, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a, [a_size_0, a_size_1], [a_stride_0, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b, [b_size_0, b_size_1], [b_stride_0, 1], [BLOCK_K, BLOCK_N])
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    inner_2d_pid = tl.program_id(0)
    num_pid_in_group = 8 * num_pid_n
    group_id = inner_2d_pid // num_pid_in_group
    first_pid_m = group_id * 8
    group_size_m = min(num_pid_m - first_pid_m, 8)
    pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
    pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
    offset_0 = pid_0 * BLOCK_M
    indices_0 = (offset_0 + tl.arange(0, BLOCK_M)).to(tl.int32)
    mask_0 = indices_0 < m
    offset_1 = pid_1 * BLOCK_N
    indices_1 = (offset_1 + tl.arange(0, BLOCK_N)).to(tl.int32)
    mask_1 = indices_1 < n
    acc = tl.full([BLOCK_M, BLOCK_N], 0, tl.float32)
    for offset_2 in tl.range(0, tl.cast(k, tl.int32), BLOCK_K, flatten=True):
        acc_copy = acc
        acc_copy_0 = acc_copy
        load = a_desc.load([offset_0, offset_2])
        load_1 = b_desc.load([offset_2, offset_1])
        acc = tl.dot(tl.cast(load, tl.float16), tl.cast(load_1, tl.float16), acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
    v_0 = tl.cast(acc, tl.float16)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_0, mask_0[:, None] & mask_1[None, :])

def matmul_helion(a: torch.Tensor, b: torch.Tensor, *, _launcher=_default_launcher):
    BLOCK_K = 64
    m, k = a.shape
    k2, n = b.shape
    out = torch.empty(m, n, dtype=a.dtype, device=a.device)
    BLOCK_M = 64
    BLOCK_N = 128
    _launcher(_helion_matmul_helion, ((m + BLOCK_M - 1) // BLOCK_M * ((n + BLOCK_N - 1) // BLOCK_N),), a, b, out, a.size(0), a.size(1), b.size(0), b.size(1), a.stride(0), b.stride(0), out.stride(0), out.stride(1), m, n, k, num_warps=4, num_stages=6, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out