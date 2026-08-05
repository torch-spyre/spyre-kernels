import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.vllm.silu_and_mul.original import _swiglustep_and_mul_kernel


def silu_and_mul(
    x: torch.Tensor, limit: float = 7.0,
    kernel_fn=_swiglustep_and_mul_kernel,
) -> torch.Tensor:
    original_shape = x.shape
    assert original_shape[-1] % 2 == 0, "Last dimension must be even"
    d = original_shape[-1] // 2

    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    n_rows = x_2d.shape[0]

    output = torch.empty(n_rows, d, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = (n_rows, triton.cdiv(d, BLOCK_SIZE))

    if "n_rows" in kernel_fn.arg_names:
        ensure_triton_allocator()
        kernel_fn[grid](
            output,
            output.stride(0),
            x_2d,
            x_2d.stride(0),
            n_rows,
            limit=limit,
            d=d,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        kernel_fn[grid](
            output,
            output.stride(0),
            x_2d,
            x_2d.stride(0),
            limit=limit,
            d=d,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output.reshape(*original_shape[:-1], d)
