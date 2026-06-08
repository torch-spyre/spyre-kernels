import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.rms_norm.original import _rms_norm_kernel
from kernels.rms_norm.tensor_descriptor import _rms_norm_kernel_td

# Upper bound on the number of programs the row-batched kernel launches.
# Each program processes a contiguous block of rows; when n_rows exceeds
# this, rows are batched so the grid stays bounded.
MAX_PROGRAMS = 32


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
    kernel_fn=_rms_norm_kernel,
) -> torch.Tensor:
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024

    if kernel_fn is _rms_norm_kernel_td:
        # Row-batched tensor-descriptor kernel: the grid is decoupled from
        # n_rows, so cap it and let each program process a block of rows.
        ensure_triton_allocator()
        grid = (min(n_rows, MAX_PROGRAMS),)
        kernel_fn[grid](
            input_2d,
            weight,
            output,
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        grid = (n_rows,)
        kernel_fn[grid](
            input_2d,
            weight,
            output,
            input_2d.stride(0),
            output.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output.reshape(original_shape)
