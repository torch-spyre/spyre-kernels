import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.rms_norm.original import _rms_norm_kernel
from kernels.rms_norm.tensor_descriptor import _rms_norm_kernel_td

# Number of rows each program of the tensor-descriptor kernel processes.
# 1 recovers one program per row. Must be a power of 2: it is the row
# dimension of the descriptor block_shape, which tl.make_tensor_descriptor
# requires to be a power of 2.
ROWS_PER_PROGRAM = 1


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
        # Row-batched tensor-descriptor kernel: each program handles
        # ROWS_PER_PROGRAM rows, so the grid scales with n_rows.
        assert ROWS_PER_PROGRAM.bit_count() == 1, (
            "ROWS_PER_PROGRAM must be a power of 2 (descriptor block_shape "
            f"requirement), got {ROWS_PER_PROGRAM}"
        )
        ensure_triton_allocator()
        grid = (triton.cdiv(n_rows, ROWS_PER_PROGRAM),)
        kernel_fn[grid](
            input_2d,
            weight,
            output,
            n_rows,
            n_cols,
            input_2d.stride(0),
            output.stride(0),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
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
