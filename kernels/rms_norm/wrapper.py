import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.rms_norm.original import _rms_norm_kernel
from kernels.rms_norm.tensor_descriptor import _rms_norm_kernel_td

# Default number of rows each tensor-descriptor program processes. 1 recovers
# one program per row. Must be a power of 2: it is the row dimension of the
# descriptor block_shape, which tl.make_tensor_descriptor requires to be a
# power of 2.
ROWS_PER_PROGRAM = 1
BLOCK_SIZE = 1024


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
    kernel_fn=_rms_norm_kernel,
    rows_per_program: int = ROWS_PER_PROGRAM,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Launch the RMS-norm kernel selected by `kernel_fn`.

    `rows_per_program` and `block_size` tune the tensor-descriptor kernel's
    tiling (ignored by the original kernel, which has no row batching). They are
    exposed so tests can sweep them through this one launch path rather than
    re-implementing the launch.

    The input is reshaped to 2D but **not** forced contiguous — its real row
    stride is passed to the kernel, so a strided (e.g. column-sliced) input is
    handled correctly.
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)

    if kernel_fn is _rms_norm_kernel_td:
        # Row-batched tensor-descriptor kernel: each program handles
        # rows_per_program rows, so the grid scales with n_rows.
        assert rows_per_program.bit_count() == 1, (
            "rows_per_program must be a power of 2 (descriptor block_shape "
            f"requirement), got {rows_per_program}"
        )
        ensure_triton_allocator()
        grid = (triton.cdiv(n_rows, rows_per_program),)
        kernel_fn[grid](
            input_2d,
            weight,
            output,
            n_rows,
            n_cols,
            input_2d.stride(0),
            output.stride(0),
            eps,
            BLOCK_SIZE=block_size,
            ROWS_PER_PROGRAM=rows_per_program,
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
            BLOCK_SIZE=block_size,
        )
    return output.reshape(original_shape)
