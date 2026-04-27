import torch
import triton

from kernels.rms_norm.original import _rms_norm_kernel


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
