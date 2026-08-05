import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.vllm.embedding.original import embedding_forward_kernel


def embedding(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    kernel_fn=embedding_forward_kernel,
) -> torch.Tensor:
    """
    Gather rows from an embedding table.

    Args:
        embeddings: [vocab_size, embedding_dim]
        indices: integer tensor of arbitrary shape; each value selects a row
            from `embeddings`.

    Returns:
        Tensor of shape `indices.shape + (embedding_dim,)`.
    """
    assert embeddings.dim() == 2
    embeddings = embeddings.contiguous()
    indices = indices.contiguous()

    ori_shape = indices.shape
    indices_flat = indices.view(-1)
    n_elements = indices_flat.numel()
    embedding_dim = embeddings.shape[1]

    output = torch.empty(
        n_elements, embedding_dim,
        device=indices.device, dtype=embeddings.dtype,
    )

    ensure_triton_allocator()

    BLOCK_SIZE_M = triton.next_power_of_2(min(128, embedding_dim))
    BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
    grid = (
        triton.cdiv(n_elements, BLOCK_SIZE_M),
        triton.cdiv(embedding_dim, BLOCK_SIZE_N),
    )

    kernel_fn[grid](
        embeddings,
        indices_flat,
        output,
        n_elements,
        embedding_dim=embedding_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output.view(*ori_shape, embedding_dim)
