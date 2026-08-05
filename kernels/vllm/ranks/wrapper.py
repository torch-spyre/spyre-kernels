import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.vllm.ranks.original import _ranks_kernel


def ranks(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    kernel_fn=_ranks_kernel,
) -> torch.Tensor:
    assert logits.dim() == 2
    assert token_ids.dim() == 1
    assert logits.shape[0] == token_ids.shape[0]
    logits = logits.contiguous()
    token_ids = token_ids.contiguous()

    num_requests, vocab_size = logits.shape
    output = torch.empty(num_requests, device=logits.device, dtype=torch.int32)

    BLOCK_SIZE = 1024
    grid = (num_requests,)
    if "num_requests" in kernel_fn.arg_names:
        ensure_triton_allocator()
        kernel_fn[grid](
            output,
            logits,
            logits.stride(0),
            token_ids,
            num_requests,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        kernel_fn[grid](
            output,
            logits,
            logits.stride(0),
            token_ids,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output
