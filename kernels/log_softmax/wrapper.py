import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.log_softmax.original import _topk_log_softmax_kernel


def topk_log_softmax(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    kernel_fn=_topk_log_softmax_kernel,
) -> torch.Tensor:
    assert logits.dim() == 2
    assert topk_ids.dim() == 2
    assert topk_ids.shape[1] == topk
    logits = logits.contiguous()
    topk_ids = topk_ids.contiguous()

    num_requests, vocab_size = logits.shape

    output = torch.empty(num_requests, topk, device=logits.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    PADDED_TOPK = triton.next_power_of_2(topk)
    grid = (num_requests,)
    if "num_requests" in kernel_fn.arg_names:
        ensure_triton_allocator()
        kernel_fn[grid](
            output,
            logits,
            logits.stride(0),
            topk_ids,
            topk,
            num_requests,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            PADDED_TOPK=PADDED_TOPK,
        )
    else:
        kernel_fn[grid](
            output,
            logits,
            logits.stride(0),
            topk_ids,
            topk,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            PADDED_TOPK=PADDED_TOPK,
        )
    return output
