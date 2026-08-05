import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.vllm.decode_softmax_reducev.original import _fwd_kernel_stage2


def decode_softmax_reducev(
    mid_o: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
    kernel_fn=_fwd_kernel_stage2,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, _splits, lv_plus_1 = mid_o.shape
    Lv = lv_plus_1 - 1
    BLOCK_DV = triton.next_power_of_2(Lv)

    o = torch.empty(batch, heads, Lv, device=mid_o.device, dtype=mid_o.dtype)
    lse = torch.empty(batch, heads, device=mid_o.device, dtype=torch.float32)

    grid = (batch, heads)
    extra = {}
    if "batch" in kernel_fn.arg_names:
        ensure_triton_allocator()
        extra = {"batch": batch, "heads": heads}
    kernel_fn[grid](
        mid_o,
        o,
        lse,
        b_seq_len,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        **extra,
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )
    return o, lse
