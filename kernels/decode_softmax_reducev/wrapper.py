import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.decode_softmax_reducev.original import _fwd_kernel_stage2
from kernels.decode_softmax_reducev.spyre import _fwd_kernel_stage2_spyre


def decode_softmax_reducev(
    mid_o: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
    kernel_fn=_fwd_kernel_stage2,
    tile_size_dv: int = 64,
    block_bh: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, _splits, lv_plus_1 = mid_o.shape
    Lv = lv_plus_1 - 1
    BLOCK_DV = triton.next_power_of_2(Lv)

    o = torch.empty(batch, heads, Lv, device=mid_o.device, dtype=mid_o.dtype)
    lse = torch.empty(batch, heads, device=mid_o.device, dtype=torch.float32)

    if kernel_fn is _fwd_kernel_stage2_spyre:
        ensure_triton_allocator()

        total_bh = batch * heads
        bh_tiles = triton.cdiv(total_bh, block_bh)
        num_programs = min(32, bh_tiles)

        # Expand seq_lens: [batch] → [batch*heads] via repeat_interleave
        b_seq_len_expanded = b_seq_len.repeat_interleave(heads)

        grid = (num_programs,)
        kernel_fn[grid](
            mid_o,
            o,
            lse,
            b_seq_len_expanded,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            o.stride(0),
            o.stride(1),
            lse.stride(0),
            batch,
            heads,
            NUM_KV_SPLITS=num_kv_splits,
            BLOCK_SIZE=tile_size_dv,
            BLOCK_BH=block_bh,
            Lv=Lv,
        )
    else:
        grid = (batch, heads)
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
            NUM_KV_SPLITS=num_kv_splits,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            num_warps=4,
            num_stages=2,
        )
    return o, lse
