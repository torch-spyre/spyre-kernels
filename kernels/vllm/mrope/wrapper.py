import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.vllm.mrope.original import _triton_mrope_forward


def triton_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    kernel_fn=_triton_mrope_forward,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_row = q.shape[0]
    n_q_head = q.shape[1] // head_size
    n_kv_head = k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    ensure_triton_allocator()

    kernel_fn[(n_row,)](
        q, k, cos, sin,
        n_row,
        n_q_head, n_kv_head,
        head_size, rotary_dim,
        pad_n_q_head, pad_n_kv_head, pad_hd,
        mrope_section[0], mrope_section[1], mrope_section[2],
        mrope_interleaved,
    )
    return q, k
