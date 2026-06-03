import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.reshape_and_cache.original import reshape_and_cache_kernel_flash


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kernel_fn=reshape_and_cache_kernel_flash,
) -> None:
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    n = num_heads * head_size

    TILE_SIZE = min(1024, triton.next_power_of_2(n))
    num_tokens = slot_mapping.shape[0]
    grid = (num_tokens, triton.cdiv(n, TILE_SIZE))

    extra = {}
    if "num_tokens" in kernel_fn.arg_names:
        ensure_triton_allocator()
        extra["num_tokens"] = num_tokens

    kernel_fn[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=0,
        v_scale=0,
        key_stride=key.stride(0),
        value_stride=value.stride(0),
        block_stride=key_cache.stride(0),
        head_stride=key_cache.stride(2),
        dim_stride_k=0,
        dim_stride_v=0,
        page_stride=key_cache.stride(1),
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=1,
        USE_HEAD_MAJOR_LAYOUT=False,
        FP8_KV_CACHE=False,
        TILE_SIZE=TILE_SIZE,
        **extra,
    )
