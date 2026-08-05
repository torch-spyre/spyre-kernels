# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.merge_attn_states.original import merge_attn_states_kernel


def merge_attn_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    kernel_fn=merge_attn_states_kernel,
) -> torch.Tensor:
    num_tokens, num_heads, head_size = prefix_output.shape
    padded_head_size = triton.next_power_of_2(head_size)

    output = torch.empty_like(prefix_output)

    grid = (num_tokens, num_heads)
    if "num_tokens" in kernel_fn.arg_names:
        ensure_triton_allocator()
        kernel_fn[grid](
            output,
            None,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            prefix_output.stride(1),
            output.stride(1),
            None,
            num_tokens,
            head_size,
            padded_head_size,
            False,
            num_tokens,
            False,
            FP8_MIN=0.0,
            FP8_MAX=0.0,
        )
    else:
        kernel_fn[grid](
            output,
            None,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            prefix_output.stride(1),
            output.stride(1),
            None,
            head_size,
            padded_head_size,
            False,
            num_tokens,
            False,
            FP8_MIN=0.0,
            FP8_MAX=0.0,
        )
    return output
