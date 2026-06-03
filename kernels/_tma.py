"""Shared utilities for kernel wrappers using tensor descriptors."""

import torch
import triton

_allocator_set = False


def ensure_triton_allocator():
    """Register a default allocator for TMA descriptors.

    tl.make_tensor_descriptor needs a global memory allocation; on
    Hopper+ GPUs Triton uses TMA and on older GPUs it falls back to
    emulation. Either way an allocator must be set once per process.
    """
    global _allocator_set
    if _allocator_set:
        return

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    _allocator_set = True
