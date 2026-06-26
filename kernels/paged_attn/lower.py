# SPDX-License-Identifier: Apache-2.0
"""KTIR lowering driver for the paged_attn kernel variants.

Consumed by ``scripts/gen_ktir.py``, which lowers each entry in ``VARIANTS`` to
``kernels/paged_attn/<variant>.ktir``.

``VARIANTS`` maps a **variant name** (which is also the source ``.py`` module
name and the output ``.ktir`` stem) to the four things the round-trip lowering
needs:

    KERNEL      : the @triton.jit function to lower
    SIGNATURE   : dict[str, str]  arg name -> Triton type ("*fp16", "i32", ...)
    CONSTEXPRS  : dict[str, value] for every constexpr arg
    GRID        : optional list, forwarded to SpyreOptions.grid

Two variants share the same signature and shapes; they differ only in how the
data-dependent K/V gather is expressed:
  - ``tensor_descriptor``: base descriptor_gather (cache viewed 2-D, 1-D rows).
  - ``spyre_aware``: extended any-rank descriptor_gather (cache 3-D, 2-D index).
"""

from kernels.paged_attn.tensor_descriptor import _paged_attn_kernel_NHD_td
from kernels.paged_attn.spyre_aware import _paged_attn_kernel_NHD_sa

# Concrete shapes match tests/ktir/test_paged_attn.py: B=2, H=4, Lq=Lk=16,
# D=64, cache of 256 slots, KV_BLOCK=16, BLK_B=2, BLK_H=4. The problem dims are
# constexprs (descriptor shapes are built from them), so they are baked into the
# KTIR; only the buffers and the scalar scale stay runtime args.
_SIGNATURE = {
    "Q": "*fp16",
    "K": "*fp16",
    "V": "*fp16",
    "SLOTS": "*i32",
    "Out": "*fp16",
    "scale": "fp32",
    "B": "i32",
    "H": "i32",
    "Lq": "i32",
    "Lk": "i32",
    "CACHE": "i32",
    "KV_BLOCK": "i32",
    "BLOCK_Q": "i32",
    "BLOCK_D": "i32",
    "BLK_B": "i32",
    "BLK_H": "i32",
}

_CONSTEXPRS = {
    "B": 2,
    "H": 4,
    "Lq": 16,
    "Lk": 16,
    "CACHE": 256,
    "KV_BLOCK": 16,
    "BLOCK_Q": 16,
    "BLOCK_D": 64,
    "BLK_B": 2,
    "BLK_H": 4,
}

VARIANTS = {
    "tensor_descriptor": {
        "KERNEL": _paged_attn_kernel_NHD_td,
        "SIGNATURE": _SIGNATURE,
        "CONSTEXPRS": _CONSTEXPRS,
        # Single program walks all (B, Lq, H) work via the explicit loops; fits 32 cores.
        "GRID": [1],
    },
    "spyre_aware": {
        "KERNEL": _paged_attn_kernel_NHD_sa,
        "SIGNATURE": _SIGNATURE,
        "CONSTEXPRS": _CONSTEXPRS,
        "GRID": [1],
    },
}
