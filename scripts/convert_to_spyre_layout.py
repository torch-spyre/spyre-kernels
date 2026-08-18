# SPDX-License-Identifier: Apache-2.0
"""Convert a logical PyTorch tensor to its Spyre physical ("sticked") layout.

The physical layout of a tensor on Spyre is decided by ``SpyreTensorLayout``
(``device_size`` / ``stride_map``) — a pure function of ``(shape, dtype,
dim_order)``, independent of any device. This script builds that layout and
reorders/pads the tensor into physical shape entirely on the host, so it needs
no Spyre device, no ``torch.compile``, and no kernel run.

The reorder itself is delegated to torch-spyre's own host-side stickifier
(``torch_spyre._inductor_triton.ktir_layout``), which is the exact code the
KTIR-CPU execution path uses — so this stays in lockstep with the real layout,
rather than reimplementing the identity
``logical_offset = sum_i device_coord[i] * stride_map[i]``.

Library use:

    from scripts.convert_to_spyre_layout import to_spyre_layout, from_spyre_layout
    phys, layout = to_spyre_layout(torch.randn(200, 4, 64, dtype=torch.float16))
    logical = from_spyre_layout(phys, (200, 4, 64), layout)

CLI use (needs the torch-spyre env):

    source ~/dev-env.sh
    python3 scripts/convert_to_spyre_layout.py --shape 200 4 64 --dtype float16
    python3 scripts/convert_to_spyre_layout.py --shape 512 256 --dtype float16 \
        --dim-order 1 0            # stick on dim 0 (e.g. a Linear weight)
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import torch

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor_triton.ktir_layout import (
    ktir_destickify,
    ktir_stickify,
)


def _row_major_strides(size: Sequence[int]) -> list[int]:
    strides = [1] * len(size)
    for k in range(len(size) - 2, -1, -1):
        strides[k] = strides[k + 1] * int(size[k + 1])
    return strides


def build_layout(
    shape: Sequence[int],
    dtype: torch.dtype,
    dim_order: Optional[Sequence[int]] = None,
) -> SpyreTensorLayout:
    """Construct the ``SpyreTensorLayout`` for a logical ``(shape, dtype)``.

    ``dim_order`` selects which logical dim is stickified (default: the last).
    When given, the explicit 4-arg constructor is used with row-major (contiguous)
    host strides.
    """
    shape = [int(s) for s in shape]
    if dim_order is None:
        # Default layout: stickify along the last dimension.
        return SpyreTensorLayout(shape, dtype)
    return SpyreTensorLayout(
        shape,
        _row_major_strides(shape),
        dtype,
        [int(d) for d in dim_order],
    )


def to_spyre_layout(
    logical: torch.Tensor,
    *,
    device_dtype: Optional[torch.dtype] = None,
    dim_order: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, SpyreTensorLayout]:
    """Logical tensor -> (physical-layout tensor, its ``SpyreTensorLayout``).

    ``device_dtype`` defaults to the input tensor's dtype (the on-device dtype
    the layout is computed for; e.g. pass ``torch.float16`` to downcast an fp32
    host tensor to the Spyre default).
    """
    dev_dtype = device_dtype if device_dtype is not None else logical.dtype
    layout = build_layout(list(logical.shape), dev_dtype, dim_order)
    physical = ktir_stickify(logical, layout)
    return physical, layout


def from_spyre_layout(
    physical: torch.Tensor,
    logical_size: Sequence[int],
    layout: SpyreTensorLayout,
) -> torch.Tensor:
    """Physical-layout tensor -> logical tensor (inverse of ``to_spyre_layout``)."""
    return ktir_destickify(physical, list(logical_size), layout)


# ─── CLI ────────────────────────────────────────────────────────────

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "int32": torch.int32,
    "int8": torch.int8,
}


def _make_logical(shape, dtype, fill, seed):
    torch.manual_seed(seed)
    if fill == "arange":
        n = 1
        for s in shape:
            n *= s
        return torch.arange(n).reshape(shape).to(dtype)
    if dtype in (torch.int32, torch.int8):
        return torch.randint(0, 100, tuple(shape), dtype=dtype)
    return torch.randn(*shape).to(dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", type=int, nargs="+", required=True,
                    help="logical tensor shape, e.g. --shape 200 4 64")
    ap.add_argument("--dtype", default="float16", choices=sorted(_DTYPES),
                    help="device dtype (default: float16)")
    ap.add_argument("--dim-order", type=int, nargs="+", default=None,
                    help="dim_order for stickification (default: stick last dim)")
    ap.add_argument("--fill", default="randn", choices=["randn", "arange"],
                    help="how to populate the logical tensor (default: randn)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dtype = _DTYPES[args.dtype]
    logical = _make_logical(args.shape, dtype, args.fill, args.seed)

    physical, layout = to_spyre_layout(logical, dim_order=args.dim_order)

    print(f"logical      shape={tuple(logical.shape)}  dtype={dtype}")
    if args.dim_order is not None:
        print(f"dim_order    {args.dim_order}")
    print(f"device_size  {list(layout.device_size)}")
    print(f"stride_map   {list(layout.stride_map)}")
    print(f"device_dtype {layout.device_dtype}")
    print(f"physical     shape={tuple(physical.shape)}  dtype={physical.dtype}")

    # Round-trip: destickify(stickify(x)) must recover x on the valid positions.
    recovered = from_spyre_layout(physical, args.shape, layout)
    ok = torch.equal(recovered.to(logical.dtype), logical)
    print(f"round-trip   {'OK (exact)' if ok else 'MISMATCH'} "
          f"(recovered shape={tuple(recovered.shape)})")
    if not ok:
        # fp downcast can make it non-exact; report the max delta as a fallback.
        delta = (recovered.to(torch.float32) - logical.to(torch.float32)).abs().max()
        print(f"             max|Δ| = {float(delta):.6g}")


if __name__ == "__main__":
    main()
