"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.mul.3_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py``'s docstring for
the exact path):

- ``assert_size_stride(arg0_1, (4096,), (1,))`` — logical flat operand,
  dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1, 12, 4096), (49152, 4096, 1))`` — logical
  second operand, same dtype.
- ``buf0 = spyre_empty_with_layout((1, 12, 4096), (49152, 4096, 1),
  torch.float16, SpyreTensorLayout(device_size=[12, 64, 1, 64], ...))`` —
  logical output, same shape/dtype as ``arg1_1``.
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 49152, stream=stream0)``
  — launched with ``xnumel=49152``.
- ``config={'XBLOCK': 1536}``, ``triton_meta={..., 'spyre_grid': (32,)}`` —
  32-program grid, each covering 2 units (``c1 // 64``) of the padded
  64-wide middle dim (``32 * 2 == 64``).

``arg0_1``'s flat `[4096]` shape has no batch dim, so `aten.mul.Tensor`
broadcasts it across all 12 rows of `arg1_1`: on the device layout, both
operands' 4096-wide axis is split into a `[64, 64]` grid of 64-element f16
sticks, and `in_ptr0`'s `[1, 64, 64]` tile is broadcast over `in_ptr1`'s
`[12, 64, 64]` batch dim before the elementwise multiply.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    ``xnumel``/``XBLOCK`` are accepted so the signature matches the full
    param set, but neither shapes the data (the kernel body derives every
    descriptor index from ``tl.program_id(0)`` alone, see ``kernel.py``):
    the buffers are built directly at each descriptor's full ``shape`` --
    ``in_ptr0`` at ``[1, 64, 64]`` and ``in_ptr1``/``out_ptr0`` at
    ``[12, 64, 64]``.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((1, 64, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((12, 64, 64)).astype(np.float16)
    out_ptr0 = np.zeros((12, 64, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.mul.Tensor`` with ``in_ptr0`` broadcast across
    ``in_ptr1``'s batch dim, in the kernel's own compute precision (f32
    multiply, truncate back to f16)."""
    x = inputs["in_ptr0"].astype(np.float32)                 # [1, 64, 64]
    y = inputs["in_ptr1"].astype(np.float32)                  # [12, 64, 64]
    out = x * y
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
    "xnumel":   "i32",
    "XBLOCK":   "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d"],
        "summary": (
            "Broadcast elementwise multiply of a flat `f16[4096]` operand "
            "over the batch dim of a `f16[1, 12, 4096]` operand, on "
            "Meta-Llama-3.1-8B-Instruct's traced `torch.mul.3` op."
        ),
        "doc": (
            "`aten.mul.Tensor` where the first operand (`in_ptr0`, logical "
            "`f16[4096]`) has no batch dim, so it broadcasts across all 12 "
            "rows of the second operand (`in_ptr1`, logical "
            "`f16[1, 12, 4096]`). On the Spyre device layout the shared "
            "4096-wide axis splits into a `[64, 64]` grid of 64-element "
            "f16 sticks; `tl.program_id(0)` selects a 2-stick-wide tile "
            "(`c1 = program_id(0) * 128`, `dim1 = c1 // 64`) per program, "
            "loading `in_ptr0`'s tile once and broadcasting it over "
            "`in_ptr1`'s full 12-row batch before the multiply. Unlike "
            "`torch.mul.1_spyre`/`torch.mul.2_spyre`, Inductor emitted no "
            "`xnumel`/`xoffset`/`xindex`/`xmask` boilerplate here at all -- "
            "every descriptor index comes straight from `tl.program_id(0)`."
        ),
        "kernel_fn":  kernel.triton_unk_fused_mul_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [49152], "XBLOCK": [1536]},
        "grid":       [32],
        # grid (32) * 2 (dim1 step size) == 64 (the padded middle dim's
        # stick count): every program gets a fixed 2-stick chunk with
        # nothing left over, same reasoning as torch.mul.1_spyre's
        # distribution_loop: False.
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
