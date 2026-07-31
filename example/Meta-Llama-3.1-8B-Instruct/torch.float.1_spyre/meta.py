"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.float.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 64, 1), (64, 1, 1))`` — logical input,
  dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 64, 1), (64, 1, 1), torch.float32,
  SpyreTensorLayout(device_size=[64, 1, 1, 32], ...,
  element_arrangement=ElementArrangement.DL16_TO_FP32))`` — logical output,
  same logical shape as the input but dtype ``torch.float32``.
- ``triton_unk_fused__to_copy_0.run(arg0_1, buf0, 64, stream=stream0)`` —
  the kernel is launched with ``xnumel=64``.
- ``config={'XBLOCK': 2}``, ``triton_meta={..., 'spyre_grid': (32,)}`` — a
  32-program grid, each program covering 2 rows of the middle (logical)
  dimension (``32 * 2 == 64 == xnumel``).

Unlike ``torch.float.3_spyre`` (single-program, whole-buffer tiles, matching
*element counts* on both sides of the store despite mismatched *shapes*),
here the per-program tiles have mismatched *element counts* too:
``desc_0``'s ``block_shape=[1, 2, 64]`` (128 elements) is loaded, cast
in place to fp32 (shape-preserving), and stored into ``desc_1`` whose
``block_shape=[1, 2, 32]`` (64 elements) — see the ``BUG`` note in
``kernel.py`` and ``VARIANTS["default"]["disabled"]`` below. This makes the
op even more clearly non-executable than ``torch.float.3_spyre``'s
count-matched-but-shape-mismatched case, so no attempt is made here to
derive a physically-faithful reshape for the oracle; ``run()`` below is a
placeholder documenting the *intended* elementwise cast only, never
exercised because the variant is disabled.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    Built directly at each descriptor's full (non-tiled) ``shape`` —
    ``in_ptr0`` at ``[1, 64, 64]`` (fp16) and ``out_ptr0`` at
    ``[1, 64, 32]`` (fp32, zero-initialized) — per ``kernel.py``.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((1, 64, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 64, 32), dtype=np.float32)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """Placeholder NumPy oracle for the *intended* elementwise fp16->fp32
    cast (`aten._to_copy`/`prims.convert_element_type.default`). Never
    exercised: the traced kernel's store has a genuine shape mismatch (see
    module docstring and ``kernel.py``'s ``BUG`` note), so this variant is
    marked ``disabled`` and this oracle is not a faithful physical-layout
    derivation the way ``torch.float.3_spyre``'s is."""
    return inputs["in_ptr0"][:, :, :32].astype(np.float32)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "out_ptr0": "*fp32",
    "xnumel":   "i32",
    "XBLOCK":   "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "dtype-cast", "program-id-1d"],
        "summary": (
            "fp16 -> fp32 dtype cast (`.float()`) on Meta-Llama-3.1-8B-"
            "Instruct's traced `torch.float.1` op, 32-program grid, "
            "relayouts a 64-wide fp16 stick into a 32-wide fp32 stick per "
            "row -- but with a genuine per-tile element-count mismatch "
            "in the store (see `disabled` below)."
        ),
        "doc": (
            "Casts every element of a logical `[1, 64, 1]` f16 tensor to "
            "f32 (`prims.convert_element_type.default`). On the Spyre "
            "device layout the fp16 input pads its innermost dim to a "
            "64-element stick (`in_ptr0` full shape `[1, 64, 64]`), while "
            "the fp32 output uses a narrower 32-element stick (`out_ptr0` "
            "full shape `[1, 64, 32]`). The grid is 32 programs "
            "(`spyre_grid=(32,)`), each covering 2 rows of the middle "
            "(logical) dimension via `c0 = program_id(0) * 2` "
            "(`32 * 2 == 64 == xnumel`). Each program's load tile "
            "(`block_shape=[1, 2, 64]`, 128 elements) is cast in place to "
            "fp32 and stored directly into a descriptor whose tile is "
            "`block_shape=[1, 2, 32]` (64 elements) -- a real shape *and* "
            "element-count mismatch present verbatim in the traced "
            "source, with no intervening reshape."
        ),
        "kernel_fn":  kernel.triton_unk_fused__to_copy_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [64], "XBLOCK": [2]},
        "grid":       [32],
        # desc_1.store() passes desc_0.load()'s value (shape [1,2,64],
        # unchanged by the shape-preserving fp32 cast) directly into a
        # descriptor whose block_shape is [1,2,32] -- different last-dim
        # extent and different total element count, with no tl.reshape.
        # Present verbatim in the traced output_code.py (see kernel.py).
        # Same bug category as torch.cat.*_spyre / torch.float.3_spyre,
        # rejected by Triton's frontend validate_store_like check before
        # any TTIR is built.
        "disabled": {
            "reason": (
                "desc_1.store() passes a [1,2,64] value (from "
                "desc_0.load(), shape-preserved through the fp32 cast) "
                "into a descriptor with block_shape [1,2,32] with no "
                "reshape -- a shape/element-count mismatch present "
                "verbatim in the traced source, rejected by Triton's "
                "frontend validate_store_like check before TTIR "
                "construction."
            ),
        },
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
