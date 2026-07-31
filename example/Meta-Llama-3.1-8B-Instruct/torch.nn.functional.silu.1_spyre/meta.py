"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.silu.1_spyre``.

Unlike every other op in this example set, this trace lowers to 5
independently-launched ``@triton.jit`` kernels rather than one fused kernel
(see ``kernel.py``'s docstring for the full breakdown and the exact traced
source path). Each of the 5 stages is exposed as its own entry in
``VARIANTS`` below, with its own ``kernel_fn`` and a per-variant
``SIGNATURE`` override where its pointer args differ from the module-level
``SIGNATURE`` (the ``example/conftest.py`` harness's ``_resolve_variant``
already supports this via ``entry.get("SIGNATURE", module_sig)``).

Caller-side settings, common to all 5 stages, are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py``:

- ``assert_size_stride(arg0_1, (1, 12, 14336), (172032, 14336, 1))`` —
  logical input, dtype ``torch.float16``.
- All intermediate/output buffers are ``spyre_empty_with_layout((1, 12,
  14336), (172032, 14336, 1), torch.float16, SpyreTensorLayout(device_size=
  [12, 224, 1, 64], ...))`` — same logical shape/dtype throughout.
- Every ``.run(...)`` call passes ``xnumel=172032``.
- ``config={'XBLOCK': 5376}``, ``triton_meta={..., 'spyre_grid': (32,)}`` —
  a 32-program grid, each covering 7 rows of the padded ``[224, 64]``
  device layout (``32 * 7 == 224``, exact).
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker, one pair per stage
# ---------------------------------------------------------------------------

def _rand_buf(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((12, 224, 64)).astype(np.float16)


def make_inputs_clone(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Stage 1 (``clone``): ``in_ptr0`` -> ``out_ptr0``, both ``[12, 224,
    64]``. ``xnumel``/``XBLOCK`` are accepted for signature parity but
    unused -- every descriptor index derives from ``tl.program_id(0)``."""
    del xnumel, XBLOCK
    in_ptr0 = _rand_buf(0)
    out_ptr0 = np.zeros((12, 224, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run_clone(inputs: dict) -> np.ndarray:
    """NumPy oracle: identity copy (``aten.clone``)."""
    return inputs["in_ptr0"].copy()


def make_inputs_neg(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Stage 2 (``neg``): ``in_ptr0`` -> ``out_ptr0``, both ``[12, 224,
    64]``."""
    del xnumel, XBLOCK
    in_ptr0 = _rand_buf(1)
    out_ptr0 = np.zeros((12, 224, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run_neg(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.neg.default``."""
    return (-inputs["in_ptr0"].astype(np.float32)).astype(np.float16)


def make_inputs_exp(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Stage 3 (``exp``): in-place on ``in_out_ptr0``, ``[12, 224, 64]``."""
    del xnumel, XBLOCK
    in_out_ptr0 = _rand_buf(2)
    return {"in_out_ptr0": in_out_ptr0}


def run_exp(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.exp.default``, in the kernel's own compute
    precision (widen to f32, ``exp``, truncate back to f16)."""
    return np.exp(inputs["in_out_ptr0"].astype(np.float32)).astype(np.float16)


def make_inputs_add_one(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Stage 4 (``+1``): in-place on ``in_out_ptr0``, ``[12, 224, 64]``."""
    del xnumel, XBLOCK
    in_out_ptr0 = _rand_buf(3)
    return {"in_out_ptr0": in_out_ptr0}


def run_add_one(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.add.Tensor(x, 1.0)``, computed in f32 (matching
    ``tmp1 = tl.full([1], 1.0, tl.float32)``) then truncated to f16."""
    return (inputs["in_out_ptr0"].astype(np.float32) + 1.0).astype(np.float16)


def make_inputs_div(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Stage 5 (``div``): in-place on ``in_out_ptr0``, reading ``in_ptr0``
    as the divisor's numerator source; both ``[12, 224, 64]``."""
    del xnumel, XBLOCK
    in_out_ptr0 = _rand_buf(4)
    # avoid a near-zero divisor
    in_ptr0 = (_rand_buf(5).astype(np.float32) + 2.0).astype(np.float16)
    return {"in_out_ptr0": in_out_ptr0, "in_ptr0": in_ptr0}


def run_div(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.div.Tensor(in_out_ptr0, in_ptr0)``, i.e. the
    final ``silu(x) = x / (1 + exp(-x))`` division, computed in f32 then
    truncated to f16 (matching ``(tmp0 / tmp1).to(tl.float16)``)."""
    x = inputs["in_out_ptr0"].astype(np.float32)
    y = inputs["in_ptr0"].astype(np.float32)
    return (x / y).astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE — module-level default is the clone (stage 1) signature; other
# stages override via their own VARIANTS["SIGNATURE"] entry.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "out_ptr0": "*fp16",
    "xnumel":   "i32",
    "XBLOCK":   "i32",
}

_INOUT_ONLY_SIGNATURE = {
    "in_out_ptr0": "*fp16",
    "xnumel":       "i32",
    "XBLOCK":       "i32",
}

_DIV_SIGNATURE = {
    "in_out_ptr0": "*fp16",
    "in_ptr0":      "*fp16",
    "xnumel":       "i32",
    "XBLOCK":       "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS -- one per stage of the 5-kernel silu(x) = x / (1 + exp(-x))
# decomposition.
# ---------------------------------------------------------------------------

_COMMON = {
    "constexpr":  ["XBLOCK"],
    "params":     {"xnumel": [172032], "XBLOCK": [5376]},
    "grid":       [32],
    # grid (32) * 7 (row step size) == 224 (the padded axis's row count):
    # every program gets a fixed 7-row chunk with nothing left over.
    "distribution_loop": False,
}

VARIANTS = {
    "default": {
        **_COMMON,
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d"],
        "summary": (
            "Stage 1/5 of `torch.nn.functional.silu.1_spyre`: `aten.clone` "
            "of the input, on Meta-Llama-3.1-8B-Instruct's traced op."
        ),
        "doc": (
            "First of 5 independently-launched kernels implementing "
            "`silu(x) = x / (1 + exp(-x))`. This stage clones the logical "
            "`f16[1, 12, 14336]` input verbatim; the clone is read again by "
            "stage 5 (`div`) after stages 2-4 compute `1 + exp(-x)` "
            "in-place on a second buffer."
        ),
        "kernel_fn":  kernel.triton_unk_fused_0,
        "reference":  run_clone,
        "inputs":     make_inputs_clone,
        "output_key": "out_ptr0",
    },
    "neg": {
        **_COMMON,
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d"],
        "summary": (
            "Stage 2/5 of `torch.nn.functional.silu.1_spyre`: `aten.neg` "
            "of the clone, on Meta-Llama-3.1-8B-Instruct's traced op."
        ),
        "doc": (
            "Second of 5 stages: negates the stage-1 clone "
            "(`tmp1 = -tmp0`), producing `-x` ahead of the `exp` stage."
        ),
        "kernel_fn":  kernel.triton_unk_fused_silu_1,
        "reference":  run_neg,
        "inputs":     make_inputs_neg,
        "output_key": "out_ptr0",
    },
    "exp": {
        **_COMMON,
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "in-place"],
        "summary": (
            "Stage 3/5 of `torch.nn.functional.silu.1_spyre`: in-place "
            "`aten.exp` of `-x`, on Meta-Llama-3.1-8B-Instruct's traced op."
        ),
        "doc": (
            "Third of 5 stages: computes `exp(-x)` in place on the "
            "stage-2 buffer (widened to f32 for the `tl.exp` call, then "
            "truncated back to f16)."
        ),
        "kernel_fn":  kernel.triton_unk_fused_silu_2,
        "SIGNATURE":  _INOUT_ONLY_SIGNATURE,
        "reference":  run_exp,
        "inputs":     make_inputs_exp,
        "output_key": "in_out_ptr0",
    },
    "add_one": {
        **_COMMON,
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "in-place"],
        "summary": (
            "Stage 4/5 of `torch.nn.functional.silu.1_spyre`: in-place "
            "`aten.add.Tensor(x, 1.0)`, on Meta-Llama-3.1-8B-Instruct's "
            "traced op."
        ),
        "doc": (
            "Fourth of 5 stages: adds the f32 constant `1.0` in place to "
            "the stage-3 `exp(-x)` buffer, producing `1 + exp(-x)` (the "
            "sigmoid denominator)."
        ),
        "kernel_fn":  kernel.triton_unk_fused_silu_3,
        "SIGNATURE":  _INOUT_ONLY_SIGNATURE,
        "reference":  run_add_one,
        "inputs":     make_inputs_add_one,
        "output_key": "in_out_ptr0",
    },
    "div": {
        **_COMMON,
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "in-place"],
        "summary": (
            "Stage 5/5 of `torch.nn.functional.silu.1_spyre`: in-place "
            "`aten.div.Tensor`, producing the final `silu(x)` result, on "
            "Meta-Llama-3.1-8B-Instruct's traced op."
        ),
        "doc": (
            "Fifth and final stage: divides the stage-1 clone (`in_ptr0`, "
            "the original `x`) by the stage-4 buffer (`in_out_ptr0`, "
            "`1 + exp(-x)`) in place, i.e. `x / (1 + exp(-x)) = "
            "x * sigmoid(x) = silu(x)`. No bug across this 5-stage "
            "decomposition -- each stage's arithmetic is correct for its "
            "role."
        ),
        "kernel_fn":  kernel.triton_unk_fused_silu_4,
        "SIGNATURE":  _DIV_SIGNATURE,
        "reference":  run_div,
        "inputs":     make_inputs_div,
        "output_key": "in_out_ptr0",
    },
}
