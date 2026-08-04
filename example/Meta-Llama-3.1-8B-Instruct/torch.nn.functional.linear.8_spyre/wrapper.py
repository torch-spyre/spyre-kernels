"""SIGNATURE + wrapper + reference oracle + input generator for
``torch.nn.functional.linear.8_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``triton_kernel.py``
docstring for the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 4096), (4096, 4096, 1))`` — logical
  activation ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (4096, 4096), (4096, 1))`` — logical
  weight, dtype ``torch.float16``. ``fx_graph_readable.py`` confirms the
  traced op is ``nn.functional.linear(x, weight)`` with **no bias**:
  ``permute(weight,[1,0])`` -> unsqueeze/expand -> ``aten.bmm`` against
  the (unsqueezed/expanded) activation.
- ``triton_bundle_0.run(arg1_1, arg0_1, buf1, buf0, stream=stream0)`` —
  the entry function's own pointer args are, in order,
  ``(weight, x, buf1_scratch, out)``; ``buf1`` is an intermediate
  scratch buffer (weight repacked into a device-tiled layout by
  ``kernel_0``) that ``kernel_1`` consumes as its second input.
- ``triton_meta={..., 'spyre_grids': {'triton_bundle_0_kernel_0': (32,),
  'triton_bundle_0_kernel_1': (32,)}, 'spyre_grid': (32,)}``.

The two helper kernels hardcode their own tensor-descriptor shapes, so
the pointer args here are built directly at those hardcoded physical
shapes (mirroring ``torch.add.1_spyre``'s convention), not at the
logical torch shapes:

- ``in_ptr0`` (weight, consumed by ``kernel_0``'s ``desc_0``): physical
  shape ``[64, 4096, 64]`` (``64*4096*64 == 4096*4096``), a plain
  contiguous reshape of the logical ``(4096, 4096)`` weight.
- ``in_ptr1`` (x, consumed by ``kernel_1``'s ``desc_0``): physical shape
  ``[1, 1, 64, 64]`` (``64*64 == 4096``), a plain contiguous reshape of
  the logical ``(1, 1, 4096)`` activation.
- ``out_ptr0`` (scratch, written by ``kernel_0``'s ``desc_1``, shape
  ``[64, 4096, 64]`` -- coincidentally the same shape as ``in_ptr0``
  here because the weight is square, but a physically distinct buffer).
  Only needs to be large enough (``16777216`` elements); initial content
  is irrelevant since ``kernel_0`` fully overwrites it before
  ``kernel_1`` reads it.
- ``out_ptr1`` (final output, ``kernel_1``'s ``desc_2``): physical shape
  ``[1, 1, 64, 64]`` (``64*64 == 4096``), a plain contiguous reshape of
  the logical ``(1, 1, 4096)`` output.

Unlike ``linear.7``/``linear.11``, ``kernel_1``'s activation descriptor
here (``desc_0``, ``block_shape=[1, 1, 64, 64]``) is fully equal to its
own ``shape`` -- no K-axis truncation. The single ``tl.dot`` genuinely
contracts over the entire ``K=4096`` (``tmp6`` reshaped to ``[1, 4096]``
against ``tmp5`` reshaped to ``[4096, 128]``), so the oracle below (full
``K=4096`` ``nn.functional.linear``, no bias) matches the kernel's own
literal math with no caveat needed on that front.

**Primary blocker (see ``DISABLED_REASON`` below):** same as every other
kernel in this bundled multi-kernel family (``linear.7``-``.11_spyre``):
this kernel does not compile through this test harness's KTIR pipeline.
Compilation fails inside ``backend._make_ktir`` (``PassManager::run
failed``) because ``ConvertFunctions`` (``third_party/spyre/lib/Dialect/
KTDP/Transforms/ConvertFunctions.cpp``) converts every ``triton::FuncOp``
to ``func::FuncOp`` but has no rewrite pattern for ``triton::CallOp`` --
so ``triton_bundle_0``'s ``tt.call`` into its ``noinline=True`` helpers
ends up naming a symbol that no longer resolves to a ``triton::FuncOp``,
which Triton's own ``tt.call`` verifier rejects. See
``torch.nn.functional.linear.7_spyre/wrapper.py`` for the full analysis
(confirmed there by actually running pytest; the root cause is identical
and shape-independent, so it applies here too).
"""

import numpy as np
import torch
import warnings

from . import triton_kernel


# ---------------------------------------------------------------------------
# Shape constants — physical (device-descriptor) shapes, hardcoded in the
# traced kernels themselves (see module docstring); this trace has no
# logical M/K/N convention to derive them from.
# ---------------------------------------------------------------------------

WEIGHT_PHYS_SHAPE = (64, 4096, 64)   # 4096*4096 = 16777216
ACT_PHYS_SHAPE = (1, 1, 64, 64)      # 1*1*4096 = 4096
SCRATCH_PHYS_SHAPE = (64, 4096, 64)  # 16777216
OUT_PHYS_SHAPE = (1, 1, 64, 64)      # 4096


SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
    "out_ptr1": "*fp16",
}

TAGS = ["bundled-multi-kernel", "tl-dot", "weight-repack"]

SUMMARY = (
    "`nn.functional.linear` (no bias) on Meta-Llama-3.1-8B-Instruct's "
    "traced `torch.nn.functional.linear.8` op: x[1,1,4096] @ "
    "weight[4096,4096].T -> out[1,1,4096], as a bundle of a "
    "weight-repack helper feeding a `tl.dot`-based matmul helper."
)

DOC = (
    "`triton_bundle_0` calls `triton_bundle_0_kernel_0` (32 "
    "programs) to repack the raw weight into a device-tiled "
    "scratch buffer, then `triton_bundle_0_kernel_1` (32 "
    "programs, one per 128-wide output-feature tile) to load an "
    "activation tile and a repacked-weight tile, contract them "
    "with a single `tl.dot(..., input_precision=\"ieee\")` over "
    "the full K=4096, and store the f16 result. This description "
    "is of the traced source's intended computation only -- see "
    "DISABLED_REASON below: the kernel does not actually compile "
    "through this harness's KTIR pipeline (same `tt.call`/"
    "`ConvertFunctions` gap as `linear.7_spyre`), so none of "
    "this has been verified to execute end to end."
)

CONSTEXPR = []
GRID = (32,)

OUTPUT_KEY = "out_ptr1"

# Same tt.call/ConvertFunctions gap as linear.7_spyre (confirmed there by
# running pytest; the root cause is identical and shape-independent, so
# it applies here too) -- see the module docstring.
DISABLED_REASON = (
    "Does not compile through this harness's KTIR pipeline: "
    "ConvertFunctions (lib/Dialect/KTDP/Transforms/"
    "ConvertFunctions.cpp) converts every tt.func/tt.return to "
    "func.func/func.return but has no rewrite pattern for "
    "tt.call, so triton_bundle_0's calls into its noinline=True "
    "helpers end up naming a symbol that now resolves to a "
    "func::FuncOp; Triton's own tt.call verifier requires a "
    "triton::FuncOp callee, so backend._make_ktir raises "
    "RuntimeError('PassManager::run failed') with "
    "\"'tt.call' op ... does not reference a valid function\" "
    "before any KTIR is produced. Every structural check and "
    "the numerical comparison fail identically at "
    "setup_method() for this reason (see "
    "torch.nn.functional.linear.7_spyre/wrapper.py for the "
    "pytest-confirmed analysis; identical root cause here, "
    "shape-independent). This is a backend gap for "
    "inter-function tt.call between noinline helpers, common "
    "to every bundled multi-kernel trace in this family, not "
    "an extraction or oracle problem."
)


def linear(in_ptr0: torch.Tensor, in_ptr1: torch.Tensor, kernel_fn=triton_kernel.triton_bundle_0) -> torch.Tensor:
    """nn.functional.linear (no bias): x[1,1,4096] @ weight[4096,4096].T
    -> out[1,1,4096].

    KNOWN BROKEN: see DISABLED_REASON above -- calling this reproduces the
    traced source's genuine tt.call/ConvertFunctions backend gap.
    """
    warnings.warn(DISABLED_REASON, RuntimeWarning)
    out_ptr0 = torch.empty(SCRATCH_PHYS_SHAPE, dtype=torch.float16, device=in_ptr0.device)
    out_ptr1 = torch.empty(OUT_PHYS_SHAPE, dtype=torch.float16, device=in_ptr0.device)
    kernel_fn[GRID](in_ptr0, in_ptr1, out_ptr0, out_ptr1)
    return out_ptr1


def make_inputs(**_unused) -> dict:
    """Build pointer-tensor inputs for the kernel, at each tensor
    descriptor's hardcoded physical shape (see module docstring)."""
    del _unused
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((4096, 4096)).astype(np.float16)
    x = rng.standard_normal((1, 1, 4096)).astype(np.float16)
    in_ptr0 = weight.reshape(64, 4096, 64)
    in_ptr1 = x.reshape(1, 1, 64, 64)
    out_ptr0 = np.zeros((64, 4096, 64), dtype=np.float16)
    out_ptr1 = np.zeros((1, 1, 64, 64), dtype=np.float16)
    return {
        "in_ptr0": in_ptr0,
        "in_ptr1": in_ptr1,
        "out_ptr0": out_ptr0,
        "out_ptr1": out_ptr1,
    }


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: full-K ``nn.functional.linear(x, weight)``, no bias.

    ``out = x @ weight.T`` over the entire ``K=4096`` -- this matches
    both true `torch.nn.functional.linear` semantics *and* what
    ``kernel_1``'s own (untruncated) descriptor/`tl.dot` literally
    computes (see module docstring).
    """
    x = inputs["in_ptr1"].reshape(1, 1, 4096).astype(np.float32)
    weight = inputs["in_ptr0"].reshape(4096, 4096).astype(np.float32)
    out = x @ weight.T  # (1, 1, 4096)
    return out.reshape(1, 1, 64, 64).astype(np.float16)
