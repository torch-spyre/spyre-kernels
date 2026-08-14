# SPDX-License-Identifier: Apache-2.0
"""Validate the generated paged-attention KTIR against a NumPy reference.

Two variants are generated from the Triton kernel source by
``scripts/gen_ktir.py`` (driver: ``kernels/paged_attn/lower.py``):

    tensor_descriptor.ktir  ->  func @_paged_attn_kernel_NHD_td   (base gather)
    spyre_aware.ktir        ->  func @_paged_attn_kernel_NHD_sa   (any-rank gather)

Both have identical semantics, so they are checked against the same NumPy paged
SDPA reference. The function name and arg order mirror the lowered kernel; the
problem dims and tile sizes are constexprs baked into the KTIR (see lower.py),
so only the buffers and the scalar scale are runtime args:

    %arg0 Q [B, Lq, H, D] f16   %arg1 K   %arg2 V   (cache buffers)
    %arg3 SLOTS [B, Lk] i32     %arg4 Out [B, H, Lq, D] f16   %arg5 scale f32

The reference is NumPy (f32 accumulation), so this runs with only ktir_cpu — no
GPU / Spyre-Triton build needed.

Run:
    .venv/bin/python -m pytest tests/ktir/test_paged_attn.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter

KERNELS_DIR = Path(__file__).resolve().parent.parent.parent / "kernels" / "paged_attn"

# Concrete shapes baked into the generated KTIR (see kernels/paged_attn/lower.py).
B, H, Lq, Lk, D, CACHE = 2, 4, 16, 16, 64, 256
# scale applied to q and k each, so scores carry scale**2 == 1/sqrt(D)
SCALE = np.float32(1.0 / np.sqrt(np.sqrt(D)))

# variant -> (ktir filename, lowered function name)
VARIANTS = {
    "tensor_descriptor": ("tensor_descriptor.ktir", "_paged_attn_kernel_NHD_td"),
    "spyre_aware": ("spyre_aware.ktir", "_paged_attn_kernel_NHD_sa"),
}


def numpy_reference(q, k, v, slots):
    """Paged SDPA in f32, matching the kernel's upcast-accumulate-downcast.

    q (B,Lq,H,D) f16; k,v (CACHE,H,D) f16; slots (B,Lk) i32 -> (B,H,Lq,D) f16.
    """
    sm_scale = float(SCALE) * float(SCALE)  # == 1/sqrt(D)
    out = np.zeros((B, H, Lq, D), np.float32)
    for b in range(B):
        idx = slots[b]
        kb = k[idx].astype(np.float32)  # (Lk, H, D)
        vb = v[idx].astype(np.float32)
        for h in range(H):
            qf = q[b, :, h, :].astype(np.float32)  # (Lq, D)
            s = (qf @ kb[:, h, :].T) * sm_scale
            s -= s.max(axis=1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=1, keepdims=True)
            out[b, h] = w @ vb[:, h, :]
    return out.astype(np.float16)


def _run(variant, q, k, v, slots):
    fname, func = VARIANTS[variant]
    mlir_path = KERNELS_DIR / fname
    if not mlir_path.is_file():
        pytest.skip(f"{fname} not generated yet (run scripts/gen_ktir.py paged_attn)")

    interp = KTIRInterpreter()
    interp.load(mlir_path.read_text())

    out = np.zeros((B, H, Lq, D), dtype=np.float16)
    # td views K/V 2-D as (CACHE, H*D); sa views them 3-D as (CACHE, H, D).
    # Both are the same contiguous buffer — pass the rank the descriptor expects.
    if variant == "tensor_descriptor":
        k_arg = k.reshape(CACHE, H * D)
        v_arg = v.reshape(CACHE, H * D)
    else:
        k_arg = k
        v_arg = v

    outputs = interp.execute_function(
        func,
        arg0=q,
        arg1=k_arg,
        arg2=v_arg,
        arg3=slots,
        arg4=out,
        arg5=SCALE,
    )
    return outputs["arg4"]


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
def test_paged_attn_ktir(variant):
    """Generated paged-attention KTIR matches the NumPy reference within tol."""
    rng = np.random.default_rng(42)
    q = (rng.standard_normal((B, Lq, H, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((CACHE, H, D)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((CACHE, H, D)) * 0.1).astype(np.float16)
    slots = np.stack(
        [rng.permutation(CACHE)[:Lk] for _ in range(B)]
    ).astype(np.int32)

    result = _run(variant, q, k, v, slots)
    expected = numpy_reference(q, k, v, slots)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-1, atol=5e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS [{variant}]: max abs error = {max_err:.6f}")
