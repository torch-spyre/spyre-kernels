"""``torch.float.1_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 74-86 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.float.1`` op
(``prims.convert_element_type.default(arg0_1, torch.float32)`` — an
``fp16 -> fp32`` dtype cast, i.e. the ``.float()`` call — on a logical
``f16[1, 64, 1]`` tensor); see
``torch-spyre/test_results_triton_20260731_095001/torch.float.1_spyre/
torch_compile_debug/run_2026_07_31_09_52_11_340036-pid_437992/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 2}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.

Like ``torch.float.3_spyre``, the two tensor descriptors have different
*dtypes and physical shapes* (``in_ptr0``: fp16, full shape ``[1, 64, 64]``,
per-program ``block_shape=[1, 2, 64]``; ``out_ptr0``: fp32, full shape
``[1, 64, 32]``, per-program ``block_shape=[1, 2, 32]``) — the fp32 output's
narrower 32-element stick vs. the fp16 input's 64-element stick.

BUG (present verbatim in the traced source): ``desc_0.load`` returns a value
of shape ``[1, 2, 64]`` (``desc_0``'s own ``block_shape``); this is cast in
place (``.to(tl.float32)``, shape-preserving) and passed directly to
``desc_1.store``, whose ``block_shape`` is ``[1, 2, 32]`` — a *different*
last-dim extent (64 vs. 32) and a different total element count (128 vs.
64), with no intervening ``tl.reshape`` or slice. Same bug category as
``torch.cat.*_spyre`` and ``torch.float.3_spyre`` (a store whose value shape
doesn't match its descriptor's ``block_shape``), rejected by Triton's
frontend ``validate_store_like`` check before any TTIR is built. See
``meta.py``'s ``VARIANTS["default"]["disabled"]``.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = (tl.program_id(0)) * 2
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = c0
    dim2 = 0
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 64, 64], strides=[4096, 64, 1], block_shape=[1, 2, 64])
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 64, 32], strides=[2048, 32, 1], block_shape=[1, 2, 32])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = tmp0.to(tl.float32)
    desc_1.store([dim0, dim1, dim2], tmp1)
