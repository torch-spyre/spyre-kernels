"""``torch.zeros.1_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body verbatim, from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.zeros.1`` op
(``aten.full.default([1, 8, 2048, 128], 0, dtype=torch.float16, ...)``); see
``torch-spyre/test_results_triton_20260731_095001/torch.zeros.1_spyre/
torch_compile_debug/run_2026_07_31_10_24_35_361455-pid_450467/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 65536}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.

``in_ptr0`` is the caller-side ``spyre_constant_tensor(0.0, ...)`` "seed" --
a single 64-element zero-valued stick (``desc_0``, ``shape=[1, 1, 1, 64]``).
The kernel loads that one stick, reshapes it (a no-op here, since the
reshape target already matches the loaded shape) and broadcasts it up to
the full store tile (``desc_1``, ``block_shape=[1, 64, 128, 64]``), so every
program writes its 64-row chunk of the output as all-zeros. Every
load/broadcast/store shape lines up exactly (no shape-mismatch bug, unlike
``torch.float.1_spyre``/``torch.index_copy_.2_spyre``).
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_zeros_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 64
    c2 = 0
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = 0
    dim2 = 0
    dim3 = 0
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 1, 1, 64], strides=[64, 64, 64, 1], block_shape=[1, 1, 1, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0 // 64
    dim_1_1 = c1
    dim_1_2 = c2
    dim_1_3 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 2048, 128, 64], strides=[16777216, 8192, 64, 1], block_shape=[1, 64, 128, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp1 = tl.reshape(tmp0, [1, 1, 1, 64])
    tmp2 = tl.broadcast_to(tmp1, [1, 64, 128, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2, dim_1_3], tmp2)
