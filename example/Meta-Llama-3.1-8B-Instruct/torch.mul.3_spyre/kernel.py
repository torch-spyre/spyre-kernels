"""``torch.mul.3_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 76-96 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.mul.3`` op
(``aten.mul.Tensor(x, y)`` broadcasting a logical flat ``f16[4096]`` tensor
``x`` over the batch dim of a logical ``f16[1, 12, 4096]`` tensor ``y``); see
``torch-spyre/test_results_triton_20260731_095001/torch.mul.3_spyre/
torch_compile_debug/run_2026_07_31_10_08_20_399685-pid_443923/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 1536}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.

Unlike ``torch.mul.1_spyre``/``torch.mul.2_spyre``, Inductor emitted no
``xnumel``/``xoffset``/``xindex``/``xmask`` boilerplate at all for this
trace — every descriptor index (``dim*``) is derived directly from
``tl.program_id(0)`` via ``c0``/``c1``, with ``xnumel``/``XBLOCK`` unused
inside the kernel body (they only size the external launch grid).
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 128
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = c1 // 64
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 64, 64], strides=[4096, 64, 1], block_shape=[1, 2, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0
    dim_1_1 = c1 // 64
    dim_1_2 = c1 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[12, 64, 64], strides=[4096, 64, 1], block_shape=[12, 2, 64])
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 64, 64], strides=[4096, 64, 1], block_shape=[12, 2, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp3 = desc_1.load([dim_1_0, dim_1_1, dim_1_2])
    tmp1 = tl.reshape(tmp0, [1, 2, 64])
    tmp2 = tl.broadcast_to(tmp1, [12, 2, 64])
    tmp4 = tmp2 * tmp3
    desc_2.store([dim_1_0, dim_1_1, dim_1_2], tmp4)
