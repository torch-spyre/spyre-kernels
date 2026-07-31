"""``torch.index_copy_.3_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body verbatim, from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.index_copy_.3`` op
(``aten.index_put_.default(arg0_1, [None, None, arg1_1], arg2_1)``, a KV-cache
style scatter along dim 2 of a ``f16[1, 8, 2048, 128]`` buffer -- byte-for-byte
the same kernel body as ``torch.index_copy_.2_spyre``, confirmed by comparing
both traces' ``output_code.py``); see
``torch-spyre/test_results_triton_20260731_095001/torch.index_copy_.3_spyre/
torch_compile_debug/run_2026_07_31_11_02_24_296942-pid_467066/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 64}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.

**Genuine trace-level bug** (see ``meta.py``'s ``disabled`` entry): the kernel
loads the index tensor (``in_ptr0``, ``desc_0``) and computes ``tmp1``/
``tmp2`` (a reshape + broadcast of it) but never uses either value in the
final ``desc_2.store(...)`` — it stores ``tmp3`` (loaded straight from
``in_ptr1`` via ``desc_1``) instead, i.e. this is an unconditional overwrite,
not a real indexed copy. Worse, the shapes don't even line up for that
overwrite: ``tmp3`` has shape ``[1, 1, 1, 1, 64]`` (``desc_1``'s
``block_shape``), but ``desc_2.store(...)`` targets a block of shape
``[1, 2048, 1, 1, 64]`` — a genuine element-count mismatch (64 vs. 131072
elements), same class of bug as ``torch.float.1_spyre``.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_index_copy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = (tl.program_id(0) // 2)
    c1 = ((tl.program_id(0) % 2)) * 64
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = 0
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 32], strides=[32, 1], block_shape=[1, 32])
    # Logical layouts -> Device layouts
    dim_1_0 = c0
    dim_1_1 = 0
    dim_1_2 = c1 // 64
    dim_1_3 = 0
    dim_1_4 = c1 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[8, 1, 2, 1, 64], strides=[128, 128, 64, 64, 1], block_shape=[1, 1, 1, 1, 64])
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[8, 2048, 2, 1, 64], strides=[262144, 128, 64, 64, 1], block_shape=[1, 2048, 1, 1, 64])
    tmp0 = desc_0.load([dim0, dim1])
    tmp3 = desc_1.load([dim_1_0, dim_1_1, dim_1_2, dim_1_3, dim_1_4])
    tmp1 = tl.reshape(tmp0, [1, 1, 1, 1, 32])
    tmp2 = tl.broadcast_to(tmp1, [1, 2048, 1, 1, 64])
    desc_2.store([dim_1_0, dim_1_1, dim_1_2, dim_1_3, dim_1_4], tmp3)
