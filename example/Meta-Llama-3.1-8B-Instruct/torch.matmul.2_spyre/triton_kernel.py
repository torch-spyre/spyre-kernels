"""``torch.matmul.2_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 76-95 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.matmul.2`` op
(``aten.bmm.default(expand[1,64,1], expand_1[1,1,1])`` -> ``f16[1,64,1]``);
see ``torch-spyre/test_results_triton_20260731_095001/torch.matmul.2_spyre/
torch_compile_debug/run_2026_07_31_10_39_44_277217-pid_457121/
torchinductor/model__0_inference_0.0/output_code.py``.

Both the batched dimension and the contracted dimension have size 1 here
(unlike ``torch.matmul.1_spyre``, whose second operand carries 12 real
output columns), so ``aten.bmm`` degenerates all the way down to a
broadcast scalar multiply: each of the 64 rows of the first operand
(padded to a full 64-wide f16 "stick", only element 0 of each row
logically real) is multiplied by the *same* single scalar (also padded to
a 64-wide stick, held in ``in_ptr1``), which is loaded once per program and
broadcast across that program's row tile.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 2}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.

Unlike ``torch.float.1_spyre``/``torch.float.3_spyre``, every load/store
shape here is consistent (``tmp4``'s shape ``[1, 2, 64]`` exactly matches
``desc_2``'s ``block_shape``) — this kernel has no shape-mismatch bug.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_bmm_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = (tl.program_id(0)) * 2
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = c0
    dim2 = 0
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 64, 64], strides=[4096, 64, 1], block_shape=[1, 2, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = 0
    dim_1_1 = 0
    dim_1_2 = 0
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[1, 1, 64], strides=[64, 64, 1], block_shape=[1, 1, 64])
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 64, 64], strides=[4096, 64, 1], block_shape=[1, 2, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = desc_1.load([dim_1_0, dim_1_1, dim_1_2])
    tmp2 = tl.reshape(tmp1, [1, 1, 64])
    tmp3 = tl.broadcast_to(tmp2, [1, 2, 64])
    tmp4 = tmp0 * tmp3
    desc_2.store([dim0, dim1, dim2], tmp4)
