"""``torch.nn.functional.silu.1_spyre`` kernels extracted from a torch-spyre
Inductor trace.

Unlike every other op in this example set, Inductor does **not** fuse
``silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`` into a single kernel here:
it lowers to 5 independently-launched top-level ``@triton.jit`` kernels
(``clone -> neg -> exp -> (+1) -> div``), each called separately in
``Runner.call`` by reusing buffers. All 5 bodies below are verbatim, from
torch-spyre's Inductor output for Meta-Llama-3.1-8B-Instruct's traced
``torch.nn.functional.silu.1`` op; see
``torch-spyre/test_results_triton_20260731_095001/torch.nn.functional.silu.1_spyre/
torch_compile_debug/run_2026_07_31_10_33_07_133162-pid_454384/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction, for each of the 5: the ``@triton_heuristics.fixed_config(...)``
decorator that wraps the kernel in the Inductor output. It only carries
autotuning/Inductor-side metadata (``config={'XBLOCK': 5376}``,
``triton_meta``, ``inductor_meta``) and pulls in ``torch._inductor.runtime``
as a dependency; the ``@triton.jit`` function it wraps is unchanged either
way, and this test suite compiles ``kernel_fn`` directly via ``ASTSource``
(see ``meta.py``), never through that decorator.

All 5 kernels operate on the same logical ``f16[1, 12, 14336]`` shape /
physical ``[12, 224, 64]`` device layout (224 rows of 64-element f16
sticks), tiled 7 rows at a time (``block_shape=[12, 7, 64]``) across a
32-program grid (``32 * 7 == 224``, exact). ``triton_unk_fused_0`` (clone)
and ``triton_unk_fused_silu_1`` (neg) read one buffer and write another;
``triton_unk_fused_silu_2`` (exp) and ``triton_unk_fused_silu_3`` (+1) both
mutate their buffer in place (``in_out_ptr0``); ``triton_unk_fused_silu_4``
(div) reads the original clone (``in_ptr0``) and divides it by the
accumulated ``exp(-x) + 1`` buffer (``in_out_ptr0``) in place, producing the
final ``silu(x)`` result. No bug in this trace -- each stage's arithmetic is
correct for its role in the 5-stage decomposition.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 448
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1 // 64
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    desc_1.store([dim0, dim1, dim2], tmp0)


@triton.jit
def triton_unk_fused_silu_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 448
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1 // 64
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = -tmp0
    desc_1.store([dim0, dim1, dim2], tmp1)


@triton.jit
def triton_unk_fused_silu_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 448
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1 // 64
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_out_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = tl.exp(tmp0.to(tl.float32)).to(tl.float16)
    desc_0.store([dim0, dim1, dim2], tmp1)


@triton.jit
def triton_unk_fused_silu_3(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 448
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1 // 64
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_out_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp0 + tmp1
    desc_0.store([dim0, dim1, dim2], tmp2)


@triton.jit
def triton_unk_fused_silu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 448
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1 // 64
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_out_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    desc_1 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 224, 64], strides=[14336, 64, 1], block_shape=[12, 7, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = desc_1.load([dim0, dim1, dim2])
    tmp2 = (tmp0 / tmp1).to(tl.float16)
    desc_0.store([dim0, dim1, dim2], tmp2)
