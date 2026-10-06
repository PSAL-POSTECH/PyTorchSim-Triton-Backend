"""An i64 gather index merged with a strided arange: the splat is built at i32.

This is what stops DeepSeek-V3's MoE router,
`triton_npu_fused_gather_mul_sigmoid_41`, at stage 3:

    tmp6 = tl.load(in_ptr1 + (tmp4 + 24*x0), xmask, ...)
                                   ^ col 37
    error: 'tensor.splat' op failed to verify that operand type matches
           element type of result
    note:  %48 = "tensor.splat"(%47) : (i32) -> tensor<128xi64>

`tmp4` is the expert index gathered from an *i64 buffer (topk's output), `x0`
is the i32 program index, and the two meet in one address.

THE ARM THIS REACHES. addState merges a structured dim with an unstructured one.
When their offsets AND strides both differ (PtrAnalysis.cpp, the
`lhsOffset != rhsOffset && lhsStride != rhsStride` branch) it flattens each side
to a plain per-element offset. The unstructured side already IS a per-element
tensor, so it is just scaled. The structured side's index is implicit -- element
k contributes `offset + stride*k` and k is never stored -- so it has to be
MATERIALISED here as `splat(offset) + make_range * splat(stride)`.

WHERE THE WIDTH IS LOST. That materialisation builds its row tensor from the
OTHER side's element type (`shapedTy.getElementType()`, i64 here) but converts
the scalars with a helper that hardcodes i32:

    if (isa<IndexType>(v.getType()))
      v = arith::IndexCastOp::create(builder, loc, builder.getI32Type(), v)

so it asks for `tensor.splat : (i32) -> tensor<128xi64>`, and tensor.splat's
verifier requires the scalar to match the result's element type. Nothing is
miscompiled -- it never gets that far. i32 was the only width this path had seen
because a gather index that comes from an i32 buffer makes the whole address
i32; an int64 index buffer is what makes the row tensor i64.

MEASURED MATRIX (develop f6232e1, and unchanged by 7e48c5f's index-buffer fix).
The discriminator is a NON-UNIT stride on the term merged with the i64 tile, not
"does it have an i64 tile" and not "does it have a multiply":

    i + 24*r    i64   FAIL  tensor.splat (i32) -> tensor<128xi64>   <- this file
    24*i + r    i64   FAIL  same
    i + 24*r    i32   PASS                                          <- control
    i + 24      i64   PASS  scalar addend: stride 0 -> normalised to 1, equal
    2 * i       i64   PASS  mulState alone: the stride goes ON the pointer op
    i           i64   PASS  no merge at all
    i + r       i64   FAIL  numerically -- a DIFFERENT defect, see
    i + r       i32   FAIL  tests/cases/mechanism/gather/ptr_gather_structured_term_dropped.py

SIBLING: tests/cases/mechanism/gather/ptr_gather_1d_arith.py walks all four addState
arms at i32 and documents which operand shape picks which. This file is the same
branch with an i64 offset tensor.

THE FIX EXISTS AND WAS MEASURED. triton_shared `82208e0 PtrAnalysis: build the
flattened offset in the offset tensor's own type` converts to the row's element
type instead of i32, extending `tt.make_range` (which has no form but i32) when
they differ. Built from that commit and `0453e53` rebased onto the in-use tip
37e28a9: this kernel PASSES and the suite goes 283 -> 287 of 313, the only
verdicts that move being this file, ptr_gather_structured_term_dropped,
ptr_gather_nested_modulo_of_arange and dl/ms_deform_attn, all FAIL -> PASS.
Nothing regresses. Not merged to `torchsim`, so it still fails on the pinned
binary -- measuring it against a build of `gather-offset-elem-type` AS IS scores
three unrelated regressions instead, because that branch predates 37e28a9,
9017bd4 and df7452a.

torch equivalent: t[i + 24*arange(BLOCK)].

i64 gather index merged with a strided arange: PtrAnalysis builds the structured side's splat at i32 into an i64 row tensor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NT = 4096
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_i64_offset_scaled_merge_kernel(t_ptr, i_ptr, o_ptr,
                                                   BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)                      # i64, so the address is i64
        tl.store(o_ptr + r, tl.load(t_ptr + (i + 24 * r)))

    return ptr_gather_i64_offset_scaled_merge_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"t": torch.randn(NT, generator=g, dtype=torch.float32),
            "i": torch.randint(0, 24, (BLOCK,), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    import torch
    t, i = inputs["t"], inputs["i"].long()
    return {"o": t[i + 24 * torch.arange(BLOCK)]}


SPEC = KernelSpec(
    name="ptr_gather_i64_offset_scaled_merge_kernel", kernel=kernel,
    signature={"t_ptr": "*fp32", "i_ptr": "*i64", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("t", IN, "float32", (NT,)), Arg("i", IN, "int64", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
