"""`x[i + r]` silently reads `x[i]`: the arange term disappears from the address.

The plainest gather there is -- a loaded index plus an arange -- and the arange
is not in the address at all. No error, no refusal, just the wrong elements.

PROVEN, not inferred. Reading the emitted output back and comparing it against
three candidates:

    got vs t[i + r] : 4.417736
    got vs t[i]     : 0.0          <- exact, every element
    got vs t[r]     : 4.122328

WHY IT VANISHES. Adding the two makes addState merge an UNSTRUCTURED dim (the
loaded index: its offset IS a per-element tensor) with a STRUCTURED one (the
arange: offset 0, stride 1, and its index is implicit -- element k contributes
`offset + stride*k`, k is never stored anywhere).

There is code for exactly this hazard, but it only runs on one branch. When the
two sides' offsets AND strides both differ, PtrAnalysis materialises the
structured side's implicit index as `splat(offset) + make_range * splat(stride)`
before merging (the branch triton_shared c19a6f1 added; its own comment spells
out that `offset * stride` folds a structured range to nothing). Here the
strides are EQUAL -- both 1 -- so control never reaches that branch and falls to
the plain "strides equal, so add the offsets" arm instead. The structured side's
offset is 0. Adding 0 is what deletes the term.

So the discriminator is the STRIDE, and it is backwards from what it looks like:
a stride of 1 is the dangerous one, because it is the only one that makes the
two strides match and skips the repair.

    x[i + r]        i32 tile   WRONG, reads x[i]        <- this file
    x[i + r]        i64 tile   WRONG, same way
    x[i + 24*r]     i32 tile   correct (strides 1 vs 24 -> repaired branch)
    x[i + 24*r]     i64 tile   crashes in that branch, a different defect --
                               tests/cases/mechanism/gather/ptr_gather_i64_offset_scaled_merge.py
    x[i + 7]        i32 tile   correct: a splatted literal has stride 0, which
                               is normalised to 1 and carries no implicit index
                               to lose (ptr_gather_1d_arith load 4)

THE ARM IS UPSTREAM'S. It traces to `0614763 Fork at commit 08684f9` in our
triton_shared fork, unchanged since: upstream never materialised a structured
index here at all, and c19a6f1 repaired only the differing-stride half.

THE FIX EXISTS AND WAS MEASURED. triton_shared `0453e53 PtrAnalysis: flatten a
mixed dim whatever the strides are` does the materialisation on both arms, and
keeps the stride-0 case out of it -- a splatted literal holds the same value at
every position and has no walk to invent, and that commit records four kernels
(roi_align and deformable_attn among them) going wrong when it is not kept out.
Built from it and `82208e0` rebased onto the in-use tip 37e28a9: this kernel
PASSES and the suite goes 283 -> 287 of 313, nothing regressing. Not merged to
`pytorchsim`, so it still fails on the pinned binary.

WHY IT WAS NOT SEEN. ptr_gather_1d_arith, the file that walks these arms, has
`i*4 + r*8` (strides differ, repaired), `i*4 + i*8` (same offset tensor on both
sides, no structured term at all) and `i + 7` (stride 0). None of them puts a
stride-1 arange next to a loaded index, which is the one combination that lands
on the unrepaired arm.

torch equivalent: x[i + arange(BLOCK)].

a stride-1 arange merged with a loaded gather index is dropped from the address; the kernel silently reads x[i].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_structured_term_dropped_kernel(x_ptr, i_ptr, o_ptr,
                                                   BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)
        tl.store(o_ptr + r, tl.load(x_ptr + (i + r)))

    return ptr_gather_structured_term_dropped_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, 24, (BLOCK,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    x, i = inputs["x"], inputs["i"].long()
    return {"o": x[i + torch.arange(BLOCK)]}


SPEC = KernelSpec(
    name="ptr_gather_structured_term_dropped_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
