"""A narrowing pointer bitcast on a chain PtrAnalysis has ALREADY refused.

    pb = tl.cast(x_ptr + i, tl.pointer_type(tl.int8))   # i is a runtime index
    tl.load(pb)                                         # byte 4*i[k]

The sibling of ptr_bitcast_narrows_the_element.py, and the other half of the
same contract. There the offset is a RANGE and PtrAnalysis crosses the cast by
multiplying the whole PtrState; here the offset is a LOADED tensor, which
visitOperandBitcast refuses to rescale -- offsets[d] * strides[d] would be
scaled twice and the factor cannot multiply a per-element tensor at all -- so
the chain falls to TritonToUnstructured, which must do the same multiplication
on its own carrier.

WHY IT NEEDS ITS OWN FILE. The two walks keep one predicate
(Utils::ptrCastElementRatio) and have to USE it the same way. They did not: the
structured walk multiplied and the unstructured one refused. MEASURED, all
three states of that guard on this kernel:

    refusing    numerically RIGHT and a rung lower -- the chain leaves the tts
                path and convert_unsupported_pointers rebuilds it. So the asymmetry was not a wrong
                answer; it was a rung, which is why nothing caught it.
    removed     tts.gather %p[%i] with the offset NOT multiplied -- byte i[k]
                instead of byte 4*i[k]. Silently wrong, so the fix is the
                multiplication and not the deletion.
    rescaling   arith.muli %i, dense<4> then tts.gather. This file.

AND THE POSITION WAS ITS OWN DEFECT. The guard sat ahead of the checks for
whose pointer this is, so it reported on pointers PtrAnalysis had already
finished: ptr_bitcast_narrows_the_element printed "a pointer bitcast that
changes the element size would rescale every offset" on the scalar base under
its own FINISHED tts.make_gather_scatter_tptr, while passing. Asked after
offsetMap, that stops.

WHAT IT PINS
  the ratio is asked where the walk OWNS the pointer, after offsetMap says so
  the offset is multiplied by it, so `tts.gather` reads 4*i[k]
  the base is re-bitcast, so the gather's pointee matches what triton loads

torch equivalent: x reinterpreted as int8, read at byte 4*i[k].

A narrowing pointer bitcast whose offset is a loaded tensor: the structured walk refuses to rescale a gathered dim, so the unstructured walk has to multiply its own offset by the same ratio.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bitcast_narrows_on_a_gathered_chain(x_ptr, i_ptr, o_ptr,
                                            BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)
        # THE OFFSET IS A LOADED TENSOR when the cast happens, so no descriptor
        # axis can carry the factor -- the per-element walk multiplies instead.
        pb = tl.cast(x_ptr + i, tl.pointer_type(tl.int8))
        tl.store(o_ptr + r, tl.load(pb))

    return bitcast_narrows_on_a_gathered_chain


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK, generator=g, dtype=torch.float32),
            "i": torch.randint(0, BLOCK, (BLOCK,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    """The byte at 4*i[k] of x, read as int8."""
    import torch
    b = inputs["x"].view(torch.int8)
    return {"o": b[4 * inputs["i"].to(torch.int64)].contiguous()}


SPEC = KernelSpec(
    name="bitcast_narrows_on_a_gathered_chain", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*i8",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (BLOCK,)), Arg("i", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "int8", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 1}},
)
