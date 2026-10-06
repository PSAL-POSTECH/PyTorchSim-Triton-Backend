"""A reshape neither side can MERGE, and both sides can SPLIT.

WHAT THIS PINS: the common refinement. A reshape between two shapes is one
statement about a run of elements, and absorbing it means one of its neighbours
saying what the other one says. There are two ways to ask that and both used to
be the whole search:

    forwards    the READERS restate on the operand's shape
    backwards   the PRODUCER restates on the result's shape

and one of those two is always a MERGE -- the one thing in this pass that can
refuse, because merging axes asks for a single stride that indexes all of them
and a strided tile has none. So a reshape of a strided tile had no plan at all,
and this file's predecessor concluded that "the tile is not a single contiguous
run of DRAM, so no reshape of it is expressible as one descriptor".

THAT WAS A STRONGER QUESTION THAN THE MACHINE ASKS. A descriptor is sizes and
strides; it expresses any strided access for free. What costs anything here is
elements moving between LANES, and a relinearisation moves none. The third
candidate is to meet in the middle: both shapes are cuts of the same run, so cut
it at the UNION of their cuts and both sides get there by SPLITTING, which never
refuses.

    4x8 cuts 32 8 1      8x4 cuts 32 4 1      union 32 8 4 1  ->  4x2x4

    MVIN   dram_stride [16, 1] over 4x8   ->  [16, 4, 1] over 4x2x4
    MVOUT  dram_stride  [8, 1] over 8x4   ->  [16, 8, 1] over 4x2x4

Two splits, no merge, and the reshape is gone. Both descriptors carry the
refined rank, which is the point: the refinement is a property of the GROUP, so
every value on the plan takes it or none of them does (rule 4).

WHY THE TILE IS STRIDED, AND IT IS THE WHOLE OF THE TEST. `S > C` makes the 4x8
window a strided read of a 4x16 buffer, so `_merge_strides` has nothing to
return for the backward plan and the forward plan is a merge on the store side.
Set `S == C` and both of the old two plans work and this kernel stops pinning
anything.

WHAT IT FAILS WITH WITHOUT IT. `SPEC.extra["forbids"]` stops absorb_layout_ops at the moment
it leaves the op standing:

    absorb_layout_ops left 1 op it declared would not survive it:
      tensor.reshape x1

ITS NEIGHBOUR IN THE SUITE is unsupported/forbids_transpose_below_absorb, which
is this kernel with a `tl.sum` in place of the store -- and it still refuses,
because a `linalg.reduce`'s `dimensions` is a list of axis NAMES and an axis
that became two has no name. Same reshape, same refinement, different reader:
that pair is what separates "no descriptor expresses this" (false) from "this
statement has no reading under a split" (true).

torch equivalent: a strided 4x8 window read back as 8x4

Pins the common refinement as a third candidate (rule 6). There is no arithmetic in this kernel at all, so the tolerance is exact and any mismatch is a layout fault.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 4, 8
S = 16          # the row stride of the LOADED window. S > C leaves the MVIN
                # strided, so the backward plan has no merge.
T = 8           # the row stride of the STORED window. T > R leaves the MVOUT
                # strided too, so the forward plan has none either. Set either
                # of them equal and one of the old two plans works and this
                # kernel stops pinning anything.


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reshape_refinement_kernel(x_ptr, o_ptr, R: tl.constexpr,
                                      C: tl.constexpr, S: tl.constexpr,
                                      T: tl.constexpr):
        r = tl.arange(0, R)
        c = tl.arange(0, C)
        a = tl.load(x_ptr + r[:, None] * S + c[None, :])       # strided 4x8
        b = tl.reshape(a, (C, R))                              # 4x8 -> 8x4
        oc = tl.arange(0, C)
        orr = tl.arange(0, R)
        tl.store(o_ptr + oc[:, None] * T + orr[None, :], b)    # strided 8x4

    return ops_reshape_refinement_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    m = inputs["x"].reshape(R, S)
    o = torch.zeros(C, T, dtype=torch.float32)
    o[:, :R] = m[:, :C].contiguous().reshape(C, R)
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ops_reshape_refinement_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr",
               "C": "constexpr", "S": "constexpr", "T": "constexpr"},
    constexprs={"R": R, "C": C, "S": S, "T": T},
    args=[Arg("x", IN, "float32", (R * S,)),
          Arg("o", OUT, "float32", (C * T,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
    extra={"forbids": {"psto-absorb-layout-ops": True}},
)
