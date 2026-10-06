"""A gathered matmul operand whose tile and INDEX are in a merged axis class,
so the class no longer names an axis and the two are banked apart.

WHAT IT PINS. `_maps_of` unions a gather's index buffer with the tile it
addresses, axis for axis -- the DMA reads the index at the same banked slot it
writes the element to, so the two are one layout. That union is right and it is
not enough: a class answers "which axis is this" only while it holds ONE axis of
each buffer, and here it holds BOTH of the tile's.

    B tile     128x128 f32   pinned to axis 1 by the matmul's MUST
    B index    128x128 i64   no demand of its own -> `mine`'s setdefault -> 0

`axis_of` says of that state that it "cannot say I do not know": every branch
returns an axis, and inside a merged class the answer is a ranking rather than a
reading. The tile and its index are ranked separately and come out different.

HOW THE CLASS MERGES, AND IT IS A CYCLE RATHER THAN AN OP. `p` is one [M, R]
tile and it is read TWICE:

    as A's gather index      its axis 1 is the matmul's K
    elementwise against acc  its axis 1 is the matmul's N

Both readings are correct and each is an ordinary identity map. Together they
union K with N, and B is the buffer that has one of each, so B's two axes end in
one class. No single op merges them; the cycle closes at whichever op comes
last. This is why `R` is the extent of both the reduction and the output axis --
one tile cannot span two axes of different lengths, and the merge needs one tile
in both roles.

WHAT GOES WRONG WHEN IT IS NOT FIXED. It COMPILES, and that is the point.
Measured on this file against the tree before the repair:

    stage 4-7   ok, every stage
    verify      1023 of 2048 elements wrong, max_abs_err 5.12109

The IR shows nothing (rule 12); the DMA's own index log is where it is visible.
Spike's `indirect_access/indirect_index1.raw` is B's gather index as the DMA
actually read it, and the two trees differ in exactly the way the mismatch
predicts:

    before   94  94  94  94  94  94  94  94      constant along the lane
    after    94 101  79 119  43 118 120  90      bidx[n], one per lane

Constant along the lane is the whole defect: every lane reads the index the
lane-0 row happens to hold, so the tile is a splat of one column of B. The same
state was first measured on moonlight's grouped MoE matmul, kernel 35, where
`dram_stride[0] = 1` turned it into `8960 + 81k` and B came back as B[k, k].

WHAT BREAKS THIS. Reverting `_cut_gather_index` -- the step-4 repair that
gives a gather its own index copy when the two came out banked apart.

THE EDGE IT CUTS IS NOT THE ONE THAT COMPLAINS, and that is the whole shape of
the repair. The disagreement is reported on the matmul's B; cutting B's own
index buys nothing, because the copy lands in the same merged class and is
ranked the same way. What repairs it is cutting the OTHER gather -- the one
whose index chain runs through the elementwise ops that union K with N -- so
the candidates are every indirect transfer's index operand, ranked by what the
copy costs, applied one at a time and withdrawn unless fewer transfers
disagree.

TWO OTHER REPAIRS WERE MEASURED AND ARE WRONG, both worth naming because both
looked right. A `_gathers` correspondence ("the index takes the tile's axis",
written the way `_folds` is) moves the mismatch one op upstream, to the copy
that FILLS the index buffer, and vectorize refuses there instead. A flood
from the MUSTs through the ops' maps fixes this file exactly and REGRESSES
tests/models/test_resnet.py -- it shipped as a1b8115 and was reverted in
a812db1 -- because it answers only the merged buffers, so inside one class the
merged ones take the flood and the unmerged ones keep the class answer and the
two disagree. ResNet's generic is three operands under identity maps, banked
[0, 0, 1].

AND CUTTING EVERY GATHER'S INDEX, WHICH IS THE OBVIOUS SIMPLIFICATION, IS WHAT
RULE 8b CALLS A FAILURE RATHER THAN A COST:

    conditional (shipped)   336 of 360, 109 round trips,   0 pointless
    unconditional           333 of 360, 589 round trips, 434 POINTLESS

WHAT THE REPAIR COST, AND WHAT IT COSTS NOW. Measured 2026-08-31 it cost one
round trip and the counter called it POINTLESS -- `torchsim.relayout = [1, 1]`, same
tile, same `dram_stride [128, 1]`. Both ends landed on axis 1, so the MVOUT and
the MVIN wrote and read the same bytes in the same layout and the copy's ONLY
effect was that a `torchsim.relayout` is in NO_EDGE, which separates the classes:

    class rooted at axis 1 of %alloc_5   holds BOTH axes of %alloc_12, _13, _14

Measured today the repair is free: 0 round trips here and 0 pointless anywhere
in the suite, so the merge is answered without buying a fence. Which change made
it free is not established here. The reading this paragraph named is still the
one to keep -- `axis_of` answering a doubled class from the demand on it rather
than from `mine`'s positional ranking, the seam `_demands_met` names.

WHAT THE SUITE ALONE DOES NOT CATCH. The flood above passed regress.py twice,
byte-identical, 336 of 360, zero PASS -> FAIL, and still broke ResNet end to
end -- no coverage kernel has its conv/batchnorm shape. Any change to this
repair has to run tests/models/test_resnet.py BEFORE the push (rule 13a), not
after.

ARITHMETIC. Exact: the matmul is over f32 rows that are small integers scaled by
2^-4, so every partial sum is representable and the tolerance is 0.

a gathered matmul operand whose tile and index land in one merged axis class, so the class names no axis and they are banked apart. POSITIVE -- without the repair it compiles and returns wrong numbers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 16          # rows
R = 128         # BOTH the reduction axis and the output axis -- see above
ROWS = 16       # how many rows of `a` there are to gather from


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_index_in_a_merged_class_kernel(o_ptr, a_ptr, b_ptr, sel_ptr,
                                              q_ptr, bidx_ptr, M: tl.constexpr,
                                              R: tl.constexpr):
        m = tl.arange(0, M)[:, None]
        r = tl.arange(0, R)[None, :]
        kk = tl.arange(0, R)[:, None]

        # ONE TILE IN TWO ROLES, AND IT HAS TO BE GENUINELY 2-D. A per-row value
        # broadcast across the row does NOT work: the broadcast is absorbed into
        # the DMA's `dram_stride = [1, 0]`, so triton-shared emits the load
        # TWICE, once per consumer, and the two roles never meet. Measured --
        # `tl.broadcast_to(sel, [M, R])` gives two `dram_arg = 3` transfers into
        # two buffers and this file then passes with the repair and without it.
        # Adding a per-COLUMN term is what makes the tile real.
        sel = tl.load(sel_ptr + m)
        q = tl.load(q_ptr + r)
        p = sel * R + q

        # ROLE ONE: p is A's gather index, so p's axis 1 is the matmul's K.
        a = tl.load(a_ptr + p)

        # B IS GATHERED TOO, and by a LOADED column index -- that is what makes
        # it an `indirect` transfer with a real index buffer to disagree with.
        # Row steps 1 in DRAM, the column comes from memory, which is the shape
        # kernel 35 has.
        bi = tl.load(bidx_ptr + r)
        b = tl.load(b_ptr + (kk + bi * R))

        acc = tl.dot(a, b)

        # ROLE TWO: the SAME p, read elementwise against the matmul's result,
        # so p's axis 1 is now the matmul's N. This is the edge that closes the
        # cycle; without it every buffer here has one axis per class.
        out = tl.where((p % 2) == 0, acc, 0.0)
        tl.store(o_ptr + (m * R + r), out)

    return gather_index_in_a_merged_class_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-8, 8, (ROWS * R,), generator=g).to(torch.float32) / 16.0
    b = torch.randint(-8, 8, (R * R,), generator=g).to(torch.float32) / 16.0
    return {"a": a, "b": b,
            "sel": torch.randperm(ROWS, generator=g)[:M].to(torch.int32),
            "q": torch.randperm(R, generator=g).to(torch.int32),
            "bidx": torch.randperm(R, generator=g).to(torch.int32)}


def reference(inputs):
    import torch
    sel = inputs["sel"].long()
    q = inputs["q"].long()
    bidx = inputs["bidx"].long()
    p = sel.reshape(M, 1) * R + q.reshape(1, R)            # [M, R]
    a = inputs["a"].reshape(-1)[p]                         # [M, R]
    # b[k, n] = b_mem[k + bidx[n] * R] -- the loaded column index, transposed
    b = inputs["b"].reshape(R, R)[bidx].t().contiguous()   # [R, R]
    acc = a @ b
    keep = (p % 2) == 0
    return {"o": torch.where(keep, acc, torch.zeros_like(acc))}


SPEC = KernelSpec(
    name="gather_index_in_a_merged_class_kernel",
    kernel=kernel,
    signature={"o_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "sel_ptr": "*i32", "q_ptr": "*i32", "bidx_ptr": "*i32",
               "M": "constexpr", "R": "constexpr"},
    constexprs={"M": M, "R": R},
    args=[Arg("o", OUT, "float32", (M, R)),
          Arg("a", IN, "float32", (ROWS * R,)),
          Arg("b", IN, "float32", (R * R,)),
          Arg("sel", IN, "int32", (M,)),
          Arg("q", IN, "int32", (R,)),
          Arg("bidx", IN, "int32", (R,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0.0,
    atol=0.0,
)
