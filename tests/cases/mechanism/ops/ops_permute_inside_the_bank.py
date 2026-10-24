"""A permutation that leaves the LANE axis where it is, so no element changes lane.

WHAT IT IS FOR. Rule 6a says the only question a layout change has to answer is
whether it moves elements ACROSS the lanes; a change of memory dimension that
does not is no conflict at all. Every permutation the suite emits today moves the
lane axis -- measured, three of them and all three move it -- so the other half
of that rule has never been exercised. This is that half.

THE SHAPE. Axis 0 is the one a rank-3 tile is banked on, and the permutation
swaps the two axes UNDER it:

    x : (4, 8, 16)  --  trans(x, 0, 2, 1)  -->  (4, 16, 8)

Lane L holds the slice i = L before and after. Nothing leaves its bank; what
changes is the ADDRESS of each element inside that bank, which strides can say.

WHAT SHOULD HAPPEN, in order of preference: absorb_layout_ops folds the
permutation into its neighbours and no trip exists; or the trip is kept and paid
for somewhere cheaper than the bus. What must NOT happen is a DRAM round trip,
which crosses the memory bus twice to rearrange data that never left a bank --
and what must REALLY not happen is the trip being dropped, because the elements
do move within the bank and a reader that is not renumbered reads the old
addresses.

WHAT IT MEASURES TODAY, and what it found. The round trip the paragraph above
forbids was there and this file could not see it: `expect_ir` counted moves and
transposes and not relayout buffers, so an MVOUT/MVIN pair through
`relayout1_dram` read as "absorbed". It rebanked the source from lane axis 2 to
lane axis 1 across the memory bus. 35bdfb2 replaced it with one crossbar pass;
the declaration now counts the trip so the swap cannot happen again unseen.

THE ABSORPTION ITSELF IS STILL UNMET, and the reason is upstream of the move.
The premise at the top -- that axis 0 is the one a rank-3 tile is banked on --
is not what fit_layout_to_lanes decides, and the walk that reads the ops is not
what decides it either. Both ends of the permutation are transfer tiles, and
`_apply_dim_order_from_dram_stride` runs first: it banks each tile on the axis
DRAM is most contiguous along, which is axis 2 for the source (dram_stride
[128, 16, 1]) and axis 2 for the destination ([128, 8, 1]). Under perm
[0, 2, 1] the destination's lane axis holds the source's axis 1, so the ends do
not correspond and `_rewrite_transpose_to_xlu_move`'s `perm[a_dst] == a_src`
cannot divert it to an in-bank copy. Axis 1 on the source would have been free.

A TRANSPOSE IS THE OP FOR WHICH "BOTH ENDS CONTIGUOUS" MEANS "THE ENDS
DISAGREE". By the time the walk reaches it `_apply_dim_order` holds both
buffers claimed and will not revisit, so the one op that knows the two tiles
are the same data never gets a vote.

THE EXTENTS ARE DIFFERENT ON PURPOSE. 8 and 16, not 8 and 8: a square swap is
type-preserving, so an answer that loses the permutation still verifies as a
type. Here losing it changes the shape, and `_drop_flat_relayouts` -- which
compares the two ends' types -- cannot mistake this one for flat the way it
mistook the cube in loop_11.

A rank-3 permutation that fixes the lane axis and swaps the two axes under it. No element changes lane, so rule 6a says this is not a conflict -- the suite has no other kernel where a permutation and the lane axis are independent.
"""

#: RE-PINNED 2026-09-23 TO THE GRAPH ROUTE. The numbers were right all
#: along -- this kernel passes with the pins ignored -- and what the old
#: pin held was a PATH the route no longer takes: CHEAPER NOW: neither the unit nor the bus, the turn happening inside each bank.
#: The declaration above is what it does today, so a change to it is seen;
#: the line above says which way the old path was better, where it was.
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

A, B, C = 4, 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def permute_inside_the_bank_kernel(x_ptr, w_ptr, o_ptr, A: tl.constexpr,
                                       B: tl.constexpr, C: tl.constexpr):
        a = tl.arange(0, A)
        b = tl.arange(0, B)
        c = tl.arange(0, C)
        x = tl.load(x_ptr + a[:, None, None] * (B * C)
                    + b[None, :, None] * C + c[None, None, :])
        y = tl.trans(x, 0, 2, 1)                     # (A, C, B), lane axis stays
        off = (a[:, None, None] * (C * B)
               + c[None, :, None] * B + b[None, None, :])
        tl.store(o_ptr + off, y + tl.load(w_ptr + off))

    return permute_inside_the_bank_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(A * B * C, generator=g, dtype=torch.float32),
            "w": torch.randn(A * C * B, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(A, B, C)
    w = inputs["w"].reshape(A, C, B)
    return {"o": (x.permute(0, 2, 1) + w).reshape(-1)}


SPEC = KernelSpec(
    name="permute_inside_the_bank_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "A": "constexpr", "B": "constexpr", "C": "constexpr"},
    constexprs={"A": A, "B": B, "C": C},
    args=[Arg("x", IN, "float32", (A * B * C,)),
          Arg("w", IN, "float32", (A * C * B,)),
          Arg("o", OUT, "float32", (A * C * B,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    # THE VALUES CANNOT SAY IT. A round trip here would be correct and wasteful,
    # so `reference` scores the good answer and the bad one alike; the claim is
    # that the permutation is ABSORBED, and only the artifact can carry it.
    expect_ir={"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0, "pytorchsim.fold": 0,
                                   "linalg.transpose": 0, "relayout": 0}},
)
