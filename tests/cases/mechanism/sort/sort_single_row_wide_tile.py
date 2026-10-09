"""A sort whose PARALLEL extent is 1, so the tile is all lane and no data.

torch equivalent: x.sort() on a single 64-element row (a MoE router's expert
ranking, which is where this came from).

WHAT MAKES IT DIFFERENT FROM sort_row_tl.py. That kernel sorts 128 rows, so the
row index is a real parallel axis and the lanes have something to hold. Here
there is ONE row. Inductor still compiles it with `XBLOCK` at the lane count --
deliberately, see PyTorchSim 8d4f21c: a parallel block of 1 leaves no axis for
the lanes, the tile banks on the REDUCTION axis instead, and stage 4 refuses
that for a reduction because every lane then needs the scalar the reduction
produced.

So the row is REPLICATED across 128 lanes and every intermediate of the sort
network is 128x the data. Measured on Moonlight at 16 experts
(`triton_npu_fused_copy__sort_40`, xnumel 1, r0_numel 64):

    320 scratchpad globals, 104224 bytes/lane, over the 65536 budget
    the data itself is 64 values + 64 indices

and the caller's shrink-the-block retry cannot reach it, because the waste is
on the LANE axis and the retry only moves reduction blocks -- the remeasured
usage came back byte-identical at R0_BLOCK 32 and 16.

THE OTHER DIRECTION IS ALSO REFUSED, which is why this is `blocked` rather than
a block-size bug. Recompiling the same kernel with `XBLOCK = 1`:

    NotVectorisable: operands of one elementwise op are banked on different
    iteration dims [0, 1]; they share an iteration space and cannot share a
    loop nest
      ins  memref<1x32xi64>     outs memref<2x1x32xi64>

That is NOT 8d4f21c's refusal (no scalar is broadcast back here -- a sort's
result is the whole row). It is the lane axis splitting INSIDE the sort
network, at the join that puts two half-rows back together.

So neither setting works, and the fix is in how the lane axis is chosen for a
kernel whose parallel extent is 1, not in the block size. Until then this is
what the failure looks like, and a change that claims to fix it has to make
this kernel pass.

WHAT **THIS** KERNEL STOPS ON, WHICH IS NOT THE MODEL'S LINE. Measured here:

    [4/8] adapt   ok       (89.1s)
    [5/8] lower   BLOCKED  NotImplementedError:
                           vlane split axis 0 peeled into the outer loop nest

It never reaches the scratchpad check, because it sorts VALUES ONLY. The
router sorts values and indices together, and that index half is what doubles
the live tiles into the overflow. Reproducing it exactly needs
`triton_helpers.sort_with_index`, which a kernel here cannot import: stage 1
runs the ttir worker under `mlir-env`'s python, which has no torch. Writing
the network out by hand (as sort_bitonic.py does) would pin the network rather
than the shape this file is about.

So the two failures are one root -- a row replicated across every lane because
the parallel axis has extent 1 -- reported at two different stages, and this
file pins the earlier one. The 104224 bytes/lane above is evidence about the
model, not about this kernel.

A THIRD FACE OF THE SAME ROOT, measured at Moonlight's real width (`--preset
block`, 64 routed experts, XBLOCK left at 128 because the scratchpad check is
never reached):

    [6/8] binary  FAIL  a vector is wider than one instruction can reach:
                        <128 x i64> needs 8192 bits (LMUL 16). The budget is
                        VLEN 512 x LMUL 8 = 4096 bits.

The element count is chosen against the type the chain STARTS with -- f32, 128
of them -- and the sort carries an INDEX beside each value, widened to i64 by
the time it is stored. Same 128 elements, twice the width. The replication is
what makes 128 the count in the first place: with a real parallel axis the tile
would be sized by the work rather than by the lane count.

    experts 16 (medium)  scratchpad 104224 > 65536      stage 6, spad check
    experts 64 (block)   <128 x i64> 8192 > 4096 bits   stage 6, mlir-translate
    XBLOCK 1             lane axis splits in the network stage 4, vectorize

WHAT WOULD BREAK IT (once it passes). Clamping XBLOCK to the parallel numel in
general: that is what 8d4f21c already measured breaking Mistral's RMSNorm and
DeepSeek, where the same shape needs the wide tile.

GREEN SINCE 2026-09-04. `tl.sort` reshapes its tile into log2(n) axes of extent 2
so a compare-exchange at stride 2^k can NAME its partner, and the reassociation
says which axes are which -- for a [128, 16] tile that is `[[0..6], [7..10]]`,
seven spelling the rows and four the columns. absorb_layout could only take a
reshape whole: forward (readers see [128, 16]) is refused for a real reason, since
merging the column run takes away the ability to name axis 3 alone, which is what
the network is FOR. So backward won, the tile stayed rank 11 to the end,
select_lane_axis had eleven interchangeable axes of 2 to choose between, and the
DMA descriptor -- which holds four dims -- peeled the lane axis into an
`affine.for`, where an outer loop iterates the axis the DMA is meant to spread.

The measurement that unlocked it: `_iteration_takes` refuses the merge 21 times
and EVERY ONE of the 21 is run 1, the columns. Run 0, the rows, is refused zero
times. "The columns cannot merge" was silently deciding "the rows cannot either".
A reshape may now merge ONE of its runs, the tile arrives as [128, 2, 2, 2, 2],
and the DMA folds the four contiguous binary axes back into 16 -- never across
the lane axis, which is the clause that made the same fold wrong on its own.

Its own tile is [128, 64] now -- the 128 being the lane count Inductor gives a
kernel whose parallel extent is 1, so each lane holds its own copy of the row and
the network stays inside a lane, which is exactly what this file said it needed.

Moonlight/Kimi MoE router at 16 experts: 320 spad globals, 104224 bytes/lane against a 65536 budget, and the shrink-the-block retry cannot move it because the waste is on the lane axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64
X = 128        # the lane count, which is what Inductor picks when xnumel is 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sort_single_row_wide_tile_kernel(x_ptr, o_ptr, X: tl.constexpr,
                                         N: tl.constexpr):
        r = tl.arange(0, N)[None, :]
        # ONE row, held X times. This is the shape Inductor emits for xnumel 1:
        # the mask is all-true, the tile is [XBLOCK, R0_BLOCK], and every lane
        # ends up with a copy of the same row.
        v = tl.broadcast_to(tl.load(x_ptr + r), [X, N])
        s = tl.sort(v, dim=1)
        tl.store(o_ptr + tl.broadcast_to(r, [X, N]), s)

    return sort_single_row_wide_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(1, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sort(dim=1).values}


SPEC = KernelSpec(
    name="sort_single_row_wide_tile_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "X": "constexpr", "N": "constexpr"},
    constexprs={"X": X, "N": N},
    args=[Arg("x", IN, "float32", (1, N)), Arg("o", OUT, "float32", (1, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # Each permute's rounds step through the rows digit by digit, so they fold into nested
    # loops: ONE push in the code per permute, where unrolled they were 2048.
    expect_ir={"11-kernel.s": {"sf.vc.ivv": 2}},
    expect="open",
    expect_why='with the middle plan gone a reshape stands, and the reduce below it folds dim 12 of a tile banked on that same axis -- vectorize refuses, because an in-bank fold leaves its answer one element per lane',
)
