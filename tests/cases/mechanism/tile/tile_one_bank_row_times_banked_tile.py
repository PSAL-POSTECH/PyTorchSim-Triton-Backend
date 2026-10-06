"""A one-bank row multiplied into a lane-banked tile -- replicated, and computed.

    row  memref<Nxf32>      read `(d0,d1) -> (d1)`   every lane needs all N
    a    memref<MxNxf32>    torchsim.lane_axis = 0       M rows, one per lane
    b    a * row            indexing_maps (d0,d1) / (d1) / (d0,d1)

WHAT THIS PINS. Every lane has to read the whole row, and the row is in a bank
that is not theirs. This machine has no instruction that crosses lanes (rule 8),
so the row has to be PRESENT in every bank -- a replicating transfer, `dram_stride`
0 on the lane axis, the same bytes staged into each. `fit_layout_to_lanes` mints it;
`_stage_one_map_row` ranks the three ways to get the row there and takes the
cheapest that applies (re-read the load wide, rebuild it from `linalg.index`, or
round-trip it through DRAM). This kernel's row is loaded and has another reader,
so it takes the round trip -- one row out and back, which the sweep reports as
`[1 round trip]`.

WHY THIS FILE EXISTS. `fit_layout_to_lanes` recognised the demand only in the
spelling where the row had already been materialised as a `[1, N]` tile and read
through a map with a literal 0 in it. `--linalg-fold-unit-extent-dims` takes that
unit extent off upstream, and the same broadcast arrives as a rank-1 buffer whose
dim is simply OMITTED from the map -- `(d0,d1) -> (d1)`. `_map_broadcast_pins`
asked for `rank_of(v) == rank` and `len(m.results) == rank` and so saw nothing,
the row was never staged, and `vectorize` refused five passes later:

    NotVectorisable: one operand of this elementwise op is in a single bank
    (ONE_LANE) while another is banked across the lanes on iteration dim 1 of
    extent 16; every lane would have to read the one-bank operand out of a bank
    that is not its own. fit_layout_to_lanes has to replicate it into every bank or
    cut the group

A map that OMITS a dim says what a map that PINS it to zero says. That sentence
was already made in `vectorize._projected_permutation` (commit 4c65daa) and
had never been carried to `fit_layout_to_lanes`; making it there is what turns this
file green. 27 other kernels moved with it.

WHAT WOULD BREAK THIS. Narrowing `_map_broadcast_pins` back to the tile-rank
spelling, or dropping the ONE_LANE/banked contradiction check in vectorize
that names this shape -- that check is what stopped it being a silent wrong
answer, which is what it was before: the vlane loop came out with a trip count of
1 and the kernel answered from one sixteenth of its input.

MEASURED, on the two kernels that were carrying this silently before the check --
both `expect="pass"`, both green until the unit extents came off upstream:

    mixed/mixed_turn_through_expand   verify: numerical mismatch 10.3251,
                                      31 of 32 elements, vlane loop `0 to 1`
    dl/cascade_merge_state            verify: numerical mismatch nan

Was expect="refuse" while fit_layout_to_lanes recognised a row broadcast only in the tile-rank spelling. It reads the map for both spellings now and stages the row with a replicating transfer; this file is the one that asked for its marker back.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 16, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def one_bank_row_times_banked_tile(a_ptr, r_ptr, c_ptr,
                                       M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        a = tl.load(a_ptr + rm[:, None] * N + rn[None, :])
        # THE ROW. One value per COLUMN, so every lane wants all N of them --
        # and after the unit extent comes off it is a rank-1 buffer read
        # `(d0,d1)->(d1)` rather than a [1, N] tile broadcast into shape.
        row = tl.load(r_ptr + rn)
        b = a * row[None, :]
        # A REDUCE ON THE OTHER AXIS, so the tile keeps its rows on the lanes
        # and the row cannot simply be banked the same way.
        tl.store(c_ptr + rm, tl.sum(b, axis=1))

    return one_bank_row_times_banked_tile


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * N, generator=g, dtype=torch.float32),
            "r": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    b = inputs["a"].reshape(M, N) * inputs["r"].reshape(1, N)
    return {"c": b.sum(dim=1).contiguous()}


SPEC = KernelSpec(
    name="one_bank_row_times_banked_tile", kernel=kernel,
    signature={"a_ptr": "*fp32", "r_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("a", IN, "float32", (M * N,)),
          Arg("r", IN, "float32", (N,)),
          Arg("c", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
