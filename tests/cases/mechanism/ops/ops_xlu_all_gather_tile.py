"""A COMPUTED 2-D tile that every lane needs whole -- the gather's deep half.

WHAT IT PINS. `torchsim.crosslane = "all-gather"` is a transpose into lane 0 and a
broadcast back out, and it used to be sized off a ROW: lower_to_vcix asked for the lane-fulls
of the source's lane axis and refused anything holding more than one value per lane.

    before   torchsim-lower-to-vcix: "4 values per lane to gather, not the 1 a row of
             8 over 256 lanes has" -- (memref<4x8xi32>, memref<4x8x16xi32>)
    after    4 passes, one per value a lane holds, ALL STAGES PASSED, max_abs_err 0

A PASS CARRIES ONE VALUE PER LANE, so a source holding `d` of them is `d` passes and
every lane ends with all 32. `lib_lanes.gather_rounds` states which element lands
where rather than counting elements per pass, which is the same correction
`755bad3` made for the transpose.

WHY THE TILE IS COMPUTED AND NOT LOADED. A LOADED tile costs nothing to replicate:
the transfer re-reads the same DRAM with a zero stride on the lane axis and the
operand never reaches the unit. What `tl.exp` returns has no descriptor of its own,
which is the same reason Mamba-2's `dt` is carried in selective_state_update. Drop
the `tl.exp` and this kernel passes on either tree, pinning nothing.

WHICH AXIS RIDES, AND WHY IT MATTERS HERE. The lanes ride axis 1, so a pass carries
one of the 8 columns and the tile is 4 deep. Banked on axis 0 instead, a pass
delivers the TRANSPOSE of what the destination spells -- `gather_rounds` answers
None, the operand stays home, and `block_scale_2d_matmul` still records that.

torch equivalent: x.reshape(M, N, S) + exp(c.reshape(M, N))[:, :, None]

A computed 4x8 tile every lane needs whole: with the layout decided above bufferize the tile is made where it is read, and NO move is left to carry -- neither the unit nor a trip.
"""

#: RE-PINNED 2026-09-23 TO THE GRAPH ROUTE. The numbers were right all
#: along -- this kernel passes with the pins ignored -- and what the old
#: pin held was a PATH the route no longer takes: the XLU op is 4 (a crossing) where it used to be 5 (a gather); the count is the same.
#: The declaration above is what it does today, so a change to it is seen;
#: the line above says which way the old path was better, where it was.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, S = 4, 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_all_gather_tile_kernel(c_ptr, x_ptr, o_ptr, M: tl.constexpr,
                                       N: tl.constexpr, S: tl.constexpr):
        m = tl.arange(0, M)[:, None]
        n = tl.arange(0, N)[None, :]
        w = tl.exp(tl.load(c_ptr + m * N + n))
        i = tl.arange(0, S)
        idx = (m * N + n)[:, :, None] * S + i[None, None, :]
        x = tl.load(x_ptr + idx)
        tl.store(o_ptr + idx, x + w[:, :, None])

    return ops_xlu_all_gather_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"c": torch.randn(M * N, generator=g, dtype=torch.float32),
            "x": torch.randn(M * N * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    w = inputs["c"].reshape(M, N).exp()
    return {"o": inputs["x"].reshape(M, N, S) + w[:, :, None]}


SPEC = KernelSpec(
    name="ops_xlu_all_gather_tile_kernel", kernel=kernel,
    signature={"c_ptr": "*fp32", "x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "S": "constexpr"},
    constexprs={"M": M, "N": N, "S": S},
    args=[Arg("c", IN, "float32", (M * N,)),
          Arg("x", IN, "float32", (M * N * S,)),
          Arg("o", OUT, "float32", (M, N, S))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE MOVE IS THE CLAIM -- one gathering trip for a tile four deep, where a
    # loaded tile would be widened in its transfer and nothing would cross.
    expect_ir={"04-transformed.mlir": {"torchsim.crosslane": 0,
                                   "dram_arg = -1": 0}},
)
