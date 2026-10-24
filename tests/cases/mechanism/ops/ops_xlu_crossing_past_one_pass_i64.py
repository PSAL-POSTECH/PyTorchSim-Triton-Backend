"""A crossing DEEPER THAN ONE PASS whose element is WIDER THAN THE QUEUES.

    load   x[row][head][col]     row-major, row stride HEADS*DIM
    store  out[head][col][row]   the ROW is the contiguous axis now
    weight indexed by ROW, which is the axis the store demotes to last

WHAT IT PINS. Two slicings of one move meet here. The bank is deeper than the
lanes it lands on, so the crossing is several PASSES -- (16, 2, 16) over 16 lanes
is 32 rows a bank against 16, HEADS riding the bank at both ends -- and the
element is 64 bits while the unit's queues carry 32, so each pass is two
CROSSINGS: the low halves, then the high halves, the SAME permutation over the
same tile both times, with the first round's answers parked in the destination
because the unit holds one tile.

WHAT THE SUITE ALREADY HAD, measured with a print in the arm across all 173
kernels that reach XLU op 4: ptr_gather_mask_wrap_grid and ptr_wrap_gather_rank3
do split a 64-bit crossing, but both ends of theirs are CONTIGUOUS and their
`wide` equals the register, so every 64-bit pass in the suite was `s_step=1
d_step=1` with one destination register. This is the first with a strided end
and the first where a crossing's answers are more than one register.

    measured  4 `XLU SIMM5 4` blocks, two per pass. The first of a pair carries
              the low words and the second the high ones -- at x in +-2^40 the
              second block's rows read 0xffffffxx / 0x000000xx -- and the two
              output tiles are the same permutation of their own input.
              64 pushes and 8 pops, against 32 and 2 for the fp32 twin.

BOTH ARMS OF `_get_rows_per_vector`, which is why the width is not a free
parameter here. A pass's source rows are k, k+2, k+4 ... -- a run at a step, so
the source goes ONE row per vector op, a strided read being no vector op at all.
Its destination rows are contiguous, so they go a REGISTER at a time; and at 64
bits a register holds 8 rows, not 16, so one crossing's answers are TWO registers
where fp32 takes one. That second register is what reads its own parked half.

WHY THE WEIGHT IS PER ROW. `* (row + 1)` makes the answer depend on WHICH element
landed WHERE: the tile is the same multiset under any permutation, and a lane
that pushed its neighbour's row would still be green without it.

WHAT BREAKS IT AGAIN, measured by breaking each on this tree:

    the parked merge   `_build_element_from_words` returning the high half
                       instead of OR-ing it over the destination: 506/512 wrong
    the parked address the split's read-back at the pass's first row rather than
                       at the register's own row -- invisible everywhere else,
                       where a crossing is one register: 256/512 wrong
    i32 for i64        one crossing, and this file's counts are halved
    HEADS = 1          16 rows a bank and one pass, so no split at all

torch equivalent: (x * weight).permute(1, 2, 0).contiguous()

A crossing deeper than one pass on a 64-bit element: two passes of two crossings each, the low halves parked in the destination for the high ones to merge over, one register per strided source row.
"""

#: RE-PINNED 2026-09-23 TO THE GRAPH ROUTE. The numbers were right all
#: along -- this kernel passes with the pins ignored -- and what the old
#: pin held was a PATH the route no longer takes: CHEAPER NOW: neither a pass nor a trip, so the vcix counts this used to pin have no subject.
#: The declaration above is what it does today, so a change to it is seen;
#: the line above says which way the old path was better, where it was.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, HEADS, DIM = 16, 2, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_crossing_past_one_pass_i64_kernel(
            x_ptr, out_ptr,
            ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr):
        y = tl.arange(0, ROWS)[:, None, None]
        head = tl.arange(0, HEADS)[None, :, None]
        col = tl.arange(0, DIM)[None, None, :]

        v = tl.load(x_ptr + ((HEADS * DIM) * y + DIM * head + col))
        tl.store(out_ptr + ((ROWS * DIM) * head + ROWS * col + y),
                 v * (y + 1).to(tl.int64))

    return ops_xlu_crossing_past_one_pass_i64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-(1 << 40), 1 << 40, (ROWS, HEADS, DIM),
                               generator=g, dtype=torch.int64)}


def reference(inputs):
    import torch
    x = inputs["x"]
    w = (torch.arange(ROWS, dtype=torch.int64) + 1)[:, None, None]
    return {"out": (x * w).permute(1, 2, 0).contiguous()}


SPEC = KernelSpec(
    name="ops_xlu_crossing_past_one_pass_i64_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "out_ptr": "*i64",
               "ROWS": "constexpr", "HEADS": "constexpr", "DIM": "constexpr"},
    constexprs={"ROWS": ROWS, "HEADS": HEADS, "DIM": DIM},
    args=[Arg("x", IN, "int64", (ROWS, HEADS, DIM)),
          Arg("out", OUT, "int64", (HEADS, DIM, ROWS))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"pytorchsim.crosslane = 4 : i64": 0}},
)
