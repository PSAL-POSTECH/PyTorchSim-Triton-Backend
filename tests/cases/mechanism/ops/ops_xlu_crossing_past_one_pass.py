"""A rebanking transpose whose bank is DEEPER than the lanes it lands on.

    load   x[row][head][col]     row-major, row stride HEADS*DIM
    store  out[head][col][row]   the ROW is the contiguous axis now
    scale  indexed by ROW, which is the axis the store demotes to last

WHAT IT PINS. The unit crosses ONE tile -- `depth` columns become `depth` lanes --
so a move whose bank holds more rows than the destination has lanes is not one
transpose but several. Here the tile is (16, 2, 16) over 16 lanes: 32 rows a bank
against 16 lanes, and HEADS rides the bank at BOTH ends, so the correspondence is
two transposes sharing one buffer. Measured, `mode = "transpose"` in two
passes: `src(0, step 2) -> dst(0, step 1)` and `src(1, step 2) -> dst(16, step 1)`,
16 rows each.

    before   fit_layout_to_lanes priced the crossing arm as "one pass or nothing"
             and took a permute instead -- `mode = "permute"`, so the count
             this file asks for is 0 and the kernel fails in stage 4
    after    two passes of the crossing, ALL STAGES PASSED, max_abs_err 0

WHY A PASS'S ROWS ARE STRIDED, which is the second thing this holds down. A pass
takes ONE row from each destination lane's block, so its rows are `k`, `k + 2`,
`k + 4` ... -- a run at a step, never a contiguous one. That run is not a vector
load: the lowering scalarises a strided view into `flw` + `vfmv.s.f`, and a SCALAR
load carries no lane (spike adds `vu_idx * vu_sram_byte` in the vector load macro
and nowhere else), so every lane pushed lane 0's row. Measured on the tree with
the split and without the lane-relative read: 480/512 elements outside tolerance,
and spike's own `XLU SIMM5 4` block printed the same 16 words for every lane.

WHAT BREAKS IT AGAIN: HEADS = 1, which leaves 16 rows a bank and one pass; reading
a pass through `bank_slice` (a strided view) rather than through each end's
`bank_view` at the pass's own offsets; or scaling by the head instead of the row,
which lets both ends agree on a lane axis and buys no move at all.

torch equivalent: (x * scale).permute(1, 2, 0).contiguous()

A crossing whose bank is deeper than the lanes it lands on: pins the split into one pass per bystander head, and the lane-relative read each pass's strided run needs.
"""
#: RE-PINNED 2026-09-25: NO CROSSING. With fusion after the layouts every op is
#: laid out on its own, and the transpose is absorbed into the store's DMA stride;
#: what moves now is the ROW scale, broadcast across the lanes. The depth-split
#: crossing this pinned is still run, and held to its values, by loop_11 and nine others.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, HEADS, DIM = 16, 2, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_crossing_past_one_pass_kernel(
            x_ptr, scale_ptr, out_ptr,
            ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr):
        y = tl.arange(0, ROWS)[:, None, None]
        head = tl.arange(0, HEADS)[None, :, None]
        col = tl.arange(0, DIM)[None, None, :]

        v = tl.load(x_ptr + ((HEADS * DIM) * y + DIM * head + col))
        s = tl.load(scale_ptr + y)
        tl.store(out_ptr + ((ROWS * DIM) * head + ROWS * col + y), v * s)

    return ops_xlu_crossing_past_one_pass_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, HEADS, DIM, generator=g, dtype=torch.float32),
            "scale": torch.randn(ROWS, 1, 1, generator=g, dtype=torch.float32)}


def reference(inputs):
    x, s = inputs["x"], inputs["scale"]
    return {"out": (x * s).permute(1, 2, 0).contiguous()}


SPEC = KernelSpec(
    name="ops_xlu_crossing_past_one_pass_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "scale_ptr": "*fp32", "out_ptr": "*fp32",
               "ROWS": "constexpr", "HEADS": "constexpr", "DIM": "constexpr"},
    constexprs={"ROWS": ROWS, "HEADS": HEADS, "DIM": DIM},
    args=[Arg("x", IN, "float32", (ROWS, HEADS, DIM)),
          Arg("scale", IN, "float32", (ROWS, 1, 1)),
          Arg("out", OUT, "float32", (HEADS, DIM, ROWS))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE PASS COUNT IS THE CLAIM, and the integer says which pass of the unit:
    # 4 is the bare crossing. A tree that cannot split one prices this as 16.
    expect_ir={"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0, "pytorchsim.distributed_reduction": 0}},
)
