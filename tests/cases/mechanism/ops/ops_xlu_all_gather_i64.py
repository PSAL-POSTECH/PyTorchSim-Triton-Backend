"""An i64 index row that every lane needs whole, and LONGER than one register.

WHAT IT PINS, AND IT IS TWO THINGS. An all-gather is a transpose into lane 0
followed by a broadcast back out, and the queues between carry 32 bits. b39fb2a
taught it to send a wide element as its two halves -- both ends put values in
LANE ORDER, so the same permutation reaches both -- but only for a row that fits
ONE register: `per != chunk` returned None and the move fell back to a trip it
could not take. Here the row is 32 wide and a 512-bit register holds 16 of what
crosses, so it is two chunks.

    before   torchsim-lower-to-vcix: "'all-gather' cannot be lowered here (the
             shapes are not the unit's)" -- (memref<32xi64>, memref<256x32xi64>)
    after    two chunks, two crossings each, ALL STAGES PASSED, max_abs_err 0

WHERE THE i64 COMES FROM, AND WHY IT IS NOT AVOIDABLE. `xindex` compared against
a `tl.int64` constant is a tensor of the index itself, and inductor writes that
comparison for every `tl.where` it splits a `cat` into. It is not a value the
kernel loaded, so it has no descriptor a transfer could re-read with a zero
stride -- and it rides its own axis for the LOAD masks while every lane needs it
whole for the tile, so it stays one element per lane and the trip is a gather
rather than a broadcast (`_xlu_broadcast_or_gather` decides on exactly that).

SHRUNK FROM Llama3's RoPE, which is where the whole width rule was found: its
index row is i64 and every token after the first was wrong. That kernel's row is
16 wide -- exactly one register -- which is why the one-chunk case was the one
that got built. The shape here keeps the two masked halves that make the row
real and drops the cos/sin, the bmm and the second input.

torch equivalent: see reference -- a rotate-half with the two arms masked.

A 32-wide i64 index row every lane needs: pins the all-gather carrying a wide element as halves across MORE THAN ONE register -- chosen above bufferize as the unit's all-gather, not a trip.
"""

#: tpuv6e: three moves, no pieces -- at the measured move latency 123 the move costs 188 against the pieces' 189
#: (TOGSim 2,735 against 2,542, #291); tpuv3 states no move latency, so the reduce's 137 keeps the pieces.
#: The row is an iota (pytorchsim-materialize-layouts remakes it); ..._loaded_row pins the two-half gather.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Y, X, NX = 256, 32, 8192


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_all_gather_i64_kernel(in_ptr0, out_ptr0, Y: tl.constexpr,
                                      X: tl.constexpr):
        yindex = tl.arange(0, Y)[:, None]
        xindex = tl.arange(0, X)[None, :]
        y0 = yindex % 32
        y2 = yindex // 128
        v = tl.load(in_ptr0 + (xindex + 32 * y2 + 256 * y0))
        half = tl.full([1, 1], 16, tl.int64)
        lo = xindex < half
        hi = xindex >= half
        a = tl.load(in_ptr0 + (16 + 32 * y2 + 256 * y0 + xindex), lo, other=0.0)
        b = tl.load(in_ptr0 + (32 * y2 + 256 * y0 + (-16) + xindex), hi,
                    other=0.0)
        tl.store(out_ptr0 + (xindex + X * yindex), tl.where(lo, -a, b) + v)

    return ops_xlu_all_gather_i64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in_ptr0": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    src = inputs["in_ptr0"]
    y = torch.arange(Y)[:, None]
    x = torch.arange(X)[None, :]
    base = 32 * (y // 128) + 256 * (y % 32)
    v = src[x + base]
    zero = torch.zeros(1)
    a = torch.where(x < 16, src[(16 + base + x).clamp(max=NX - 1)], zero)
    b = torch.where(x >= 16, src[(base + x - 16).clamp(min=0)], zero)
    return {"out_ptr0": torch.where(x < 16, -a, b) + v}


SPEC = KernelSpec(
    name="ops_xlu_all_gather_i64_kernel", kernel=kernel,
    signature={"in_ptr0": "*fp32", "out_ptr0": "*fp32", "Y": "constexpr",
               "X": "constexpr"},
    constexprs={"Y": Y, "X": X},
    args=[Arg("in_ptr0", IN, "float32", (NX,)),
          Arg("out_ptr0", OUT, "float32", (Y, X))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE GATHER, AND THE ROW IS WIDE. Drop either masked arm and the index row
    # folds into the predicate, the move disappears and nothing is pinned.
    expect_ir={"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 3},
               "09-custom.mlir": {"cut": 0, "piece_box": 0}},
    expect_ir_on={"tpuv3": {"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0,
                                                    "pytorchsim.distributed_reduction": 0},
                            "09-custom.mlir": {"cut": 0, "piece_box": 24}}},
    expect_stage="togcheck",
)
