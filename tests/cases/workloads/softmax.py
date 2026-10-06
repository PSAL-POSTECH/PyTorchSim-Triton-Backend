"""Row-wise softmax -- the first complex op: two reductions and a broadcast back.

torch equivalent: torch.softmax(x.reshape(M, N), dim=1)

WHY THIS IS THE NEXT RUNG AFTER reduce.py.  `reduce.py` reduces and stores. This
one reduces, then feeds the SCALAR IT PRODUCED BACK INTO THE TILE -- twice, on
either side of an exp:

    m = max(x, axis=1)      # rank-2 -> rank-1
    e = exp(x - m[:, None]) # rank-1 -> rank-2, broadcast along the reduced axis
    s = sum(e, axis=1)
    o = e / s[:, None]

The reduce direction was already covered; the BROADCAST direction is what is new.
It is also the direction the lane banking makes non-obvious: `m` is one value per
row, rows are the lane axis, so the broadcast is per-lane replication along the
in-lane axis -- no lane crossing, which is the only reason this is expressible at
all (see reduce.py for why the other axis is not).

Numerically this also parks a subtraction between two reductions, so the whole
chain has to keep the same tile in the scratchpad across four linalg ops without
a DRAM round trip.

SHAPE.  Rows are the lane axis, so BM = 128 = TORCHSIM_COMPILE_VECTORLANE_SIZE. The reduced
axis N lives inside a lane and is a full tile, i.e. this is the "one pass"
softmax, not the online/streaming one -- N is small enough to hold.

Complex-op tier: reduce THEN broadcast back along the reduced axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 256          # rows total
N = 64           # the reduced axis, kept inside a lane
BM = 128         # rows per program == TORCHSIM_COMPILE_VECTORLANE_SIZE


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(x_ptr, o_ptr, N: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        idx = rows[:, None] * N + cols[None, :]

        x = tl.load(x_ptr + idx)
        m = tl.max(x, axis=1)
        e = tl.exp(x - m[:, None])
        s = tl.sum(e, axis=1)
        tl.store(o_ptr + idx, e / s[:, None])

    return softmax_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.softmax(inputs["x"].reshape(M, N), dim=1).reshape(-1)}


SPEC = KernelSpec(
    name="softmax_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"N": N, "BM": BM, "BN": N},
    args=[
        Arg("x", IN, "float32", (M * N,)),
        Arg("o", OUT, "float32", (M * N,)),
    ],
    grid=(M // BM,),
    reference=reference,
    make_inputs=make_inputs,
    # exp + two reductions in a different order than torch's.
    rtol=1e-5, atol=1e-6,
)
