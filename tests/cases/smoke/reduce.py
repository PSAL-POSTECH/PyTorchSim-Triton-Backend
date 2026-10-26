"""Row-wise sum -- the reduction probe.

torch equivalent: x.reshape(M, N).sum(dim=1)

WHY THIS SHAPE.  The scratchpad is lane-banked: the MVIN DMA scatters tile dim 0
across the lanes (element i -> lane i), and a scalar load reads lane 0 only. So a
reduction is only expressible if the axis being reduced lives INSIDE a lane. That
is exactly this kernel: dim 0 (BM = the lane count) is spread across lanes, and
axis 1 is reduced within each lane. The other direction (axis=0) would have to
reduce ACROSS banks, which needs a lane-crossing primitive no pass emits today.

It is also why stage 2 passes `transpose-reduce-to-rank0=false`: with the default
the linalg lowering inserts a linalg.transpose to move every reduced axis to dim 0,
which would move the reduction onto the lanes -- the one place it cannot go.

STATUS: this is a probe, not a passing baseline. vectorize only widens
identity-elementwise generics; linalg.reduce is left alone and falls through to
convert-linalg-to-loops. See the run for where that lands.

Reduction probe: the lane-banked layout only admits a reduction whose axis lives inside a lane. vectorize does not handle linalg.reduce.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 512          # rows total
N = 64           # columns -- the reduced axis, kept inside a lane
BM = 128         # rows per program == PYTORCHSIM_COMPILE_VECTORLANE_SIZE (dim 0 -> lanes)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_kernel(x_ptr, o_ptr, N: tl.constexpr, BM: tl.constexpr,
                      BN: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        tl.store(o_ptr + rows, tl.sum(x, axis=1))

    return reduce_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].reshape(M, N).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"N": N, "BM": BM, "BN": N},
    args=[
        Arg("x", IN, "float32", (M * N,)),
        Arg("o", OUT, "float32", (M,)),
    ],
    grid=(M // BM,),
    reference=reference,
    make_inputs=make_inputs,
    # Summation order differs from torch's pairwise reduction.
    rtol=1e-4, atol=1e-4,
)
