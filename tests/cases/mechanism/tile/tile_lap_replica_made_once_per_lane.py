"""A GEMV whose vector is made by a chain (silu) and read across rows that lap the lanes: the vector
is made once in each lane, at [lanes, R], and every lap reads it -- not made again at each of the
four laps of [512, R]. Breaks if p02 builds the replica in the reader's full shape again, or if
fuse re-fuses its chain into the reader."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, R = 512, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_lap_replica_made_once_per_lane_kernel(x_ptr, w_ptr, o_ptr, M: tl.constexpr, R: tl.constexpr):
        m = tl.arange(0, M)[:, None]
        r = tl.arange(0, R)[None, :]
        x = tl.load(x_ptr + r)
        s = x / (1.0 + tl.exp(-x))
        w = tl.load(w_ptr + m + M * r)
        tl.store(o_ptr + tl.arange(0, M), tl.sum(s * w, 1))

    return tile_lap_replica_made_once_per_lane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, generator=g), "w": torch.randn(M * R, generator=g)}


def reference(inputs):
    import torch
    x = inputs["x"]
    s = x / (1.0 + torch.exp(-x))
    return {"o": inputs["w"].view(R, M).t() @ s}


SPEC = KernelSpec(
    name="tile_lap_replica_made_once_per_lane_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "R": "constexpr"},
    constexprs={"M": M, "R": R},
    args=[Arg("x", IN, "float32", (R,)), Arg("w", IN, "float32", (M * R,)), Arg("o", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-passes/05-torchsim-fuse.mlir": {"linalg.generic": 2, "linalg.reduce": 0, "math.exp": 1}},
)
