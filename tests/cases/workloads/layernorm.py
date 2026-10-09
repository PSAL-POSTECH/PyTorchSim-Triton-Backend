"""LayerNorm with a residual add and a learned affine -- both broadcast
directions in one kernel, plus a reduction whose input is not the loaded tile.

torch equivalent:

    h = x + res
    y = (h - h.mean(-1, keepdim=True)) * rsqrt(h.var(-1, unbiased=False) + eps)
    o = y * w + b

WHY THIS ONE AND NOT JUST softmax AGAIN.  softmax broadcasts a per-ROW scalar
back along the in-lane axis. This kernel does that AND the other direction:

    mean, rstd   one value per row     -> replicated ALONG a lane   (as softmax)
    w, b         one value per column  -> replicated ACROSS lanes

The second is the one the banking makes interesting. `w` is a rank-1 tile of N
elements and every lane needs the whole of it, so it cannot be split across the
lanes the way a row tile is -- it has to land in each lane's bank identically.
That is a different transfer, decided by the same lane-axis machinery, and a
kernel that mixes the two forces the two decisions to be consistent inside one
scratchpad.

The second reduction is also not a reduction OF THE LOADED TILE: `sum((h-mean)^2)`
reduces a value produced by an elementwise chain that itself reads the first
reduction's result. So the chain is

    load -> add -> reduce -> broadcast -> sub -> mul -> reduce -> broadcast -> ...

with two reductions on the same axis separated by a broadcast, which is the
deepest tensor chain in the suite that still has to stay in the scratchpad end to
end. Any pass that materialises an intermediate to DRAM shows up here as a second
transfer pair.

WHAT PROVES IT FIRED
  08-transformed.mlir has exactly two `torchsim.transfer` reads for x/res, one each for
  w/b, and one write -- no round trip in the middle -- and the w/b transfers
  carry a lane split that replicates rather than divides.

Complex-op tier: per-row and per-column broadcasts in one kernel, two chained reductions, five buffers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 256          # rows total
N = 64           # the normalised axis, kept inside a lane
BM = 128         # rows per program == TORCHSIM_COMPILE_VECTORLANE_SIZE
EPS = 1e-5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def layernorm_kernel(x_ptr, res_ptr, w_ptr, b_ptr, o_ptr,
                         N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         EPS: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        idx = rows[:, None] * N + cols[None, :]

        h = tl.load(x_ptr + idx) + tl.load(res_ptr + idx)

        mean = tl.sum(h, axis=1) / N
        d = h - mean[:, None]
        var = tl.sum(d * d, axis=1) / N
        y = d * tl.rsqrt(var[:, None] + EPS)

        w = tl.load(w_ptr + cols)
        b = tl.load(b_ptr + cols)
        tl.store(o_ptr + idx, y * w[None, :] + b[None, :])

    return layernorm_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(M * N, generator=g, dtype=torch.float32),
        "res": torch.randn(M * N, generator=g, dtype=torch.float32),
        "w": torch.randn(N, generator=g, dtype=torch.float32),
        "b": torch.randn(N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    h = (inputs["x"] + inputs["res"]).reshape(M, N)
    d = h - h.mean(dim=1, keepdim=True)
    y = d * torch.rsqrt(h.var(dim=1, unbiased=False, keepdim=True) + EPS)
    return {"o": (y * inputs["w"] + inputs["b"]).reshape(-1)}


SPEC = KernelSpec(
    name="layernorm_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "res_ptr": "*fp32", "w_ptr": "*fp32",
               "b_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "EPS": "constexpr"},
    constexprs={"N": N, "BM": BM, "BN": N, "EPS": EPS},
    args=[
        Arg("x", IN, "float32", (M * N,)),
        Arg("res", IN, "float32", (M * N,)),
        Arg("w", IN, "float32", (N,)),
        Arg("b", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (M * N,)),
    ],
    grid=(M // BM,),
    reference=reference,
    make_inputs=make_inputs,
    # Two reductions and an rsqrt, none of them in torch's order.
    rtol=1e-4, atol=1e-4,
)
