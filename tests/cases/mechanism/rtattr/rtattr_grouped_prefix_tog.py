"""Pins a prefix sum the TOG recomputes: group g's rows start at the sum of sizes[0:g], so the run records
sizes as one array and the TOG adds its elements up in C++ (offsets 0, 320, 1344 for sizes 5, 16, 0, 9) and
skips the empty group. Breaks if the running offset is read as its init 0, or sizes[g] as a constant."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

G, BM, D = 4, 16, 64
SIZES = [5, 16, 0, 9]
M = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_grouped_prefix_tog(x_ptr, sz_ptr, o_ptr, G: tl.constexpr, BM: tl.constexpr, D: tl.constexpr):
        rm = tl.arange(0, BM)
        rd = tl.arange(0, D)
        off = 0
        for g in range(0, G):
            m = tl.load(sz_ptr + g)
            x = tl.load(x_ptr + (off + rm)[:, None] * D + rd[None, :], mask=(rm < m)[:, None], other=0.0)
            tl.store(o_ptr + g * D + rd, tl.sum(x, 0))
            off += m

    return rtattr_grouped_prefix_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, D, generator=g), "sz": torch.tensor(SIZES, dtype=torch.int32)}


def reference(inputs):
    import torch
    out, off = [], 0
    for m in SIZES:
        out.append(inputs["x"][off:off + m].sum(0))
        off += m
    return {"o": torch.stack(out)}


SPEC = KernelSpec(
    name="rtattr_grouped_prefix_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "sz_ptr": "*i32", "o_ptr": "*fp32", "G": "constexpr", "BM": "constexpr",
               "D": "constexpr"},
    constexprs={"G": G, "BM": BM, "D": D},
    args=[Arg("x", IN, "float32", (M, D)), Arg("sz", IN, "int32", (G,)), Arg("o", OUT, "float32", (G, D))],
    expect_ir={"09-custom.mlir": {"torchsim.runtime_attribute": 2}},
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
