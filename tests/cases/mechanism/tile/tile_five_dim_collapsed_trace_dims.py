"""A rank-5 tile contiguous enough to fold into four descriptor dims: the TOG must name the folded
DMA TOGSim reads, not the tile's five dims. Pins the timing stage. Breaks it: folding after the IR
the TOG is built from (old tree: TOGSim refused 'tile.size: 5')."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT  # noqa: E402

D, N = 2, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def five_dim_collapsed_trace_dims_kernel(x_ptr, o_ptr, D: tl.constexpr, N: tl.constexpr):
        i0 = tl.arange(0, D)
        i1 = tl.arange(0, D)
        i2 = tl.arange(0, D)
        i3 = tl.arange(0, D)
        n = tl.arange(0, N)
        off = (i0[:, None, None, None, None] * (D * D * D * N)
               + i1[None, :, None, None, None] * (D * D * N)
               + i2[None, None, :, None, None] * (D * N)
               + i3[None, None, None, :, None] * N
               + n[None, None, None, None, :])
        x = tl.load(x_ptr + off)
        tl.store(o_ptr + off, x * 2.0)

    return five_dim_collapsed_trace_dims_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(D ** 4 * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] * 2.0}


SPEC = KernelSpec(
    name="five_dim_collapsed_trace_dims_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "D": "constexpr", "N": "constexpr"},
    constexprs={"D": D, "N": N},
    args=[Arg("x", IN, "float32", (D ** 4 * N,)), Arg("o", OUT, "float32", (D ** 4 * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_stage="timing",
)
