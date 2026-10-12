"""Five effective tile dims, lane axis in the inner four, folded into one descriptor ([0], [1],
[2, 3, 4]) before the TOG, so TOGSim reads the DMA the ELF issues. Pins the timing stage. Breaks
it: folding below the TOG (old tree: 'tile.size: 5'), or typing the fold off the bank strides."""
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
    def five_dim_peel_lane_inner_kernel(x_ptr, o_ptr, D: tl.constexpr,
                                        N: tl.constexpr):
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
        s = tl.sum(x, axis=0)
        off_o = (i1[:, None, None, None] * (D * D * N)
                 + i2[None, :, None, None] * (D * N)
                 + i3[None, None, :, None] * N
                 + n[None, None, None, :])
        tl.store(o_ptr + off_o, s)

    return five_dim_peel_lane_inner_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(D * D * D * D * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(D, D, D, D, N).sum(dim=0).reshape(-1)}


SPEC = KernelSpec(
    name="five_dim_peel_lane_inner_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "D": "constexpr", "N": "constexpr"},
    constexprs={"D": D, "N": N},
    args=[Arg("x", IN, "float32", (D * D * D * D * N,)),
          Arg("o", OUT, "float32", (D * D * D * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, expect_stage="timing",
)
