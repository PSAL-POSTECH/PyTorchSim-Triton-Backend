"""A vcix layout of 128 threads in a warp on a machine of 256 lanes: the anchor pass refuses it
rather than searching for or defaulting to another layout. Pins the refusal path of
torchsim-anchor-layouts' anchorVcix; the harness does not match the message, so read it in
stage.log: "the vcix layout ... is not one this machine can take".
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 256


def kernel():
    import triton
    import triton.language as tl
    from triton.experimental.gluon.language._layouts import BlockedLayout
    from triton.language.extra import torchsim

    HALF: tl.constexpr = BlockedLayout(size_per_thread=[1, 2], threads_per_warp=[1, 128],
                                       warps_per_cta=[1, 1], order=[1, 0])

    @triton.jit
    def vcix_half_lanes_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        y = torchsim.vcix("vexp", [tl.load(x_ptr + offs)], layout=HALF)
        tl.store(o_ptr + offs, y)

    return vcix_half_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.exp(inputs["x"])}


SPEC = KernelSpec(
    name="vcix_half_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_why="torchsim-anchor-layouts: the kernel's vcix layout puts 128 threads in a warp and "
               "this machine has 256 lanes; the layout rule is a fixed convention, refuse otherwise",
)
