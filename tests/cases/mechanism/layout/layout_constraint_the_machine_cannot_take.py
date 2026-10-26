"""A layout_constraint of 64 threads in a warp on a machine of 128 lanes: the anchor pass refuses it rather
than searching for or defaulting to another layout ("the pytorchsim_extern.layout_constraint layout ... is
not one this machine can take" in stage.log; the harness does not match the message)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

M, N = 8, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.experimental.gluon.language._layouts import BlockedLayout
    from triton.language.extra import pytorchsim

    HALF: tl.constexpr = BlockedLayout(size_per_thread=[1, 2], threads_per_warp=[1, 64],
                                       warps_per_cta=[1, 1], order=[1, 0])

    @triton.jit
    def layout_constraint_half_lanes_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        tl.store(o_ptr + offs, pytorchsim.layout_constraint(tl.load(x_ptr + offs), HALF))

    return layout_constraint_half_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"]}


SPEC = KernelSpec(
    name="layout_constraint_half_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("triton-compile", Reason.TRITON_PASS_FAILED),
    expect_why="pytorchsim-anchor-layouts: the kernel's layout puts 64 threads in a warp and this machine "
               "has 128 lanes; the layout rule is a fixed convention, refuse otherwise",
)
