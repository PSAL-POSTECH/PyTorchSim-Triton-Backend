"""A constrain to a linear layout of 64 lanes on a machine of 128: the anchor refuses it rather than
searching for or defaulting to another layout. Pins the refusal path every kernel-named layout
shares (torchsim-anchor-layouts' machineLayout); the harness does not match the message, so read it
in stage.log: "the constrain layout ... is not one this machine can take".
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

M, N = 8, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.experimental.gluon.language import DistributedLinearLayout
    from triton.language.extra import torchsim

    HALF: tl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 64]],
        lane_bases=[[0, 1 << i] for i in range(6)],
        warp_bases=[], block_bases=[], shape=[M, N])

    @triton.jit
    def constrain_half_lanes_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        tl.store(o_ptr + offs, torchsim.constrain(tl.load(x_ptr + offs), HALF))

    return constrain_half_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].clone()}


SPEC = KernelSpec(
    name="constrain_half_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("triton-compile", Reason.TRITON_PASS_FAILED),
    expect_why="torchsim-anchor-layouts: the kernel's constrain layout puts 64 lanes in a warp and "
               "this machine has 128; a kernel-named layout is taken as named or refused",
)
