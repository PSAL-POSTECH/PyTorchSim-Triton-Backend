"""vcix_xlu_broadcast_lane0's push and pop on x[64, 128] f32: 64 rows a lane, 8 a register, so one
vcix.iv and one vcix.v.i, each in an 8-trip loop. Breaks if the unit's depth does not accumulate
across pushes or the pops come back out of order. torch: x[:, :1].expand(64, 128)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 64, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import torchsim

    @triton.jit
    def vcix_xlu_broadcast_big_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + offs)
        torchsim.vcix("iv", opcode=3, imm=1, args=[x], layout=torchsim.lane_layout(1))
        y = torchsim.vcix("v.i", opcode=1, imm=1, args=[], layout=torchsim.lane_layout(1),
                          out=(M, N), dtype=tl.float32)
        tl.store(o_ptr + offs, y)

    return vcix_xlu_broadcast_big_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"][:, :1].expand(M, N).contiguous()}


SPEC = KernelSpec(
    name="vcix_xlu_broadcast_big_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"01-ttir.mlir": {"torchsim.vcix": 2},
               "08-transformed.mlir": {"torchsim.crosslane_move": 0, "torchsim.vcix": 2},
               "10-lowered.mlir": {"vcix.iv": 1, "vcix.v.i": 1, "torchsim.crosslane_move": 0}},
)
