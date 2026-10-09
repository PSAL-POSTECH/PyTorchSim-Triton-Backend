"""vcix_xlu_broadcast_lane0's push and pop on a tile deeper than one register: x[64, 256] f32, lanes
on axis 1, is 64 rows per lane and vlen 512 holds 16, so 09-custom has one vcix.iv and one vcix.v.i, each
in a 4-trip loop at vl 16. Pins that the unit's depth accumulates across pushes and the pops return
the rows in order: a replicate over one register's rows would get every row past the 16th wrong.
torch: x[:, :1].expand(64, 256).
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 64, 256


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
