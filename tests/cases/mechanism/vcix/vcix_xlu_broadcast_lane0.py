"""A kernel drives the cross-lane unit itself: vcix("iv", opcode=3, imm=1) pushes an 8x256 tile, lanes
on axis 1, and vcix("v.i", opcode=1, imm=1) pops it back. Its author knows that encoding is the tpu
model's xlu_push and xlu_pop with operation 1 (replicate after: every lane reads lane 0's row); the
compiler does not -- it enforces the layout and passes the encoding through. Pins: 01 has the two
torchsim.vcix; 05 issues one vcix.iv and one vcix.v.i of 8 rows each (vlen 512 holds a lane's 8
f32 rows in one register, as p10 reads them) and no compiler crossing. Without the stateful forms
and p10's lowering of them this does not compile. torch: x[:, :1].expand(8, 256).
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 256


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import torchsim

    @triton.jit
    def vcix_xlu_broadcast_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + offs)
        torchsim.vcix("iv", opcode=3, imm=1, args=[x], layout=torchsim.lane_layout(1))
        y = torchsim.vcix("v.i", opcode=1, imm=1, args=[], layout=torchsim.lane_layout(1),
                          out=(M, N), dtype=tl.float32)
        tl.store(o_ptr + offs, y)

    return vcix_xlu_broadcast_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"][:, :1].expand(M, N).contiguous()}


SPEC = KernelSpec(
    name="vcix_xlu_broadcast_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"01-ttir.mlir": {"torchsim.vcix": 2},
               "04-transformed.mlir": {"torchsim.crosslane_move": 0, "torchsim.vcix": 2},
               "05-lowered.mlir": {"vcix.iv": 1, "vcix.v.i": 1, "torchsim.crosslane_move": 0}},
)
