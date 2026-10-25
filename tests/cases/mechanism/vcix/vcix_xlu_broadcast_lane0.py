"""A kernel drives the cross-lane unit by its encodings: xlu_push (iv 3, imm 1) of an 8x128 f32 tile,
lanes on axis 1, then xlu_pop (v.i 1) -- replicate lane 0's row. Pins one vcix.iv and one vcix.v.i
of a lane's 8 rows (one zvl256b register) and no compiler crossing. torch: x[:, :1].expand(8, 128)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 128


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
    expect_ir={"01-ttir.mlir": {"pytorchsim_extern.vcix": 2},
               "08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0, "pytorchsim.distributed_reduction": 0,
                                       "pytorchsim_extern.vcix": 2},
               "10-lowered.mlir": {"vcix.iv": 1, "vcix.v.i": 1, "pytorchsim_tpu.xlu_move": 0,
                                   "pytorchsim.distributed_reduction": 0}},
)
