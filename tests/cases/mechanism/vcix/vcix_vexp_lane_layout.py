"""A kernel writes an encoding -- vcix("v.iv", opcode=3, imm=0) -- that its author knows is the tpu
model's vexp; the compiler does not: it enforces the layout (lanes on axis 1 of an 8x256 tile) and
passes the encoding through. Pins: one torchsim.vcix in 01-ttir with that #ttg.blocked layout; 04-anchored
pins its operand and result; 09-custom issues one vcix.v.iv opcode 3 imm 0 over a lane's 8 rows, no XLU
crossing. Without the builtin, the plugin op or p10's row path this does not compile. torch: exp(x).
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 256


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import torchsim

    @triton.jit
    def vcix_vexp_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + offs)
        y = torchsim.vcix("v.iv", opcode=3, imm=0, args=[x], layout=torchsim.lane_layout(1))
        tl.store(o_ptr + offs, y)

    return vcix_vexp_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.exp(inputs["x"])}


SPEC = KernelSpec(
    name="vcix_vexp_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-5,
    expect_ir={"01-ttir.mlir": {"torchsim.vcix": 1, "threadsPerWarp = [1, 256]": 1},
               "04-anchored.mlir": {"torchsim.vcix": 1, "torchsim.pin": 2},
               "08-transformed.mlir": {"torchsim.crosslane_move": 0, "dram_arg = -1": 0},
               "10-lowered.mlir": {"vcix.v.iv": 1}},
)
