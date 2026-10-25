"""A kernel writes vexp by its encoding (v.iv 3, imm 0); the compiler enforces the layout it names,
lanes on axis 1 of an 8x128 tile, and passes the encoding through. Pins the #ttg.blocked layout in
01-ttir, its two pins in 04-anchored and one vcix.v.iv in 10-lowered, no crossing. torch: exp(x)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 128


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
    expect_ir={"01-ttir.mlir": {"pytorchsim_extern.vcix": 1, "threadsPerWarp = [1, 128]": 1},
               "04-anchored.mlir": {"pytorchsim_extern.vcix": 1, "pytorchsim.pin": 2},
               "08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0, "pytorchsim.distributed_reduction": 0},
               "10-lowered.mlir": {"vcix.v.iv": 1}},
)
