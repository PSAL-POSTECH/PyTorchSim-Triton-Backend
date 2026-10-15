"""A row sum narrowed to bf16 as one scalar: the vector path converts with vfncvtbf16, but a scalar f32 -> bf16
lowers to the libcall __truncsfbf2, which the RISC-V link has no definition of (binary stage, measured on 6ebafec).
validation/vector stores its max/sum/slice scalars as f32 until this passes; lifting it should drop `expect`."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scalar_bf16_store_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + idx).to(tl.float32)
        tl.store(o_ptr, tl.sum(tl.sum(x, axis=1), axis=0).to(tl.bfloat16))

    return ops_scalar_bf16_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    return {"o": inputs["x"].float().sum().reshape(1).to(inputs["x"].dtype)}


SPEC = KernelSpec(
    name="ops_scalar_bf16_store_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*bf16", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "bfloat16", (M * N,)), Arg("o", OUT, "bfloat16", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect="blocked",
    expect_why="a scalar f32 -> bf16 truncation lowers to __truncsfbf2, which the RISC-V link does not define",
)
