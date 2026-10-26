"""Pins a value the kernel computes, outside loops, as a scalar runtime attribute: each program copies the
row of x at the argmax of its scores, so the run records the argmax itself and the TOG loads that row.
Breaks if the read of the computed index survives into the TOG, or is read as 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, N, D = 4, 64, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_argmax_row_tog(s_ptr, x_ptr, o_ptr, N: tl.constexpr, D: tl.constexpr):
        pid = tl.program_id(0)
        rn = tl.arange(0, N)
        rd = tl.arange(0, D)
        i = tl.argmax(tl.load(s_ptr + pid * N + rn), 0)
        tl.store(o_ptr + pid * D + rd, tl.load(x_ptr + i * D + rd))

    return rtattr_argmax_row_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"s": torch.randn(S, N, generator=g), "x": torch.randn(N, D, generator=g)}


def reference(inputs):
    return {"o": inputs["x"][inputs["s"].argmax(1)]}


SPEC = KernelSpec(
    name="rtattr_argmax_row_tog", kernel=kernel,
    signature={"s_ptr": "*fp32", "x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "D": "constexpr"},
    constexprs={"N": N, "D": D},
    args=[Arg("s", IN, "float32", (S, N)), Arg("x", IN, "float32", (N, D)), Arg("o", OUT, "float32", (S, D))],
    expect_ir={"09-custom.mlir": {"pytorchsim.runtime_attribute": 1}},
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
