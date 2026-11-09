"""fp8 arithmetic is refused: addf and mulf ON f8E5M2 vectors. The vector unit converts fp8, it does not
compute on it. Breaks if fp8_to_zvfp8 lets an fp8 arith op through to the i8 retype, which would then add
the bits as integers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_fp8_vector_arith_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r).to(tl.float8e5)
        y = tl.load(y_ptr + r).to(tl.float8e5)
        s = (x + y) * x                 # arith.addf + arith.mulf on f8E5M2
        tl.store(o_ptr + r, s.to(tl.float32))

    return ops_fp8_vector_arith_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    d = torch.float8_e5m2
    a = inputs["x"].to(d).float()
    b = inputs["y"].to(d).float()
    return {"o": ((a + b).to(d).float() * a).to(d).float()}


SPEC = KernelSpec(
    name="ops_fp8_vector_arith_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("lower", Reason.DTYPE_NOT_RUN),
    expect_why="fp8 is storage, convert and matrix-unit input; arithmetic on it is not run",
)
