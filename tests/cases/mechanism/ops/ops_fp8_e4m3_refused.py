"""float8_e4m3fn, refused by decision as float8_e5m2 is (ops_fp8_roundtrip.py). The harness does not
match the message, so read it in the run's output: transform's "8-bit float is refused on this
backend", not Triton's "type fp8e4nv not supported in this architecture"."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_fp8_e4m3_refused(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        tl.store(o_ptr + r, tl.load(x_ptr + r).to(tl.float8e4nv).to(tl.float32))

    return ops_fp8_e4m3_refused


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].to(torch.float8_e4m3fn).to(torch.float32)}


SPEC = KernelSpec(
    name="ops_fp8_e4m3_refused", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_why="8-bit float is not on this backend's path, and that is a decision rather than a "
               "limitation waiting to lift (ops_fp8_roundtrip.py says why)",
)
