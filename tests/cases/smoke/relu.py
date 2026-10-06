"""Elementwise ReLU -- unary op with a select/max, single input.

torch equivalent: torch.relu(x)

Useful because the comparison lowers to arith.maximumf (or a cmpf/select pair
depending on triton version) -- a good probe for whether your passes handle
non-arithmetic elementwise ops.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def relu_kernel(x_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(o_ptr + offs, tl.maximum(x, 0.0))

    return relu_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.relu(inputs["x"])}


SPEC = KernelSpec(
    name="relu_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"n": N, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (N,)),
    ],
    grid=(N // BLOCK,),
    reference=reference,
    make_inputs=make_inputs,
)
