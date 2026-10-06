"""Elementwise multiply + a fused scalar bias -- two-op elementwise chain.

torch equivalent: x * y + 1.0

Checks that a chain of arith ops lands in ONE linalg.generic rather than being
split, which is what the vendored fusion-sensitive passes downstream expect.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 2048
BLOCK = 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mul_kernel(x_ptr, y_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        # UNMASKED -- see add.py
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(o_ptr + offs, x * y + 1.0)

    return mul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(N, generator=g, dtype=torch.float32),
        "y": torch.randn(N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    return {"o": inputs["x"] * inputs["y"] + 1.0}


SPEC = KernelSpec(
    name="mul_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "n": "constexpr", "BLOCK": "constexpr"},
    constexprs={"n": N, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("y", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (N,)),
    ],
    grid=(N // BLOCK,),
    reference=reference,
    make_inputs=make_inputs,
)
