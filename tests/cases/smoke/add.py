"""Elementwise add -- the smallest kernel that exercises the whole pipeline.

torch equivalent: x + y

No loop in the kernel body -- does NOT exercise the scf.for -> affine.for gap. Use gemm.py for that.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128


def kernel():
    """Lazy: no triton import at module scope. See pytorchsim_triton_compiler/contract/kernel_spec.py."""
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(o_ptr + offs, x + y)

    return add_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(N, generator=g, dtype=torch.float32),
        "y": torch.randn(N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    return {"o": inputs["x"] + inputs["y"]}


SPEC = KernelSpec(
    name="add_kernel",
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
