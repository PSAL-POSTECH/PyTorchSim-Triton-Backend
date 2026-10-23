"""Pins a branch that yields a scalar as a TOG if/else with no result: program 0 loads x[0] in its then
arm to make 2.0, program 1 takes 3.0, and the skeleton drops the results nothing it states reads. Breaks
if the arms' yields keep their reads alive into the TOG ("a scalar DRAM read reaches the TOG unmarked")."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, G = 8, 128, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_if_yields_a_scalar_tog(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * R * C + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + offs)
        if pid == 0:
            s = tl.load(x_ptr) * 0.0 + 2.0
        else:
            s = 3.0
        y = x * s
        tl.store(y_ptr + offs, y)

    return loop_if_yields_a_scalar_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(G * R, C, generator=g) * 0.5}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(G, R, C)
    return {"y": torch.cat([x[0] * 2.0, x[1] * 3.0])}


SPEC = KernelSpec(
    name="loop_if_yields_a_scalar_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (G * R, C)), Arg("y", OUT, "float32", (G * R, C))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
