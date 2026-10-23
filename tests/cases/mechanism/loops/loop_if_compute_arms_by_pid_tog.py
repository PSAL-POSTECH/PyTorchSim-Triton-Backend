"""Pins a branch whose arms only compute and whose condition is a program id as a TOG if/else the TOG
computes itself: program 0 runs eight exps, program 1 one add. Breaks if the branch is folded into the
segment around it (one compute fewer per arm in tog.cpp), or the TOG cannot state its condition."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, G = 8, 128, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_if_compute_arms_by_pid_tog(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * R * C + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + offs)
        if pid == 0:
            y = x
            for _ in tl.static_range(8):
                y = tl.exp(y * 0.5) - 1.0
        else:
            y = x + 1.0
        tl.store(y_ptr + offs, y)

    return loop_if_compute_arms_by_pid_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(G * R, C, generator=g) * 0.5}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(G, R, C)
    y = x[0]
    for _ in range(8):
        y = torch.exp(y * 0.5) - 1.0
    return {"y": torch.cat([y, x[1] + 1.0])}


SPEC = KernelSpec(
    name="loop_if_compute_arms_by_pid_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (G * R, C)), Arg("y", OUT, "float32", (G * R, C))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect_ir={"tog.cpp": {"togsim::compute": 2}},
)
