"""Pins a branch whose arms only compute as a TOG if/else chosen by data: program p reads f[p], and
f > 0 runs eight exps where f <= 0 runs one add, so program 0 (f = 1) and program 1 (f = -1) take
different arms. Breaks if the branch is refused, folded into one segment, or one arm is timed for both."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, G = 8, 128, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_if_compute_arms_by_data_tog(x_ptr, f_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * R * C + tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + offs)
        if tl.load(f_ptr + pid) > 0:
            y = x
            for _ in tl.static_range(8):
                y = tl.exp(y * 0.5) - 1.0
        else:
            y = x + 1.0
        tl.store(y_ptr + offs, y)

    return loop_if_compute_arms_by_data_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(G * R, C, generator=g) * 0.5, "f": torch.tensor([1.0, -1.0])}


def reference(inputs):
    import torch
    x, f = inputs["x"].reshape(G, R, C), inputs["f"]
    ys = []
    for p in range(G):
        y = x[p]
        if f[p] > 0:
            for _ in range(8):
                y = torch.exp(y * 0.5) - 1.0
        else:
            y = y + 1.0
        ys.append(y)
    return {"y": torch.cat(ys)}


SPEC = KernelSpec(
    name="loop_if_compute_arms_by_data_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*fp32", "y_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (G * R, C)), Arg("f", IN, "float32", (G,)),
          Arg("y", OUT, "float32", (G * R, C))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect_ir={"tog.cpp": {"togsim::compute": 3}},
)
