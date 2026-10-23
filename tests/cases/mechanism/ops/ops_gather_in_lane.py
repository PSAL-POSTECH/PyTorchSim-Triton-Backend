"""A [128, 8] tile gathered along its registers: lane l, register d takes register idx[l, d] of its own
lane (here d ^ 3), with no lane crossing.

torch equivalent: x.exp().gather(1, (arange(8) ^ 3).expand(128, 8)).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

L, D = 128, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_gather_in_lane_kernel(x_ptr, y_ptr, L: tl.constexpr, D: tl.constexpr):
        at = tl.arange(0, L)[:, None] * D + tl.arange(0, D)[None, :]
        x = tl.exp(tl.load(x_ptr + at))
        idx = tl.broadcast_to(tl.arange(0, D)[None, :] ^ 3, (L, D))
        tl.store(y_ptr + at, tl.gather(x, idx, 1))

    return ops_gather_in_lane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(L, D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"y": inputs["x"].exp().gather(1, (torch.arange(D) ^ 3).expand(L, D))}


SPEC = KernelSpec(
    name="ops_gather_in_lane_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "L": "constexpr", "D": "constexpr"},
    constexprs={"L": L, "D": D},
    args=[Arg("x", IN, "float32", (L, D)), Arg("y", OUT, "float32", (L, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
