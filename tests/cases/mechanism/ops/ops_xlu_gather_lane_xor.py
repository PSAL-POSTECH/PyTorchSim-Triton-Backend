"""A row on the lanes gathered by an index tile: lane l reads lane idx[l] (here l ^ 1).

torch equivalent: x.exp()[torch.arange(128) ^ 1].
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_gather_lane_xor_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.exp(tl.load(x_ptr + r))
        tl.store(y_ptr + r, tl.gather(x, r ^ 1, 0))

    return ops_xlu_gather_lane_xor_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"y": inputs["x"].exp()[torch.arange(N) ^ 1]}


SPEC = KernelSpec(
    name="ops_xlu_gather_lane_xor_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
