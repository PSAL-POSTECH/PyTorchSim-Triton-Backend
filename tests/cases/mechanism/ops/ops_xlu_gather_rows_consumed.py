"""A gather along the lanes on a [128, 128] tile (lane l of register r reads lane l ^ 1 of register r),
its result read by the next loop a register group of rows a trip: each trip pops its own rows.

torch equivalent: x.exp().gather(0, lane ^ 1) * 2 + x.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

L, D = 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_gather_rows_consumed_kernel(x_ptr, y_ptr, L: tl.constexpr, D: tl.constexpr):
        lane = tl.arange(0, L)[:, None]
        reg = tl.arange(0, D)[None, :]
        pos = reg * L + lane
        x = tl.load(x_ptr + pos)
        at = tl.broadcast_to(lane ^ 1, (L, D))
        tl.store(y_ptr + pos, tl.gather(tl.exp(x), at, 0) * 2 + x)

    return ops_xlu_gather_rows_consumed_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(L * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(D, L)
    return {"y": (x.exp()[:, torch.arange(L) ^ 1] * 2 + x).reshape(-1)}


SPEC = KernelSpec(
    name="ops_xlu_gather_rows_consumed_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "L": "constexpr", "D": "constexpr"},
    constexprs={"L": L, "D": D},
    args=[Arg("x", IN, "float32", (L * D,)), Arg("y", OUT, "float32", (L * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
