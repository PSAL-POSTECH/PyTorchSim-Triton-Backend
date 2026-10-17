"""One row reduced while it is spread over the lanes: softmax's statistics over [1, N].

    x    (1, N), element j on lane j % lanes (the row as the converter lays it)
    out  x - max(x) and sum(exp(x - max(x)))

WHAT THIS PINS. With every other dim a unit, the reduced dim keeps its lanes; the lowering folds
each lane's own elements, then the lanes' partials through the cross-lane unit, and hands the
statistic back to the spread row. Gathering the row into one lane walks it one lane long.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 8192


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_one_row_across_lanes(x_ptr, o_ptr, s_ptr, N: tl.constexpr):
        rn = tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + rn)
        m = tl.max(x, axis=1)[:, None]
        e = tl.exp(x - m)
        tl.store(o_ptr + rn, x - m)
        tl.store(s_ptr + tl.arange(0, 1), tl.sum(e, axis=1))

    return reduce_one_row_across_lanes


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    m = x.max()
    return {"o": x - m, "s": torch_sum(x - m)}


def torch_sum(d):
    return d.exp().sum().reshape(1)


SPEC = KernelSpec(
    name="reduce_one_row_across_lanes", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "s_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,)),
          Arg("s", OUT, "float32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
