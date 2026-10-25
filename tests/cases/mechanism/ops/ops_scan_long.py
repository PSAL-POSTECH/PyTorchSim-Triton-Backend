"""A prefix scan twice the lanes long, folded as XLA's ReduceWindowRewriter folds it: [2, 256],
each row scanned, the row totals scanned and added back. Breaks if the fold goes, or a row total
cut from its tile feeds an XLU move that lower_to_xlu cannot lower.

A 512-wide scan on 256 lanes: two folds, one per scan, keep its shifts off DRAM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def longscan_kernel(x_ptr, o_ptr, p_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        a = tl.load(x_ptr + r)
        tl.store(o_ptr + r, tl.cumsum(a, axis=0))
        tl.store(p_ptr + r, tl.cumprod(a * 0.01 + 1.0, axis=0))

    return longscan_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x.cumsum(0), "p": (x * 0.01 + 1.0).cumprod(0)}


SPEC = KernelSpec(
    name="longscan_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "p_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,)),
          Arg("p", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-passes/02-pytorchsim-expand-unsupported-ops.mlir": {"tensor.expand_shape": 2}},
)
