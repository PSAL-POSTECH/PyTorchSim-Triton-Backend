"""GRID: a step on TWO axes. The pieces are the product, and the threshold is
on the product -- not on either axis.

Both axes carry `// K`, so the runs multiply: 4 on axis 0 times 4 on axis 1 is
16 pieces, which is over MAX_WRAP_PIECES (12) and therefore addresses per
element. That is the point -- one axis alone would be inside the threshold and
this file is what says the cap is read on the PRODUCT.

Two stepped axes: the pieces are the product, 4 x 4 = 16, over the threshold of 12.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 16, 16, 5          # ceil(16/5) = 4 runs per axis, 16 in the product
SM = (M + K - 1) // K
SN = (N + K - 1) // K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_two_axes_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                             K: tl.constexpr, SN: tl.constexpr):
        r = tl.arange(0, M)[:, None]
        c = tl.arange(0, N)[None, :]
        v = tl.load(x_ptr + (r // K) * SN + (c // K))
        tl.store(o_ptr + r * N + c, v)

    return step_two_axes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(SM * SN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(M)[:, None]
    c = torch.arange(N)[None, :]
    return {"o": inputs["x"][(r // K) * SN + (c // K)].reshape(-1)}


SPEC = KernelSpec(
    name="step_two_axes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr", "K": "constexpr", "SN": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "SN": SN},
    args=[Arg("x", IN, "float32", (SM * SN,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
