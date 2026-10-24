"""Pins a tl.reduce whose combiner is its body (online softmax) folded ALONG the vector by a
torchsim.reduction: 255 after the first-element init, cut to an odd divisor per trip (85 on v6e),
exp padded. Breaks if the fold seeds every slot with the init, or trips on the odd widths."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _online_softmax(m1, l1, m2, l2):
        m = tl.maximum(m1, m2)
        return m, l1 * tl.exp(m1 - m) + l2 * tl.exp(m2 - m)

    @triton.jit
    def reduce_general_combiner_along_the_vector(x_ptr, m_ptr, l_ptr, M: tl.constexpr,
                                                 N: tl.constexpr):
        rm = tl.arange(0, M)
        x = tl.load(x_ptr + rm[:, None] * N + tl.arange(0, N)[None, :])
        m, l = tl.reduce((x, tl.full((M, N), 1.0, tl.float32)), 1, _online_softmax)
        tl.store(m_ptr + rm, m)
        tl.store(l_ptr + rm, l)

    return reduce_general_combiner_along_the_vector


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    m = x.amax(dim=1)
    return {"m": m, "l": torch.exp(x - m[:, None]).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_general_combiner_along_the_vector", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*fp32", "l_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)),
          Arg("m", OUT, "float32", (M,)),
          Arg("l", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-passes/07-pytorchsim-vectorize.mlir": {"torchsim.reduction": 1}},
)
