"""`(x % K) < K` as a mask term: always true, and MaskAnalysis reads it so.

WHAT IT PINS: that a remainder compared against its own modulus states nothing.
Left standing it is a `%` inside a mask, which MaskAnalysis declines, and the
whole access falls to a gather over an index it did not need.

WHAT PROVES IT, with parseCmp answering the term before it builds an
unstructured mask for it, and without:
    on   tts.make_tptr + tts.load
    off  tts.make_tptr + tts.gather

A remainder is smaller than its modulus, so the term is the whole extent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 64, 7


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_remainder_bound_is_true_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                            K: tl.constexpr):
        i = tl.arange(0, N)
        # The left term cannot be false; the right one is the real guard.
        keep = ((i % K) < K) & (i < N - 3)
        v = tl.load(x_ptr + i, keep, other=0.0)
        tl.store(o_ptr + i, v)

    return mask_remainder_bound_is_true_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N)
    return {"o": torch.where(i < N - 3, inputs["x"], torch.zeros(N))}


SPEC = KernelSpec(
    name="mask_remainder_bound_is_true_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.load": 1}},
)
