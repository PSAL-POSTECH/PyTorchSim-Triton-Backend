"""`(i - N) // K` on a negative base: divsi truncates toward zero, so the runs are 1, K, ..., K, K-1 long and
not the K-long runs from a multiple of K that divideByReshapingTheAxis's split writes. Breaks (wrong values)
if a split of `x // K` is taken without proving x >= 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 64, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_split_negative_base_truncates_kernel(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr):
        i = tl.arange(0, N)
        q = (i - N) // K
        tl.store(o_ptr + i, tl.load(x_ptr + (q + N // K)))

    return ptr_div_split_negative_base_truncates_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N // K + 1, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N)
    q = torch.div(i - N, K, rounding_mode="trunc")
    return {"o": inputs["x"][q + N // K].clone()}


SPEC = KernelSpec(
    name="ptr_div_split_negative_base_truncates_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "float32", (N // K + 1,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
