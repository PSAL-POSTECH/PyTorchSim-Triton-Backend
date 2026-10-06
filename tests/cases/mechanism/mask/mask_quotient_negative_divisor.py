"""`(i // -4) < 3` -- the divisor is NEGATIVE, so the bound cannot travel.

foldBoundThroughDiv reads `(x // K) < c` as `x < c*K`, which holds only for
K > 0 and c > 0.  With K = -4 the quotient walks the other way: every i in
[0, 64) gives a quotient of 0 or less, so the mask is ALL TRUE, while `i < -12`
is all false.  Nothing is loaded and every lane comes back `other`.

WHAT WOULD BREAK IT: dropping `k <= 0` from the guard.  Until this file the
suite had no kernel with a negative divisor at all, so that half of the
condition was free to delete -- measured, the pass fires ten times over the
whole suite and the k/c gate rejected none of them.

torch equivalent: x, in full -- the mask is true everywhere.

foldBoundThroughDiv's k > 0 half. With K negative the identity `x // K < c` <-> `x < c*K` reverses, and the whole tile stops being read.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64
K = -4
C = 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_quotient_negative_divisor_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                              K: tl.constexpr, C: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x_ptr + i, (i // K) < C, other=-1.0)
        tl.store(o_ptr + i, v)

    return mask_quotient_negative_divisor_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N)
    keep = torch.div(i, K, rounding_mode="trunc") < C     # every lane
    return {"o": torch.where(keep, inputs["x"], torch.full((N,), -1.0))}


SPEC = KernelSpec(
    name="mask_quotient_negative_divisor_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr", "C": "constexpr"},
    constexprs={"N": N, "K": K, "C": C},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
