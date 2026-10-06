"""A rem and a div from one index, laid on DIFFERENT axes: an outer sum.

`(x % K)*a + (x // K)*(K*a)` is `x*a` only while both halves sit on the same
axis. Expanded onto axis 0 and axis 1 it is a rank-2 index, and flattening it
puts one row's value in every column.

torch equivalent: o[p][q] = (p % K) + (q // K) * K

WHAT BREAKS IT: psto-reflatten-div-rem matching the two halves by their value, their divisor and their scale ratio, and not by WHERE each one sits. Both peel to the same `i` here, so it collapsed the sum to `i` re-wrapped through the rem side's expand -- 32 of 64 elements wrong, every row repeated across the columns. The sibling is ptr_reflatten_div_rem.py, the flattening the pass is for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 8, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_rem_on_two_axes_kernel(o_ptr, N: tl.constexpr,
                                       K: tl.constexpr):
        i = tl.arange(0, N)
        lhs = (i % K)[:, None]           # on axis 0
        rhs = (i // K)[None, :]          # on axis 1 -- a different axis
        v = lhs + rhs * K                # (N, N), an OUTER sum
        r = tl.arange(0, N)
        tl.store(o_ptr + r[:, None] * N + r[None, :], v)

    return ptr_div_rem_on_two_axes_kernel


def make_inputs(torch, seed=0):
    return {}


def reference(inputs):
    import torch
    o = torch.zeros(N * N, dtype=torch.int32)
    for p in range(N):
        for q in range(N):
            o[p * N + q] = (p % K) + (q // K) * K
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_div_rem_on_two_axes_kernel",
    kernel=kernel,
    signature={"o_ptr": "*i32", "N": "constexpr", "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("o", OUT, "int32", (N * N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
