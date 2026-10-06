"""A remainder masked against a bound AT or ABOVE its modulus -- always true.

`t % K` lives in `[0, K)`, so `< B` excludes nothing once `B >= K`. The mask is
a statement that cannot be false, and leaving it standing costs two rungs: the
`%` stays inside the mask where nothing reads it, the pointer is never rebuilt
as tts.make_tptr, and the access becomes one address per element.

THE GUARD USED TO ASK B == K. That is the case a kernel writes when it masks a
remainder against its own modulus; this file is the one that writes a BLOCK
wider than the wrap, which is just as always-true and used to fall.

    load 1   (t % 8) masked by  t % 8 < 8      B == K, and this already worked
    load 2   (t % 4) masked by  t % 4 < 16     B >  K, the case this pins

THE TWO MODULI DIFFER SO THAT CSE CANNOT MERGE THEM. Written with one modulus
the two loads are the same expression, cse folds them into one, and the file
stops being able to tell the two arms apart.

BOTH MASKS ARE ALWAYS TRUE, so a regression does not show as a wrong number --
it shows as load 2 falling to one address per element while load 1 stays
structured, which is why expect_ir pins the rungs.

torch equivalent: o = cat([x[t % 8], x[t % 4]]).

A remainder masked at or above its modulus. Both loads are always true; before the guard asked >= instead of ==, the second fell to one address per element while the first stayed structured.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64
K = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_remainder_bound_at_or_above_kernel(x_ptr, o_ptr,
                                                N: tl.constexpr,
                                                K: tl.constexpr):
        t = tl.arange(0, N)
        u = t % K
        v = t % (K // 2)
        a = tl.load(x_ptr + u, mask=u < K, other=0.0)
        b = tl.load(x_ptr + v, mask=v < 2 * K, other=0.0)
        tl.store(o_ptr + t, a)
        tl.store(o_ptr + N + t, b)

    return mask_remainder_bound_at_or_above_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    t = torch.arange(N, dtype=torch.int64)
    x = inputs["x"]
    return {"o": torch.cat([x[t % K], x[t % (K // 2)]])}


SPEC = KernelSpec(
    name="mask_remainder_bound_at_or_above_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "float32", (K,)), Arg("o", OUT, "float32", (2 * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # TWO STRUCTURED LOADS AND NO GATHER. The numbers are the same either way,
    # so the rung is the only thing that moves and the only thing worth pinning.
    expect_ir={"03-ttshared.mlir": {"tts.make_tptr": 4, "tts.gather": 0}},
)
