"""GRID: `p // K` on an axis whose MASK bound is a runtime value.

`static_mask_dims` is a box, one bound per axis, and `m` elements of an axis cut
into [outer, inner] is a box only when inner divides m. A runtime m cannot say
that, so the axis must not be cut -- and the division has to learn it from the
ACCESS, which the address expression never sees.

A runtime mask bound on the axis a `//` would cut: the division has to read the access's mask before it decides.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K, NLIVE = 256, 4, 100        # K divides N, so the cut is otherwise free
NX = N // K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_bound_is_runtime_kernel(x_ptr, o_ptr, n, N: tl.constexpr,
                                     K: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (p // K), p < n, other=0.0)
        tl.store(o_ptr + p, v, p < n)

    return mask_bound_is_runtime_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    o = torch.zeros(N)
    o[p < NLIVE] = inputs["x"][(p // K)][p < NLIVE]
    return {"o": o}


SPEC = KernelSpec(
    name="mask_bound_is_runtime_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32",
               "N": "constexpr", "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE CLAIM IS THE RUNG, and there are three to tell apart. The cut is
    # refused, so this is the index table -- NOT the split (the mask could not
    # follow it) and NOT one address per element, which is where it lands if the
    # division splits first and the mask refuses afterwards.
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 1}},
    extra={"scalar_args": [("n", "int32_t")], "scalar_values": {"n": NLIVE}},
)
