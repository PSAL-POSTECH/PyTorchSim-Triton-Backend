"""A fusable elementwise chain living ONLY in the scf.for body: 3 generics
enter fuse and 1 leaves, where a top-level-only gate left all 3.
Breaks it: hoisting the chain out, or shrinking it below two generics.

POSITIVE. Elementwise chain confined to the scf.for body; pins fuse's gate counting generics through nested regions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
K = 4
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop18_elementwise_chain_in_body_kernel(x_ptr, y_ptr, o_ptr,
                                                K: tl.constexpr,
                                                BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = x_ptr + r
        q = y_ptr + r
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            a = tl.load(p)
            b = tl.load(q)
            acc += a * b + a                # 3 generics, all inside scf.for
            p = p + BLOCK
            q = q + BLOCK
        tl.store(o_ptr + r, acc)

    return loop18_elementwise_chain_in_body_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x, y = inputs["x"], inputs["y"]
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(K):
        a = x[k * BLOCK:(k + 1) * BLOCK]
        b = y[k * BLOCK:(k + 1) * BLOCK]
        acc += a * b + a
    return {"o": acc}


SPEC = KernelSpec(
    name="loop18_elementwise_chain_in_body_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "K": "constexpr", "BLOCK": "constexpr"},
    constexprs={"K": K, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("y", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
