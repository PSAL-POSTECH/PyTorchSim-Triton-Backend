"""A wrapped pointer advanced by a loop: the advance lands inside the modulo.

`p = base + (i % W)` then `p += W` each iteration means `((i % W) + kW)`.
PtrAnalysis folds the advance into the wrapped tptr's OFFSET, so the descriptor
means `((i + kW) % W)` -- which is `i % W`, the same address every iteration.

torch equivalent: o[i] = sum(x[(i % W) + k*W] for k in range(K))

WHAT BREAKS IT: carrying the loop's advance in the tptr's axis OFFSET when that axis wraps. It is then taken modulo too, and 48 per trip against a wrap of 48 cancels exactly -- every lane came back K*x[i%W], the same address read K times, against a reference walking K rows. Same kernel with `off = i` was exact throughout, which is what named the wrap as the ingredient. Fixed by giving tts.make_tptr a base_offset that is added AFTER the wrap and by letting the loop carry it as its own component: the axis offset stays 0 and `arith.addi` advances the base offset instead.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK, W, K = 64, 48, 4
XN = W * K              # every address the loop reaches is in range


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_advanced_in_a_loop_kernel(x_ptr, o_ptr, W: tl.constexpr,
                                           BLOCK: tl.constexpr,
                                           K: tl.constexpr):
        i = tl.arange(0, BLOCK)
        p = x_ptr + (i % W)
        acc = tl.zeros((BLOCK,), tl.float32)
        for _ in range(K):
            acc += tl.load(p)
            p += W                       # OUTSIDE the wrap, not inside it
        tl.store(o_ptr + i, acc)

    return ptr_wrap_advanced_in_a_loop_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(XN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(BLOCK, dtype=torch.float32)
    for i in range(BLOCK):
        o[i] = sum(x[(i % W) + k * W] for k in range(K))
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_wrap_advanced_in_a_loop_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "W": "constexpr",
               "BLOCK": "constexpr", "K": "constexpr"},
    constexprs={"W": W, "BLOCK": BLOCK, "K": K},
    args=[Arg("x", IN, "float32", (XN,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect_ir={"03-linalg.mlir": {"tts.make_tptr": 2}},
)
