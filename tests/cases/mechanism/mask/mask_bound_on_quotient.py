"""A guard on a QUOTIENT -- `(x // K) < c` -- which MaskAnalysis reads as
`x < c*K`, the only form it can place on a lane.

WHAT IT PINS: resnet's conv shape at one axis. A bound on `x // K` is a bound
on `x` and has to arrive as one; left as a quotient the mask is unanalysable
and the store becomes a gather.

WHAT PROVES IT, with parseCmp folding the bound through the division and
without it:
    on   tts.make_tptr + tts.load
    off  tts.make_tptr + tts.gather

foldBoundThroughDiv: `(x // K) < c` is `x < c*K`, exact for every x when c > 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, ROWS, BLOCK = 8, 3, 32       # rows 0..23 exist; the block stages 32
N = K * ROWS


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_bound_on_quotient_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                      ROWS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        keep = (i // K) < ROWS          # == i < ROWS*K
        v = tl.load(x_ptr + i, keep, other=0.0)
        tl.store(o_ptr + i, v)

    return mask_bound_on_quotient_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(BLOCK)
    x = torch.cat([inputs["x"], torch.zeros(BLOCK - N)])
    return {"o": torch.where(i < N, x, torch.zeros(BLOCK))}


SPEC = KernelSpec(
    name="mask_bound_on_quotient_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr",
               "ROWS": "constexpr", "BLOCK": "constexpr"},
    constexprs={"K": K, "ROWS": ROWS, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"03-linalg.mlir": {"tts.gather": 0, "tts.load": 1}},
)
