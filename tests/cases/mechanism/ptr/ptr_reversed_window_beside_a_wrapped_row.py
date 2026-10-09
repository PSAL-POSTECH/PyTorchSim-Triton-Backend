"""A reversed window on one axis beside a wrapped (`%`, `//`) row index on the other: a flipped
conv weight, as Inductor copies it for ConvTranspose2d.

WHAT IT IS. `aten.convolution(transposed=True)` decomposes to a convolution over the weight
flipped along its spatial dims (prims.rev), and Inductor folds the flip into the copy kernel
that lays the weight out for the template:

    tl.load(w_ptr + (3 + (-1)*x2 + 4*y1 + 256*y0))      y0 = y % 64, y1 = y // 64
    tl.store(o_ptr + (y0 + 64*x2 + 256*y1), ...)

WHAT IT USED TO DO. PtrAnalysis subtracts only a scalar, so `3 - x2` is demoted to a gather
on the x axis, and `parkWrapOnTheGatheredAxis` then looked the gathered axis up in the wrapped
side's state by the gathering side's axis index. The y side had split its stepped axis
(`4 * (y // 64)`) into two state axes, so x was state axis 2 there and axis 1 on the wrapped
side: `dimHasModulo(2)` on a rank-2 state. The release build failed the pass with no
diagnostic; the asserting build stops at `Assertion dim < getRank()`. The axes are now matched
by tensor axis, which is what addState's alignSplits does once the two sides meet.

torch equivalent: o = w.view(64, 64, 4).flip(-1).permute(1, 2, 0).reshape(-1)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Y, X = 4096, 4
YB, XB = 256, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_reversed_window_beside_a_wrapped_row(w_ptr, o_ptr, YB: tl.constexpr, XB: tl.constexpr):
        yindex = tl.program_id(1) * YB + tl.arange(0, YB)[:, None]
        x2 = tl.program_id(0) * XB + tl.arange(0, XB)[None, :]
        y0 = yindex % 64
        y1 = yindex // 64
        v = tl.load(w_ptr + (3 + (-1) * x2 + 4 * y1 + 256 * y0))
        tl.store(o_ptr + (y0 + 64 * x2 + 256 * y1), v)
    return ptr_reversed_window_beside_a_wrapped_row


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(Y * X, generator=g, dtype=torch.float32)}


def reference(inputs):
    w = inputs["w"].view(64, 64, 4)
    return {"o": w.flip(-1).permute(1, 2, 0).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ptr_reversed_window_beside_a_wrapped_row", kernel=kernel,
    signature={"w_ptr": "*fp32", "o_ptr": "*fp32", "YB": "constexpr", "XB": "constexpr"},
    constexprs={"YB": YB, "XB": XB},
    args=[Arg("w", IN, "float32", (Y * X,)),
          Arg("o", OUT, "float32", (Y * X,))],
    grid=(1, Y // YB), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
