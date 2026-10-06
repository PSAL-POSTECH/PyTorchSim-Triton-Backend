"""A wrap on the MIDDLE axis of a rank-3 tile.

    idx = r4*64 + (r8 % M)*8 + r8        wrap on axis 1 of three

The two axes beside it keep their real dram strides -- `[64, 0, 1]` -- so the
descriptor still walks both while only the wrapped one is named per element.
That is the whole reason a wrap is worth holding instead of falling to one
address per element.

WHAT IT REPLACED. convert_accesses_to_transfer refused this by name, calling itself a
TRIPWIRE for the day upstream could build one; _emit_wrap next to it already
said "any dim works" because it is built at full tile shape from `linalg.index
d`. Upstream can build one now, and the tripwire was right about itself.

M=3 under 8, so the middle axis wraps twice over.

A wraparound on an interior axis: the axes beside it keep their dram strides, which is what the wrap slot buys.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

D0, D1, D2, M = 4, 8, 8, 3
N = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_middle_axis(x_ptr, o_ptr, M: tl.constexpr):
        r4 = tl.arange(0, 4)
        r8 = tl.arange(0, 8)
        c = tl.load(x_ptr + r4[:, None, None] * 64
                    + (r8 % M)[None, :, None] * 8 + r8[None, None, :])
        tl.store(o_ptr + r4[:, None, None] * 64 + r8[None, :, None] * 8
                 + r8[None, None, :], c)

    return wrap_middle_axis


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r4, r8 = torch.arange(4), torch.arange(8)
    idx = (r4[:, None, None] * 64 + (r8 % M)[None, :, None] * 8
           + r8[None, None, :])
    return {"o": x[idx].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_middle_axis", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr"},
    constexprs={"M": M},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (D0 * D1 * D2,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"03-ttshared.mlir": {"tts.gather": 0}},
)
