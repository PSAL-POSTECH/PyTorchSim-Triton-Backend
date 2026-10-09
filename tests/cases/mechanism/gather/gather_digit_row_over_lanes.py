"""A row read at `x // 2`, broadcast down the lanes: group norm's per-group statistic.

WHAT IT IS. Inductor's group norm (Stable Diffusion's UNet, two channels a group)
reads `mean[x // 2]` once per column and uses it in every row. The floor divide
cuts the column axis into digits, so the read is a 1 x 8 x 2 tile collapsed to
16, and the rows it is broadcast over are the lanes.

WHAT IT USED TO DO. materialize_layouts followed a replica back through a
collapse only when the collapse dropped unit dims:

    DigitRefused: a replica read across the lanes whose producer,
    tensor.collapse_shape, cannot make its copies; innermost:
    tensor.collapse_shape ... tensor<1x8x2xf32> into tensor<16xf32>

The replica is now built in the wide shape with the kept dim split the way the
collapse splits it, then collapsed back.

torch equivalent: o[y, x] = a[y, x] - m[x // 2]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Y, X = 32, 64
YB, XB = 32, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_digit_row_over_lanes(a_ptr, m_ptr, o_ptr,
                                    YB: tl.constexpr, XB: tl.constexpr):
        xoff = tl.program_id(0) * XB
        y = tl.arange(0, YB)[:, None]
        x = xoff + tl.arange(0, XB)[None, :]
        a = tl.load(a_ptr + y + 32 * x)
        m = tl.load(m_ptr + (x // 2))
        tl.store(o_ptr + x + 64 * y, a - m)
    return gather_digit_row_over_lanes


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(Y * X, generator=g, dtype=torch.float32),
            "m": torch.randn(X // 2, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(X, Y).t()
    m = inputs["m"][torch.arange(X) // 2].reshape(1, X)
    return {"o": (a - m).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="gather_digit_row_over_lanes", kernel=kernel,
    signature={"a_ptr": "*fp32", "m_ptr": "*fp32", "o_ptr": "*fp32",
               "YB": "constexpr", "XB": "constexpr"},
    constexprs={"YB": YB, "XB": XB},
    args=[Arg("a", IN, "float32", (Y * X,)),
          Arg("m", IN, "float32", (X // 2,)),
          Arg("o", OUT, "float32", (Y * X,))],
    grid=(X // XB,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
