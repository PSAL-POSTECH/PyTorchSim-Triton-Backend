"""A row read through a modulo index, broadcast down the lanes: RoPE's frequency table.

WHAT IT IS. Inductor's rotary embedding reads `inv_freq[x % 16]` once per column
and multiplies it into every row. The modulo is no stride, so the read becomes a
gather with an index tile (1 x 16), and the rows it is broadcast over are the
lanes -- a replica read across the lanes whose producer is that gather.

WHAT IT USED TO DO. materialize_layouts built such a replica by re-issuing the
load in the wide shape, and declined any load that carried an index:

    DigitRefused: a replica read across the lanes whose producer,
    tensor.collapse_shape, cannot make its copies; innermost: torchsim.transfer

Llama's attention (fused rope + bmm) stopped there. The index is a tile like any
other, so the wide load now takes a wide index, built the same way.

torch equivalent: o[y, x] = a[y, x] * t[x % 16]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Y, X, T = 32, 64, 16
YB, XB = 32, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_modulo_row_over_lanes(a_ptr, t_ptr, o_ptr,
                                     YB: tl.constexpr, XB: tl.constexpr):
        xoff = tl.program_id(0) * XB
        y = tl.arange(0, YB)[:, None]
        x = xoff + tl.arange(0, XB)[None, :]
        a = tl.load(a_ptr + x + 64 * y)
        t = tl.load(t_ptr + (x % 16))
        tl.store(o_ptr + x + 64 * y, a * t)
    return gather_modulo_row_over_lanes


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(Y * X, generator=g, dtype=torch.float32),
            "t": torch.randn(T, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(Y, X)
    t = inputs["t"][torch.arange(X) % T].reshape(1, X)
    return {"o": (a * t).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="gather_modulo_row_over_lanes", kernel=kernel,
    signature={"a_ptr": "*fp32", "t_ptr": "*fp32", "o_ptr": "*fp32",
               "YB": "constexpr", "XB": "constexpr"},
    constexprs={"YB": YB, "XB": XB},
    args=[Arg("a", IN, "float32", (Y * X,)),
          Arg("t", IN, "float32", (T,)),
          Arg("o", OUT, "float32", (Y * X,))],
    grid=(X // XB,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
