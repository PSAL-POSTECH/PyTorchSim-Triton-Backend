"""An in-bank gather whose INDEX element is wider than the data's: bf16 rows read by an i32 index.
The register group is sized by the data (i16), so the index row read at its own width is twice that
and overruns one LMUL-8 group. One row per lane, so nothing of the row base -- only the width."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

L, D = 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_index_wider_than_the_element_kernel(x_ptr, i_ptr, y_ptr, L: tl.constexpr,
                                                   D: tl.constexpr):
        at = tl.arange(0, L)[:, None] * D + tl.arange(0, D)[None, :]
        x = tl.exp(tl.load(x_ptr + at)).to(tl.bfloat16)
        idx = tl.load(i_ptr + at)
        tl.store(y_ptr + at, tl.gather(x, idx, 1))

    return gather_index_wider_than_the_element_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(L, D, generator=g, dtype=torch.float32),
            "i": torch.randint(0, D, (L, D), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    return {"y": inputs["x"].exp().to(torch.bfloat16).gather(1, inputs["i"].long())}


SPEC = KernelSpec(
    name="gather_index_wider_than_the_element_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "y_ptr": "*bf16", "L": "constexpr", "D": "constexpr"},
    constexprs={"L": L, "D": D},
    args=[Arg("x", IN, "float32", (L, D)), Arg("i", IN, "int32", (L, D)),
          Arg("y", OUT, "bfloat16", (L, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
