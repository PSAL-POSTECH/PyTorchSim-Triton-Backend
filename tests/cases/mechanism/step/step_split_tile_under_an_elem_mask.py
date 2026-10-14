"""A read whose rows step and wrap -- `(m // K) * SH + (m % K) * SW` -- under a mask loaded from memory.

The split makes the tile [BM/K, K, BC] while the mask stays [BM, BC], and tts.load wants its
elem_mask in the tile's shape: the shape of a convolution's input read through a view.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BC, K, SH, SW = 64, 16, 4, 256, 32
NX = (BM // K - 1) * SH + (K - 1) * SW + BC


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_split_tile_under_an_elem_mask_kernel(x_ptr, f_ptr, o_ptr, BM: tl.constexpr,
                                                  BC: tl.constexpr, K: tl.constexpr,
                                                  SH: tl.constexpr, SW: tl.constexpr):
        m = tl.arange(0, BM)
        c = tl.arange(0, BC)
        keep = tl.load(f_ptr + m) != 0
        row = (m // K) * SH + (m % K) * SW
        v = tl.load(x_ptr + row[:, None] + c[None, :], mask=keep[:, None], other=0.0)
        tl.store(o_ptr + m[:, None] * BC + c[None, :], v)

    return step_split_tile_under_an_elem_mask_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "f": (torch.arange(BM, dtype=torch.int32) % 3).contiguous()}


def reference(inputs):
    import torch
    m = torch.arange(BM)
    row = (m // K) * SH + (m % K) * SW
    v = inputs["x"][row[:, None] + torch.arange(BC)[None, :]]
    keep = (inputs["f"] != 0)[:, None]
    return {"o": torch.where(keep, v, torch.zeros_like(v)).reshape(-1)}


SPEC = KernelSpec(
    name="step_split_tile_under_an_elem_mask_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32", "BM": "constexpr",
               "BC": "constexpr", "K": "constexpr", "SH": "constexpr", "SW": "constexpr"},
    constexprs={"BM": BM, "BC": BC, "K": K, "SH": SH, "SW": SW},
    args=[Arg("x", IN, "float32", (NX,)), Arg("f", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BM * BC,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
