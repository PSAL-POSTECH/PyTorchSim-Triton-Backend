"""A WRAP whose base is a scalar tt.bitcast: the tile is i32, the argument
f32, and taking the element type from the argument allocates an f32
scratchpad. Breaks it: _emit_wrap reading ShapedType(dram_arg.type) again.

The wrap entrance with a scalar-bitcast base: the tile is i32 and the DRAM argument is f32, so an emitter that reads the element type off the argument allocates the wrong scratchpad.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

WM = 24          # wrap extent
BLOCK = 32       # tile, larger than WM so the wrap is inside the tile
NW = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_bitcast_base_kernel(w_ptr, o_ptr, WM: tl.constexpr,
                                     BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        pb = tl.cast(w_ptr, tl.pointer_type(tl.int32))
        tl.store(o_ptr + r, tl.load(pb + (r % WM)))

    return ptr_wrap_bitcast_base_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(NW, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(BLOCK)
    return {"o": inputs["w"].view(torch.int32)[r % WM]}


SPEC = KernelSpec(
    name="ptr_wrap_bitcast_base_kernel", kernel=kernel,
    signature={"w_ptr": "*fp32", "o_ptr": "*i32", "WM": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"WM": WM, "BLOCK": BLOCK},
    args=[Arg("w", IN, "float32", (NW,)), Arg("o", OUT, "int32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
