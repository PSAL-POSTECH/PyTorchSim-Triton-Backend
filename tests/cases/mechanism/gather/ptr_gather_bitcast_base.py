"""A GATHER whose base is a scalar tt.bitcast: the tile is i32, the argument
f32, so the scratchpad is made of the wrong type unless _tile_elem is asked.
Breaks it: _emit_partial_unstructured reading ShapedType(dram_arg.type) again.

The gather entrance with a scalar-bitcast base: the tile is i32 and the DRAM argument is f32, so an emitter that reads the element type off the argument allocates the wrong scratchpad.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_bitcast_base_kernel(x_ptr, i_ptr, o_ptr,
                                       BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)
        pb = tl.cast(x_ptr, tl.pointer_type(tl.int32))
        tl.store(o_ptr + r, tl.load(pb + i))

    return ptr_gather_bitcast_base_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": torch.randint(0, N, (BLOCK,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].view(torch.int32)[inputs["i"].long()]}


SPEC = KernelSpec(
    name="ptr_gather_bitcast_base_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "int32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
