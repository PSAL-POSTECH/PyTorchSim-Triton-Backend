"""A load whose mask is itself loaded (no box can state it) at an offset that is negative where the mask is off:
`x[lane - 64]` with lanes 0..63 masked. The mask stays an element mask, so the load is a full-extent box that
starts 64 elements before its tensor (placed first, so below DRAM), its first element masked off.

WHAT BREAKS IT: a transfer whose mask selects among its elements after the move seeded as moving its first
element, its DRAM offset inside its argument. Here the offset is -64 against [0, 63]: the interval is empty and
simplify-index-arithmetic's worklist narrowed forever. The transfer now says `element_masked`, and the box
read must not touch memory below the tensor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK, HALF = 128, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_loaded_mask_before_the_tensor_kernel(x_ptr, f_ptr, o_ptr, BLOCK: tl.constexpr, HALF: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = tl.load(f_ptr + lane) != 0
        v = tl.load(x_ptr + (lane - HALF), live, other=0.0)
        tl.store(o_ptr + lane, v)

    return mask_loaded_mask_before_the_tensor_kernel


def make_inputs(torch, seed=0):
    import torch as t
    g = torch.Generator().manual_seed(seed)
    return {"f": (t.arange(BLOCK) >= HALF).to(t.int32),
            "x": torch.randn(HALF, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.cat([torch.zeros(HALF), inputs["x"]])}


SPEC = KernelSpec(
    name="mask_loaded_mask_before_the_tensor_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32", "BLOCK": "constexpr", "HALF": "constexpr"},
    constexprs={"BLOCK": BLOCK, "HALF": HALF},
    args=[Arg("x", IN, "float32", (HALF,)), Arg("f", IN, "int32", (BLOCK,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
