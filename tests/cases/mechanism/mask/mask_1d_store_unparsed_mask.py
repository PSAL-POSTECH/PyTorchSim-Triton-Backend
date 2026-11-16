"""A store whose mask MaskAnalysis cannot parse keeps its structured address and its masked-off lanes unwritten.

`o_ptr + r` is a plain contiguous address and `r < bnd` has a bound read from memory,
which MaskAnalysis refuses. The store must stay a tts.store carrying the whole mask
as its element mask (no tts.scatter); o is INOUT so a masked-off lane that was
written would show as a changed sentinel. Breaks if the store gives up on the address
or lowers the element mask as an unmasked write.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

BLOCK = 32
BNDVAL = 12


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_1d_store_unparsed_mask_kernel(x_ptr, b_ptr, o_ptr,
                                           BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        bnd = tl.load(b_ptr + r)
        v = tl.load(x_ptr + r)
        tl.store(o_ptr + r, v, mask=r < bnd)

    return mask_1d_store_unparsed_mask_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK, generator=g, dtype=torch.float32),
            "b": torch.full((BLOCK,), BNDVAL, dtype=torch.int32),
            "o": torch.full((BLOCK,), 7.0, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(BLOCK)
    return {"o": torch.where(r < BNDVAL, inputs["x"], inputs["o"])}


SPEC = KernelSpec(
    name="mask_1d_store_unparsed_mask_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (BLOCK,)), Arg("b", IN, "int32", (BLOCK,)),
          Arg("o", INOUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.scatter": 0, "tts.gather": 0}},
)
