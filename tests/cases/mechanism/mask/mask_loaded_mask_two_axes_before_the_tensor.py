"""A loaded row mask over a 2-D box that starts two rows before its tensor (`x[(r - 2) * 16 + c]`, rows 0 and 1
masked off): an element-masked read of two moving axes. The select drops rows 0 and 1, but the box is read whole,
and with x first in DRAM its first rows are below DRAM (Spike: User load segfault @ dram_base - 128).

Clamping one axis cannot be exact here: a row the clamp cuts may hold valid elements, so the access is refused by
name (MASKED_BOX_BEFORE_TENSOR) rather than read out of range. WHAT WOULD LIFT IT: a clamp on every moving axis,
or the masked rows parked as a gather's are.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

BM, BN, SKIP = 8, 16, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_loaded_mask_two_axes_before_the_tensor_kernel(
            x_ptr, f_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr, SKIP: tl.constexpr):
        r = tl.arange(0, BM)
        c = tl.arange(0, BN)
        live = tl.load(f_ptr + r) != 0
        v = tl.load(x_ptr + ((r[:, None] - SKIP) * BN + c[None, :]), live[:, None], other=0.0)
        tl.store(o_ptr + (r[:, None] * BN + c[None, :]), v)

    return mask_loaded_mask_two_axes_before_the_tensor_kernel


def make_inputs(torch, seed=0):
    import torch as t
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn((BM - SKIP) * BN, generator=g, dtype=torch.float32),
            "f": (t.arange(BM) >= SKIP).to(t.int32)}


def reference(inputs):
    import torch
    return {"o": torch.cat([torch.zeros(SKIP * BN), inputs["x"]])}


SPEC = KernelSpec(
    name="mask_loaded_mask_two_axes_before_the_tensor_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32", "BM": "constexpr", "BN": "constexpr",
               "SKIP": "constexpr"},
    constexprs={"BM": BM, "BN": BN, "SKIP": SKIP},
    args=[Arg("x", IN, "float32", ((BM - SKIP) * BN,)), Arg("f", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("transform", Reason.MASKED_BOX_BEFORE_TENSOR),
    expect_why="an element-masked box of two moving axes that starts before its tensor: no exact clamp, and a "
               "masked-off element must not be read",
)
