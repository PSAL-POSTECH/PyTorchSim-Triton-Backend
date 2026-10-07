"""Widening conversions across int16 and uint8, the dtypes the tables lacked.

int16 and the unsigned types were absent from tests/runtime/verify.py NP_DTYPE and
pytorchsim_triton_compiler/lowering/wrapper.py C_TYPE / DTYPE_BYTES, so a spec naming one died with a KeyError
on the dtype string before anything about the kernel was reached. Inductor
emits them routinely -- `_type_of` gives *i16 for any torch.int16 buffer and *u8
for uint8 -- so any frontend graph with a .to(int16) lands here.

One kernel carrying all of them, so a missing entry fails this file rather than
being spread over several. int8 -> int16 is the pair that first hit it.

widening conversions; exercises the *i16 and *u8 entries of the dtype tables.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def widen_dtype(i8_ptr, i16_ptr, u8_ptr, o16_ptr, o32_ptr, of32_ptr,
                    BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(o16_ptr + offs, tl.load(i8_ptr + offs).to(tl.int16))
        tl.store(o32_ptr + offs, tl.load(i16_ptr + offs).to(tl.int32))
        tl.store(of32_ptr + offs, tl.load(u8_ptr + offs).to(tl.float32))

    return widen_dtype


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda lo, hi: torch.randint(lo, hi, (N,), generator=g, dtype=torch.int32)
    # uint8 below 128, so the widening is sign-agnostic: this is about the dtype
    # tables, not about uint8 -> int8 reinterpretation.
    return {"i8": r(-128, 128).to(torch.int8),
            "i16": r(-1000, 1000).to(torch.int16),
            "u8": r(0, 128).to(torch.uint8)}


def reference(inputs):
    import torch
    return {"o16": inputs["i8"].to(torch.int16),
            "o32": inputs["i16"].to(torch.int32),
            "of32": inputs["u8"].to(torch.float32)}


SPEC = KernelSpec(
    name="widen_dtype",
    kernel=kernel,
    signature={"i8_ptr": "*i8", "i16_ptr": "*i16", "u8_ptr": "*u8",
               "o16_ptr": "*i16", "o32_ptr": "*i32", "of32_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[
        Arg("i8", IN, "int8", (N,)),
        Arg("i16", IN, "int16", (N,)),
        Arg("u8", IN, "uint8", (N,)),
        Arg("o16", OUT, "int16", (N,)),
        Arg("o32", OUT, "int32", (N,)),
        Arg("of32", OUT, "float32", (N,)),
    ],
    grid=(N // BLOCK,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
