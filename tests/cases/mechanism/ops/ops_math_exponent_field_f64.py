"""ilogb on f64 is REFUSED BY NAME: the exponent field is per-FORMAT (f32 23/127,
f64 52/1023, bf16 sixteen bits with neither) and only f32 has a kernel.
Breaks it: widening the constants to the type's width while keeping 23 and 127.

convert_unsupported_ops, REFUSAL. ilogb, logb and scalbn need WHERE the exponent field is: binary16 (10, 15), binary32 (23, 127), binary64 (52, 1023), bfloat16 (7, 127) -- four derivations, not one derivation with a width, and bf16 is why the guard keys on the TYPE and not on width (bf16 and f16 are both sixteen bits and share no field boundary). A widened constant that kept 23 and 127 would not fail; it would return an ordinary-looking integer that is wrong for every input. MEASURED BEFORE: the run also stopped, but with "error: 'arith.cmpi' op requires all operands to have the same type" from inside convert_unsupported_ops own canonicalize, naming neither ilogb nor f64. AFTER: _Fatal naming both. This file does NOT merely pin "it stops": the moment somebody makes the constants follow the width it COMPILES, and run.py prints EXPECTED A REFUSAL BUT EVERY STAGE PASSED, which is a failure. TO LIFT IT: give _Native.exponent_field the (shift, mask, bias) for the format and turn this kernel positive with a torch reference.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_exponent_field_f64_kernel(p_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = tl.load(p_ptr + r)
        tl.store(o_ptr + r, libdevice.ilogb(p).to(tl.float64))

    return ops_math_exponent_field_f64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"p": (torch.rand(N, generator=g, dtype=torch.float64) * 3.0
                  + 0.05).contiguous()}


def reference(inputs):
    import torch
    return {"o": torch.floor(torch.log2(inputs["p"])).contiguous()}


SPEC = KernelSpec(
    name="ops_math_exponent_field_f64_kernel",
    kernel=kernel,
    signature={"p_ptr": "*fp64", "o_ptr": "*fp64", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("p", IN, "float64", (N,)), Arg("o", OUT, "float64", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("transform", Reason.MATH_UNLOWERED),
    expect_why="ilogb is read out of binary32's exponent field (shift 23, "
               "bias 127) and binary64 puts it at shift 52, bias 1023. Only "
               "f32 has a kernel, so every other format is refused by name "
               "rather than answered with a plausible wrong integer.",
)
