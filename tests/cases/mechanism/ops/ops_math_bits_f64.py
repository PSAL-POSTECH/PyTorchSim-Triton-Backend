"""nextafter on f64: the step-and-compare bit derivations were never binary32's,
only their i32 constants were -- measured before, `arith.addi (i64, i32)`.
Breaks it: f.ki(1) instead of f.kbits(1), or driving a VCIX op at e64.

convert_unsupported_ops, POSITIVE. The bit-pattern derivations read their integer width from the float type on one side (bits()) and wrote i32 on the other (ki); on f32 the two agree by coincidence and on nothing else. MEASURED BEFORE on this kernel: MLIRError "'arith.addi' op requires the same type for all operands and results", note (i64, i32) -> i64 -- naming neither nextafter nor f64. The same shape on f64 atan2 was 'arith.cmpi' (i64, i32) -> i1. AFTER: PASS, max_abs_err 0. What this pins is that STEP-and- COMPARE was never binary32-specific: nextafter leans only on the pattern being monotone in the magnitude for a fixed sign, true of every IEEE binary format. The exponent-field half is a different claim and is refused -- see ops_math_exponent_field_f64.py. f64 is runnable here because spike has elen:64 and nothing in this derivation is transcendental; a derivation reaching math.exp could not be driven this way, spike answers "[torchsim_vexp] Unsupported vsew=64 (only e16/e32 supported)", which is why the kernel is nextafter and not atan2. Lanes 0 and 1 take x = 0, the smallest- subnormal arm.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_bits_f64_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        y = tl.load(y_ptr + r)
        tl.store(o_ptr + r, libdevice.nextafter(x, y))

    return ops_math_bits_f64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(N, generator=g, dtype=torch.float64) * 4.0 - 2.0
    y = torch.rand(N, generator=g, dtype=torch.float64) * 4.0 - 2.0
    x[0] = 0.0
    x[1] = 0.0
    y[0] = 1.0
    y[1] = -1.0
    return {"x": x.contiguous(), "y": y.contiguous()}


def reference(inputs):
    import torch
    return {"o": torch.nextafter(inputs["x"], inputs["y"]).contiguous()}


SPEC = KernelSpec(
    name="ops_math_bits_f64_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp64", "y_ptr": "*fp64", "o_ptr": "*fp64",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float64", (N,)), Arg("y", IN, "float64", (N,)),
          Arg("o", OUT, "float64", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
