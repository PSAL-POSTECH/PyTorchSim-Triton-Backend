"""The math ops built OUT OF the seven the vector unit has.

The machine's transcendental vocabulary is exp, erf, tanh, sin, cos, log and
atan -- seven instructions in spike's torchsim extension
(riscv/insns/torchsim_v*.h), the same seven lower_to_vcix maps to sf.vc.v.iv --
plus what RVV proper covers: absf, sqrt, rsqrt, ceil, floor, trunc, round,
roundeven, copysign, fma.

Every other elementwise math function reaches convert_unsupported_ops in one of two
shapes, and the difference is not about the function but about whether
triton-shared's converter table happened to hold the name:

    libdevice.acos   __nv_acosf   MATCHED    -> math.acos in a linalg.generic
    libdevice.erfc   __nv_erfcf   UNMATCHED  -> tt.extern_elementwise, no body

This file drives BOTH doors and every rewrite family behind them.  What it
proves is not that the arithmetic is right -- three outputs summing twenty-odd
functions would be a poor way to show that -- but that NOTHING NON-NATIVE
SURVIVES convert_unsupported_ops, which is the invariant the rest of the backend is
entitled to assume.

COVERS -- the three identity families convert_unsupported_ops covers.

  A. exp/log rescaling                    the exp2 / exp10 / log2 / log10
                  / expm1 / log1p entries of _DERIVE
      Each is one multiply away from the native op: exp2(x) = exp(x*ln2),
      log10(x) = log(x)*(1/ln10).  expm1 and log1p are the two that give up
      the near-zero cancellation their accurate versions exist for, which is
      why the input here is held away from zero -- tests/cases/ has no
      business pinning an accuracy claim the derivation does not make.

  B. trig and hyperbolic from sin/cos/exp  tan / sinh / cosh / sinpi /
                  cospi
      tan = sin/cos, sinh/cosh = (e^x -/+ e^-x)/2, and the pi-scaled pair is
      one multiply.  cosh is the one that needs no sign care; sinh's is in the
      subtraction order.

  C. the inverses, from atan / log / sqrt  asin / acos / asinh / acosh /
                  atanh / atan2
      asin(x) = atan(x/sqrt(1-x^2)) with the endpoints selected rather than
      inherited from a division by zero; acos is pi/2 minus it.  The inverse
      hyperbolics are logs of a sqrt, and asinh's uses |x| with the sign put
      back by copysign so the log's argument never cancels.  atan2 is atan
      plus a quadrant select, and it is the one whose upstream form was a
      SCALAR libm call -- see WHY THIS FILE EXISTS below.

WHAT PROVES IT FIRED   work/ops_math_derived_kernel/

  03-ttshared.mlir still carries the two upstream shapes side by side: the
  matched names as math.* inside linalg.generics, the unmatched ones as bare
  tensor-level tt.extern_elementwise with symbol = "__nv_exp10f", "__nv_log2f",
  "__nv_expm1f", "__nv_sinpif", "__nv_cospif".

  04-passes/03-torchsim-convert-unsupported-ops.mlir has ZERO tt.extern_elementwise, and
  every math op left in it is in NATIVE_MATH.  Grepping the whole of
  04-transformed.mlir for `math.` returns only names from that set -- that is the
  invariant, and it is checkable in one line.

  06-kernel.s contains NO `call` at all.  That is the point of the exercise:
  before this rewrite existed, math.atan2 went through convert-math-to-llvm to
  an LLVM intrinsic that RISC-V has no vector form for, and llc scalarised it
  into one `call atan2f` per element.

WHY THIS FILE EXISTS
  tests/cases/mechanism/ops/ops_math_elementwise.py already drove tt.extern_elementwise
  through BOTH of triton-shared's converters, and was marked `blocked` with the
  reason "libdevice.atan2 lowers to a scalar `call atan2f`, so LLVM lifts each
  element into fa0 -- and there is one scalar register file for every lane".
  That block has lifted, and it lifted because atan2 stopped being emitted as
  something the target has to have a library for.  This file is the positive
  side of that: the census file says which converter ran, this one says the
  result is executable.

ARITHMETIC
  Three sums, one per family, each term O(1)-O(10) so no term can hide inside
  another's rounding.  Inputs are chosen per family and passed separately: `p`
  is positive and away from zero (the log family and acosh want that), `u` is
  strictly inside (-1, 1) (asin, acos, atanh).  The endpoints |u| = 1 are NOT
  driven here -- they are a different claim, and the edge selects that own them
  are exercised by _derive_asin_from_atan's own select, not by a sum.

POSITIVE KERNEL: every stage succeeds and the result matches torch.

convert_unsupported_ops, identity families A/B/C: every non-native transcendental rebuilt from the seven VCIX ops plus sqrt/copysign. Drives BOTH upstream shapes -- the math.* ops triton-shared converted and the tt.extern_elementwise it did not. POSITIVE.
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
    # triton resolves names from the DEFINING function's globals, and these are
    # locals of kernel(); publish them so the @triton.jit body can see them.
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_derived_kernel(p_ptr, u_ptr, a_ptr, b_ptr, c_ptr,
                                BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = tl.load(p_ptr + r)        # (0.05, 3.05), positive
        u = tl.load(u_ptr + r)        # (-0.95, 0.95), strictly inside +-1

        # A. exp/log rescaling.  exp10 takes p*0.1 so the result stays O(10)
        #    and the sum is not dominated by one term.
        a = (libdevice.exp2(p) + libdevice.log2(p)
             + libdevice.exp10(p * 0.1) + libdevice.log10(p)
             + libdevice.expm1(p) + libdevice.log1p(p))

        # B. trig / hyperbolic.  tan takes u, not p: tan has a pole at pi/2 and
        #    p reaches 3.05, which straddles it.
        b = (libdevice.tan(u) + libdevice.sinh(p) + libdevice.cosh(p)
             + libdevice.sinpi(u) + libdevice.cospi(u))

        # C. the inverses.  acosh needs an argument >= 1, hence p + 1.
        c = (libdevice.asin(u) + libdevice.acos(u)
             + libdevice.asinh(p) + libdevice.acosh(p + 1.0)
             + libdevice.atanh(u) + libdevice.atan2(u, p))

        tl.store(a_ptr + r, a)
        tl.store(b_ptr + r, b)
        tl.store(c_ptr + r, c)

    return ops_math_derived_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"p": (torch.rand(N, generator=g, dtype=torch.float32) * 3.0
                  + 0.05).contiguous(),
            "u": (torch.rand(N, generator=g, dtype=torch.float32) * 1.9
                  - 0.95).contiguous()}


def reference(inputs):
    import torch
    p, u = inputs["p"], inputs["u"]
    a = (torch.exp2(p) + torch.log2(p)
         + torch.pow(torch.tensor(10.0), p * 0.1) + torch.log10(p)
         + torch.expm1(p) + torch.log1p(p))
    b = (torch.tan(u) + torch.sinh(p) + torch.cosh(p)
         + torch.sin(torch.pi * u) + torch.cos(torch.pi * u))
    c = (torch.asin(u) + torch.acos(u)
         + torch.asinh(p) + torch.acosh(p + 1.0)
         + torch.atanh(u) + torch.atan2(u, p))
    return {"a": a.contiguous(), "b": b.contiguous(), "c": c.contiguous()}


SPEC = KernelSpec(
    name="ops_math_derived_kernel",
    kernel=kernel,
    signature={"p_ptr": "*fp32", "u_ptr": "*fp32", "a_ptr": "*fp32",
               "b_ptr": "*fp32", "c_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("p", IN, "float32", (N,)), Arg("u", IN, "float32", (N,)),
          Arg("a", OUT, "float32", (N,)), Arg("b", OUT, "float32", (N,)),
          Arg("c", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # sinh/cosh reach ~10 at p = 3.05, so the absolute tolerance has to leave
    # room for f32 rounding at that magnitude; the relative one is what matters.
    rtol=1e-4, atol=1e-4,
)
