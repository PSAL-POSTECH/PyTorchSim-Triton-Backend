"""The math ops built out of the BIT PATTERN, and the two that need a polynomial.

Companion to ops_math_derived.py, which covers the rewrites that are one
multiply or one log away from a native transcendental.  The families here are
the ones whose derivation is not analysis at all:

  D. the float is read as an integer      _derive_signbit_from_sign_bit,
                  and the ilogb / logb / scalbn / nextafter / isnan /
                  isinf entries of _DERIVE
      An IEEE binary32 is a sign bit, an 8-bit biased exponent and a mantissa,
      and five of these are that decomposition done with arith.bitcast and
      shifts.  signbit is one shift and one mask; ilogb is the exponent field
      minus 127; scalbn goes the other way, BUILDING an exponent field and
      multiplying by it, which is why it does not go through exp2.  nextafter
      leans on the fact that the bit pattern is monotone in the magnitude for
      a fixed sign, so stepping to the adjacent float is +-1 on the integer.

      isnan and isinf are the two PREDICATES, and they are here for a second
      reason: __nv_isnanf returns an INT, but the natural way to compute it is
      arith.cmpf, which yields i1.  _emit_in_result_type closes that gap, and this
      is the kernel that makes it run.

  E. rounding and remainder               the fmod / remainder / fdim /
                  saturate / hypot / cbrt / pow entries of _DERIVE
      fmod and remainder differ ONLY in how the quotient is rounded -- trunc
      for fmod, roundeven for remainder -- and both roundings are native, so
      neither needs a division-free algorithm.  cbrt and pow are the two that
      still go through exp/log, but their difficulty is entirely in the sign
      and zero handling, which is why they are grouped here rather than with
      the identities.

  F. the erf family                       erfc / normcdf / erfinv
      erfc and normcdf are one native erf plus arithmetic.  erfinv is the only
      rewrite in the pass that is a fitted POLYNOMIAL rather than an identity:
      Giles' single-precision minimax (GTC 2010), the same rational form
      CUDA's own erfinvf uses, evaluated by Horner through math.fma.  Its
      tolerance below is looser than every other line in this suite for that
      reason, and deliberately so -- an approximation that is pinned to f32
      exactness is a test that will fail for being right.

WHAT PROVES IT FIRED   work/ops_math_bits_kernel/

  03-ttshared.mlir carries SIXTEEN unmatched symbols as tensor-level
  tt.extern_elementwise -- __nv_cbrtf, __nv_erfcf, __nv_erfinvf, __nv_fdimf,
  __nv_fmodf, __nv_hypotf, __nv_ilogbf, __nv_isinff, __nv_isnanf, __nv_logbf,
  __nv_nextafterf, __nv_normcdff, __nv_remainderf, __nv_saturatef,
  __nv_scalbnf, __nv_signbitf -- and NOT ONE math.* op for any of them.  This
  is the shape ops_math_derived.py's other door does not reach: triton-shared
  emitted an op nothing downstream can hold, and left it there.

  04-passes/03-torchsim-convert-unsupported-ops.mlir has none of them, and the only math ops
  anywhere in 04-transformed.mlir are from NATIVE_MATH.  06-kernel.s has no `call`.

  The integer-valued ones are visible in 03 as a linalg.generic whose result
  element type is i32 while its block argument is f32 -- that asymmetry is
  _emit_in_result_type, and it occurs nowhere else in the backend.

ARITHMETIC
  Three outputs, one per family, and the integer-valued members are converted
  to f32 before they join a sum so the comparison stays in one dtype.

  `p` is positive and away from zero.  `w` spans both signs.  `n` is a small
  signed integer, for scalbn's exponent argument.  NaN and infinity are NOT
  fed in here: torch.allclose in tests/runtime/verify.py has no equal_nan, so a NaN on
  both sides still reports a mismatch, and a coverage kernel that has to be
  read with that caveat in hand is worse than one that does not.  isnan/isinf
  therefore run over finite input and return all-zero, which still exercises
  the cmpf and the i1 -> i32 widening -- what they cannot claim is the true
  branch, and that is claimed by the sweep instead.

POSITIVE KERNEL: every stage succeeds and the result matches torch.

convert_unsupported_ops, families D/E/F: the bit-pattern rewrites, the two roundings behind fmod/remainder, and the erf group including the one fitted polynomial (erfinv). Every symbol here arrives as an UNMATCHED tt.extern_elementwise. POSITIVE.
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
    def ops_math_bits_kernel(p_ptr, w_ptr, n_ptr, d_ptr, e_ptr, f_ptr,
                             BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = tl.load(p_ptr + r)        # (0.05, 3.05), positive
        w = tl.load(w_ptr + r)        # (-4, 4), both signs, never 0
        n = tl.load(n_ptr + r)        # small signed int32

        # D. read the float as an integer.  The three int-valued ones are cast
        #    back to f32 so the sum has one dtype.
        d = (libdevice.signbit(w).to(tl.float32)
             + libdevice.ilogb(p).to(tl.float32)
             + libdevice.logb(p)
             + libdevice.scalbn(p, n)
             + libdevice.nextafter(p, w)
             + libdevice.isnan(w).to(tl.float32)
             + libdevice.isinf(w).to(tl.float32))

        # E. rounding and remainder.  w is never zero, so the two divisions
        #    are defined; pow takes p (positive) so the sign path is not the
        #    claim here -- ops_math_derived.py's docstring says where it is.
        e = (libdevice.fmod(w, p) + libdevice.remainder(w, p)
             + libdevice.fdim(w, p) + libdevice.saturatef(w)
             + libdevice.hypot(w, p) + libdevice.cbrt(w)
             + libdevice.pow(p, 2.5))

        # F. the erf family.  erfinv's argument must be inside (-1, 1); w/8
        #    puts it in (-0.5, 0.5), where Giles' central branch runs.
        f = (libdevice.erfc(w) + libdevice.normcdf(w)
             + libdevice.erfinv(w * 0.125))

        tl.store(d_ptr + r, d)
        tl.store(e_ptr + r, e)
        tl.store(f_ptr + r, f)

    return ops_math_bits_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(N, generator=g, dtype=torch.float32) * 3.0 + 0.05
    # (-4, -0.25) U (0.25, 4): both signs, and bounded away from zero so fmod
    # and remainder have a defined divisor and hypot never underflows.
    mag = torch.rand(N, generator=g, dtype=torch.float32) * 3.75 + 0.25
    sign = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    w = mag * sign
    n = torch.randint(-3, 4, (N,), generator=g, dtype=torch.int32)
    return {"p": p.contiguous(), "w": w.contiguous(), "n": n.contiguous()}


def reference(inputs):
    import torch
    p, w, n = inputs["p"], inputs["w"], inputs["n"]
    d = (torch.signbit(w).to(torch.float32)
         + torch.floor(torch.log2(p))            # ilogb, for a normal float
         + torch.floor(torch.log2(p))            # logb, the same value as f32
         + torch.ldexp(p, n)
         + torch.nextafter(p, w)
         + torch.isnan(w).to(torch.float32)
         + torch.isinf(w).to(torch.float32))
    # IEEE remainder is not torch.remainder (which is the floored modulus):
    # it rounds the quotient to nearest-even.
    ieee_rem = w - torch.round(w / p) * p
    e = (torch.fmod(w, p) + ieee_rem
         + (w - p).clamp(min=0.0) + w.clamp(0.0, 1.0)
         + torch.hypot(w, p) + torch.sign(w) * torch.abs(w).pow(1.0 / 3.0)
         + torch.pow(p, 2.5))
    f = (torch.erfc(w)
         + torch.distributions.Normal(0.0, 1.0).cdf(w)
         + torch.erfinv(w * 0.125))
    return {"d": d.contiguous(), "e": e.contiguous(), "f": f.contiguous()}


SPEC = KernelSpec(
    name="ops_math_bits_kernel",
    kernel=kernel,
    signature={"p_ptr": "*fp32", "w_ptr": "*fp32", "n_ptr": "*i32",
               "d_ptr": "*fp32", "e_ptr": "*fp32", "f_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("p", IN, "float32", (N,)), Arg("w", IN, "float32", (N,)),
          Arg("n", IN, "int32", (N,)),
          Arg("d", OUT, "float32", (N,)), Arg("e", OUT, "float32", (N,)),
          Arg("f", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # Loose because of ONE term: erfinv is Giles' minimax fit, good to ~4 ulp
    # relative, and pinning it tighter would be pinning the fit rather than
    # the lowering.  Every other term here is exact to f32 rounding.
    rtol=2e-3, atol=2e-4,
)
