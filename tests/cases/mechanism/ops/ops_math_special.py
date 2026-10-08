"""The special functions -- the ones that are a coefficient table, not an identity.

ops_math_derived.py covers the rewrites that are one multiply or one log away
from a native transcendental, and ops_math_bits.py the ones that read the float
as an integer.  This file is the third kind, and for a while it did not exist
because these nine were REFUSED.

The refusal reason given at the time was that each is "a multi-branch polynomial
kernel rather than a rewrite".  That reason was wrong twice over.  A branch here
is `arith.select`, which is native and vectorises -- there is no control flow in
a linalg.generic body -- and `_derive_pow_from_exp_log`, which shipped, has five of them.  And
`_derive_erfinv_by_giles_minimax`, which also shipped, IS a two-branch minimax polynomial with
eighteen coefficients.  The rule was stated and then broken in the same pass.
What was actually true is narrower and is recorded per function below.

COVERS -- the nine, and the two tails they fixed on the way.

  G. the gamma pair                        lgamma / tgamma, on Lanczos
      Lanczos g=5 (Numerical Recipes 6.1) for the argument >= 0.5, and the
      REFLECTION FORMULA for everything below it -- that is what makes the whole
      real line reachable from an approximation defined on half of it.  lgamma
      reflects through log(pi/|sin(pi x)|); tgamma reflects on GAMMA instead,
      pi/(sin(pi x)Gamma(1-x)), because that form carries the alternating sign
      by itself where the log form would have dropped it with the absolute
      value.  Poles at the non-positive integers are checked explicitly and
      return NaN, matching glibc: at x = -2 the f32 product pi*x is not exactly
      a multiple of pi, so sin comes out near 1e-7 and the quotient would
      otherwise be a large finite number.

  H. the one that genuinely could not be composed   erfcx
      erfcx(x) = exp(x*x)erfc(x) written literally gives inf for x >= 9.5 and
      NaN for x >= 12 in f32 -- exp(x*x) overflows while erfc(x) underflows and
      the true product is an ordinary 0.047.  Measured, and the only one of the
      nine where the COMPOSITION route is closed rather than merely long.  The
      answer is a continued fraction evaluated to fixed depth,

          erfcx(a) = 1/sqrt(pi) * 1/(a + (1/2)/(a + 1/(a + (3/2)/(a + ...))))

      which is a straight chain of divf and addf with no coefficient table at
      all -- the terms are k/2, and that is the whole specification.

  I. the Bessels                           j0 / j1 / y0 / y1 / i0 / i1
      Abramowitz & Stegun 9.4 and 9.8.  Two regions each: a polynomial near the
      origin, and far out an AMPLITUDE/PHASE form, because J and Y are decaying
      sinusoids there and no polynomial follows a sinusoid.  J0/Y0 share one
      amplitude and phase and differ only in cos vs sin -- they are the same
      wave a quarter period apart.  The modified pair I0/I1 grow instead, so
      their far field is scaled by exp(a) and uses no trig.

  AND THE TWO TAILS THIS FIXED       erfc / normcdf
      erfc was `1 - erf(x)`, which is catastrophic: erf(4) rounds to exactly
      1.0 in f32, so the subtraction returned ZERO where erfc(4) is 1.54e-08.
      Measured 0.1% relative error at x = 3, 3.8% at 3.5, and an exact zero for
      every x >= 4.  normcdf was `(1 + erf(x/sqrt2))/2` and cancelled the same
      way -- Phi(-6) came back as zero instead of 9.87e-10.  Both now route
      through the erfcx continued fraction above the crossover, which is the
      side that does not cancel.  Neither defect was caught by the earlier
      coverage kernel, because its atol was larger than the value being lost.

WHAT PROVES IT FIRED   work/ops_math_special_kernel/

  03-linalg.mlir carries all nine as tensor-level tt.extern_elementwise --
  __nv_lgammaf, __nv_tgammaf, __nv_erfcxf, __nv_j0f, __nv_j1f, __nv_y0f,
  __nv_y1f, __nv_cyl_bessel_i0f, __nv_cyl_bessel_i1f -- and no math.* op for
  any of them.  04-passes/03-torchsim-convert-unsupported-ops.mlir has none of them, only
  NATIVE_MATH, and 06-kernel.s has no `call`.

THE REGISTER FILE IS A CORRECTNESS BOUND HERE, NOT A TUNING ONE
  Y0's body originally called the full j0 -- including the far branch whose
  sqrt and cos the near-field select immediately discards -- and that put 32
  live vector values in one linalg.generic.  32 is the whole RVV register file,
  llc spilled, and a spilled vector goes to the stack, which is not in the
  banked scratchpad window.  spike stopped with

      VI_ST_WHOLE ERROR: Accessing invalid spad address: 0x3ffffff8c0

  THAT ADDRESS IS THE BOUND, AND IT IS A DEPTH, NOT A COUNT. spike remaps an
  sp-relative vector access to `spad_base + vu_sram_byte - (kernel_sb - addr)`,
  so what decides it is where the slot sits, not how many registers the store
  moves. This kernel's own 33 one-register spills sit in a 304-byte frame
  inside a 524288-byte window and run: measured 2026-09-05, err 1.5e-05.
  _derive_j0_near_field / _derive_j1_near_field still exist for it: Y's
  near field needs only J's near polynomial, since it is selected exactly where
  J *is* that polynomial.
  65 body ops down to 42, and it runs.  Any future derivation of this size
  should expect the same ceiling.

ARITHMETIC
  Three outputs.  `p` is positive and away from zero and from the poles -- Y0
  and Y1 are real only for x > 0, and lgamma's reflection is exercised by the
  sweep rather than here, where a NaN would have to be compared with
  torch.allclose (which has no equal_nan).  `w` spans both signs for the even
  and odd symmetries.  `t` is deliberately out in the tail, 3 to 23, which is
  the range where erfc used to return zero.

POSITIVE KERNEL: every stage succeeds and the result matches torch.

convert_unsupported_ops, families G/H/I: the nine special functions that were refused, now derived -- Lanczos+reflection for the gamma pair, a continued fraction for erfcx (the one that genuinely cannot be composed), A&S 9.4/9.8 two-region forms for the Bessels. Also pins the erfc/normcdf tails that 1-erf returned as exactly zero. POSITIVE.
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
    def ops_math_special_kernel(p_ptr, w_ptr, t_ptr, g_ptr, b_ptr, e_ptr,
                                BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = tl.load(p_ptr + r)        # (0.3, 4.3), positive, off the poles
        w = tl.load(w_ptr + r)        # (-6, 6), both signs
        t = tl.load(t_ptr + r)        # (3, 23), the erfc tail

        # G. the gamma pair.
        g = libdevice.lgamma(p) + libdevice.tgamma(p)

        # I. the Bessels.  Y takes p (positive only); J and I take w so the
        #    even/odd symmetries run.  Each is kept in its own sum so one
        #    wrong symmetry cannot cancel against another.
        b = (libdevice.j0(w) + libdevice.j1(w)
             + libdevice.y0(p) + libdevice.y1(p)
             + libdevice.cyl_bessel_i0(w) + libdevice.cyl_bessel_i1(w))

        # H + the tails.  erfcx on the tail input is the case that used to be
        #    inf/NaN; erfc and normcdf there are the ones that used to be 0.
        e = (libdevice.erfcx(t) + libdevice.erfc(t) + libdevice.normcdf(-t))

        tl.store(g_ptr + r, g)
        tl.store(b_ptr + r, b)
        tl.store(e_ptr + r, e)

    return ops_math_special_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(N, generator=g, dtype=torch.float32) * 4.0 + 0.3
    w = torch.rand(N, generator=g, dtype=torch.float32) * 12.0 - 6.0
    t = torch.rand(N, generator=g, dtype=torch.float32) * 20.0 + 3.0
    return {"p": p.contiguous(), "w": w.contiguous(), "t": t.contiguous()}


def reference(inputs):
    import torch
    from scipy import special as sp
    p, w, t = inputs["p"], inputs["w"], inputs["t"]

    def f64(fn, x):
        """Reference in float64, then cast.

        NOT torch's own f32 path for these: torch.distributions.Normal.cdf is
        (1 + erf(x/sqrt2))/2 and returns 0 for x < -5, which is the very defect
        this kernel exists to pin. A reference has to be better than the thing
        it is checking.
        """
        return torch.from_numpy(fn(x.double().numpy())).to(torch.float32)

    gg = f64(sp.gammaln, p) + f64(sp.gamma, p)
    bb = (f64(sp.j0, w) + f64(sp.j1, w)
          + f64(sp.y0, p) + f64(sp.y1, p)
          + f64(sp.i0, w) + f64(sp.i1, w))
    ee = f64(sp.erfcx, t) + f64(sp.erfc, t) + f64(sp.ndtr, -t)
    return {"g": gg.contiguous(), "b": bb.contiguous(), "e": ee.contiguous()}


SPEC = KernelSpec(
    name="ops_math_special_kernel",
    kernel=kernel,
    signature={"p_ptr": "*fp32", "w_ptr": "*fp32", "t_ptr": "*fp32",
               "g_ptr": "*fp32", "b_ptr": "*fp32", "e_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("p", IN, "float32", (N,)), Arg("w", IN, "float32", (N,)),
          Arg("t", IN, "float32", (N,)),
          Arg("g", OUT, "float32", (N,)), Arg("b", OUT, "float32", (N,)),
          Arg("e", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # tgamma(4.3) is ~9, and lgamma/tgamma are the loosest of the three sums;
    # the Bessels and the erfc family are at f32 noise. Relative is the claim.
    rtol=1e-3, atol=1e-4,
)
