"""lib_math: the math this machine does in one instruction -- VCIX_MATH on the vector unit,
RVV_MATH in the ISA -- and how every other function is built from those: UPSTREAM_EXPANDS by
MLIR's math-expand-ops, DERIVE here. Both keyed by the math op's own name (math.powf -> powf)."""

import math


VCIX_MATH = ("math.exp", "math.erf", "math.tanh", "math.sin", "math.cos",
             "math.log", "math.atan")

RVV_MATH = ("math.absf", "math.sqrt", "math.rsqrt", "math.ceil", "math.floor",
            "math.trunc", "math.roundeven", "math.round", "math.copysign",
            "math.fma")

NATIVE_MATH = frozenset(VCIX_MATH + RVV_MATH)

UPSTREAM_EXPANDS = ("tan", "sinh", "cosh", "acosh", "atanh", "exp2", "fpowi")


class Native:
    """Scalar float ops at the current IP, built ONLY from NATIVE_MATH and
    arith. That -- not one op per call, which poly, nan, exponent_field and
    not_ already break -- is what keeps a derivation inside the ISA."""

    def __init__(self, elem):
        self.elem = elem

    def k(self, v):
        from mlir.ir import FloatAttr
        from mlir.dialects import arith
        return arith.ConstantOp(self.elem, FloatAttr.get(self.elem, v)).result

    def ki(self, v, width=32):
        from mlir.ir import IntegerAttr, IntegerType
        from mlir.dialects import arith
        ty = IntegerType.get_signless(width)
        return arith.ConstantOp(ty, IntegerAttr.get(ty, v)).result

    def bits_width(self):
        """The width of the integer `bits` returns, read from the float type."""
        from mlir.ir import FloatType
        return FloatType(self.elem).width

    def bits(self, x):
        """float -> its integer bit pattern (vfmv/vreinterpret, free)."""
        from mlir.ir import IntegerType
        from mlir.dialects import arith
        return arith.BitcastOp(
            IntegerType.get_signless(self.bits_width()), x).result

    def unbits(self, i):
        from mlir.dialects import arith
        return arith.BitcastOp(self.elem, i).result

    def kbits(self, v):
        """An integer constant in the width `bits` returns."""
        return self.ki(v, self.bits_width())

    def sign_bit(self):
        """The sign bit's index: the MSB, in every IEEE binary format."""
        return self.kbits(self.bits_width() - 1)

    def exponent_field(self, who):
        """(shift, mask, bias) of binary32's exponent field, else a NAMED refusal:
        binary16 is (10, 15) and bfloat16 has neither, which a widened constant would
        answer plausibly and wrongly."""
        if str(self.elem) != "f32":
            raise ValueError(
                f"{who} is read out of binary32's exponent field (shift 23, "
                f"mask 0xFF, bias 127) and the element type here is "
                f"{self.elem}. binary16 puts it at (10, 15), binary64 at "
                f"(52, 1023), and bfloat16 is sixteen bits with neither, so "
                f"this is a per-FORMAT derivation and not a per-width one. "
                f"Refused rather than answered with a plausible wrong integer: "
                f"add the format's field here together with a kernel that "
                f"runs it (tests/cases/mechanism/ops/ops_math_exponent_field_f64.py).")
        return self.ki(23), self.ki(0xFF), self.ki(127)

    def itof(self, i):
        from mlir.dialects import arith
        return arith.SIToFPOp(self.elem, i).result

    def add(self, a, b):
        from mlir.dialects import arith
        return arith.AddFOp(a, b).result

    def sub(self, a, b):
        from mlir.dialects import arith
        return arith.SubFOp(a, b).result

    def mul(self, a, b):
        from mlir.dialects import arith
        return arith.MulFOp(a, b).result

    def div(self, a, b):
        from mlir.dialects import arith
        return arith.DivFOp(a, b).result

    def neg(self, a):
        from mlir.dialects import arith
        return arith.NegFOp(a).result

    def maxnum(self, a, b):
        from mlir.dialects import arith
        return arith.MaxNumFOp(a, b).result

    def minnum(self, a, b):
        from mlir.dialects import arith
        return arith.MinNumFOp(a, b).result

    def shri(self, a, b):
        from mlir.dialects import arith
        return arith.ShRSIOp(a, b).result

    def shli(self, a, b):
        from mlir.dialects import arith
        return arith.ShLIOp(a, b).result

    def andi(self, a, b):
        from mlir.dialects import arith
        return arith.AndIOp(a, b).result

    def ori(self, a, b):
        from mlir.dialects import arith
        return arith.OrIOp(a, b).result

    def addi(self, a, b):
        from mlir.dialects import arith
        return arith.AddIOp(a, b).result

    def subi(self, a, b):
        from mlir.dialects import arith
        return arith.SubIOp(a, b).result

    def cmp(self, pred, a, b):
        from mlir.dialects import arith
        from mlir.dialects.arith import CmpFPredicate
        return arith.CmpFOp(getattr(CmpFPredicate, pred), a, b).result

    def cmpi(self, pred, a, b):
        from mlir.dialects import arith
        from mlir.dialects.arith import CmpIPredicate
        return arith.CmpIOp(getattr(CmpIPredicate, pred), a, b).result

    def sel(self, c, a, b):
        from mlir.dialects import arith
        return arith.SelectOp(c, a, b).result

    def false(self):
        return self.ki(0, width=1)

    def true(self):
        return self.ki(1, width=1)

    def not_(self, c):
        """`c` is i1; xor with 1 is the negation arith has no op for."""
        from mlir.dialects import arith
        return arith.XOrIOp(c, self.true()).result

    def _m(self, cls, *a):
        from mlir.dialects import math as m
        return getattr(m, cls)(*a).result

    def exp(self, x):      return self._m("ExpOp", x)
    def log(self, x):      return self._m("LogOp", x)
    def sin(self, x):      return self._m("SinOp", x)
    def cos(self, x):      return self._m("CosOp", x)
    def tanh(self, x):     return self._m("TanhOp", x)
    def erf(self, x):      return self._m("ErfOp", x)
    def atan(self, x):     return self._m("AtanOp", x)
    def sqrt(self, x):     return self._m("SqrtOp", x)
    def rsqrt(self, x):    return self._m("RsqrtOp", x)
    def absf(self, x):     return self._m("AbsFOp", x)
    def floor(self, x):    return self._m("FloorOp", x)
    def ceil(self, x):     return self._m("CeilOp", x)
    def trunc(self, x):    return self._m("TruncOp", x)
    def roundeven(self, x): return self._m("RoundEvenOp", x)
    def round(self, x):    return self._m("RoundOp", x)
    def copysign(self, x, y): return self._m("CopySignOp", x, y)
    def fma(self, a, b, c): return self._m("FmaOp", a, b, c)

    def poly(self, x, coeffs):
        """Horner on `coeffs` highest-degree first, as a chain of fma."""
        acc = self.k(coeffs[0])
        for c in coeffs[1:]:
            acc = self.fma(acc, x, self.k(c))
        return acc

    def poly_asc(self, x, coeffs):
        """Horner on `coeffs`, LOWEST-degree first, as A&S tabulates them."""
        return self.poly(x, list(reversed(coeffs)))

    def nan(self):
        return self.div(self.k(0.0), self.k(0.0))


_LN2 = math.log(2.0)
_LN10 = math.log(10.0)
_PI = math.pi
_INF = float("inf")


def _derive_exp10_from_exp(f, x):   return f.exp(f.mul(x, f.k(_LN10)))


def _derive_log2_from_log(f, x):    return f.mul(f.log(x), f.k(1.0 / _LN2))


def _derive_log10_from_log(f, x):   return f.mul(f.log(x), f.k(1.0 / _LN10))


def _derive_sinpi_from_sin(f, x):   return f.sin(f.mul(x, f.k(_PI)))


def _derive_cospi_from_cos(f, x):   return f.cos(f.mul(x, f.k(_PI)))


def _derive_absi_from_compare(f, x):
    """|x| on an INTEGER: `x < 0 ? -x : x`, so nothing rounds."""
    from mlir.ir import IntegerType
    z = f.ki(0, IntegerType(x.type).width)
    return f.sel(f.cmpi("slt", x, z), f.subi(z, x), x)


def _derive_saturate_from_minmax(f, x): return f.minnum(f.maxnum(x, f.k(0.0)),
                                                        f.k(1.0))


def _derive_fdim_from_max(f, x, y):   return f.maxnum(f.sub(x, y), f.k(0.0))


def _derive_hypot_from_sqrt(f, x, y): return f.sqrt(f.add(f.mul(x, x),
                                                          f.mul(y, y)))


def _derive_rhypot_from_rsqrt(f, x, y): return f.rsqrt(f.add(f.mul(x, x),
                                                             f.mul(y, y)))


def _derive_log1p_from_log(f, x):
    """log1p(x) = log(1+x). NOT the accurate log1p: below |x| = 2^-24 the addition
    rounds 1+x to 1 and this answers 0 where the true value is x."""
    return f.log(f.add(f.k(1.0), x))


def _derive_expm1_from_exp(f, x):
    """expm1(x) = exp(x) - 1, with the same cancellation caveat as log1p."""
    return f.sub(f.exp(x), f.k(1.0))


def _derive_asinh_from_log(f, x):
    """asinh(x) = log(x + sqrt(x^2 + 1)), taken on |x| and signed back: the sum
    cancels for x very negative, and the odd symmetry keeps the log's arg >= 1."""
    a = f.absf(x)
    r = f.log(f.add(a, f.sqrt(f.add(f.mul(a, a), f.k(1.0)))))
    return f.copysign(r, x)


def _derive_asin_from_atan(f, x):
    """asin(x) = atan(x / sqrt(1 - x^2)), with +-pi/2 SELECTED at the endpoints
    rather than inherited from atan(+-inf), which only host libm gives."""
    t = f.sub(f.k(1.0), f.mul(x, x))
    r = f.atan(f.div(x, f.sqrt(t)))
    edge = f.copysign(f.k(_PI / 2), x)
    return f.sel(f.cmp("OLE", t, f.k(0.0)), edge, r)


def _derive_acos_from_asin(f, x):
    """acos(x) = pi/2 - asin(x)."""
    return f.sub(f.k(_PI / 2), _derive_asin_from_atan(f, x))


def _derive_atan2_from_atan_and_signbit(f, y, x):
    """atan2(y, x) = atan(y/x) plus pi where SIGNBIT(x) -- `x < 0` is false for -0.0
    and drops half an axis. On x = 0 the value is selected, not inherited."""
    base = f.atan(f.div(y, x))
    shift = f.copysign(f.k(_PI), y)
    neg_x = f.cmpi("ne", _derive_signbit_from_sign_bit(f, x), f.kbits(0))
    r = f.sel(neg_x, f.add(base, shift), base)
    at_origin = f.sel(neg_x, shift, f.copysign(f.k(0.0), y))
    on_axis = f.sel(f.cmp("OEQ", y, f.k(0.0)), at_origin,
                    f.copysign(f.k(_PI / 2), y))
    return f.sel(f.cmp("OEQ", x, f.k(0.0)), on_axis, r)


def _derive_pow_from_exp_log(f, x, y):
    """pow(x, y) = exp(y*log|x|), then the sign, the x=0 limit, the NaN and y=0. THE
    SIGN TEST IS signbit: pow(-0.0, 3) is -0.0 and `-0.0 < 0.0` is FALSE."""
    mag = f.exp(f.mul(y, f.log(f.absf(x))))
    at0 = f.sel(f.cmp("OGT", y, f.k(0.0)), f.k(0.0), f.k(_INF))
    mag = f.sel(f.cmp("OEQ", x, f.k(0.0)), at0, mag)

    yi = f.roundeven(y)
    is_int = f.cmp("OEQ", yi, y)
    half = f.mul(yi, f.k(0.5))
    is_odd = f.cmp("ONE", f.roundeven(half), half)

    nan = f.div(f.k(0.0), f.k(0.0))
    frac_over_neg = f.andi(f.cmp("OLT", x, f.k(0.0)), f.not_(is_int))
    mag = f.sel(frac_over_neg, nan, mag)

    neg_x = f.cmpi("ne", _derive_signbit_from_sign_bit(f, x), f.kbits(0))
    r = f.sel(f.andi(neg_x, f.andi(is_int, is_odd)), f.neg(mag), mag)
    return f.sel(f.cmp("OEQ", y, f.k(0.0)), f.k(1.0), r)


def _derive_cbrt_from_exp_log(f, x):
    """cbrt(x) = sign(x) * exp(log|x| / 3), with 0 selected rather than left to log(0)."""
    r = f.exp(f.mul(f.log(f.absf(x)), f.k(1.0 / 3.0)))
    r = f.copysign(r, x)
    return f.sel(f.cmp("OEQ", x, f.k(0.0)), x, r)


def _derive_rcbrt_from_cbrt(f, x):
    """rcbrt(x) = 1 / cbrt(x)."""
    return f.div(f.k(1.0), _derive_cbrt_from_exp_log(f, x))


def _derive_fmod_from_trunc(f, x, y):
    """fmod: the remainder with the quotient truncated toward zero."""
    return f.sub(x, f.mul(f.trunc(f.div(x, y)), y))


def _derive_remainder_from_roundeven(f, x, y):
    """IEEE remainder: the quotient rounded to nearest-even."""
    return f.sub(x, f.mul(f.roundeven(f.div(x, y)), y))


_ERFC_SPLIT = 2.0
_ERFCX_CF_DEPTH = 16


def _derive_erfcx_tail_by_continued_fraction(f, a):
    """erfcx(a) = 1/sqrt(pi) / (a + (1/2)/(a + 1/(a + (3/2)/(a + ...)))) for a >= 2,
    bottom-up to a fixed depth: divf and addf only, no table to transcribe wrong."""
    r = f.k(0.0)
    for k in range(_ERFCX_CF_DEPTH, 0, -1):
        r = f.div(f.k(k * 0.5), f.add(a, r))
    return f.div(f.k(1.0 / math.sqrt(math.pi)), f.add(a, r))


def _derive_erfc_from_erf_and_tail(f, x):
    """erfc(x), NOT as 1 - erf(x): erf(4) rounds to exactly 1.0 in f32, so that form
    returns zero for every x >= 4. Past the crossover erfc(a) = exp(-a*a)*erfcx(a)."""
    a = f.absf(x)
    small = f.sub(f.k(1.0), f.erf(x))
    big = f.mul(f.exp(f.neg(f.mul(a, a))),
                _derive_erfcx_tail_by_continued_fraction(f, a))
    tail = f.sel(f.cmp("OLT", x, f.k(0.0)), f.sub(f.k(2.0), big), big)
    return f.sel(f.cmp("OLE", a, f.k(_ERFC_SPLIT)), small, tail)


def _derive_erfcx_from_scaled_erfc(f, x):
    """erfcx(x) = exp(x*x)*erfc(x). THE ONE THAT CANNOT BE COMPOSED: written
    literally f32 gives inf from x >= 9.5 for a value of 0.047."""
    a = f.absf(x)
    small = f.mul(f.exp(f.mul(a, a)), f.sub(f.k(1.0), f.erf(a)))
    pos = f.sel(f.cmp("OLE", a, f.k(_ERFC_SPLIT)), small,
                _derive_erfcx_tail_by_continued_fraction(f, a))
    neg = f.sub(f.mul(f.k(2.0), f.exp(f.mul(a, a))), pos)
    return f.sel(f.cmp("OLT", x, f.k(0.0)), neg, pos)


def _derive_normcdf_from_erfc(f, x):
    """Phi(x) = erfc(-x/sqrt2)/2, NOT (1 + erf(x/sqrt2))/2: erf(-4.3) is -1.0 in f32,
    so the erf form answers zero for Phi(-6). erfc puts the tail where it survives."""
    return f.mul(f.k(0.5),
                 _derive_erfc_from_erf_and_tail(
                     f, f.div(f.neg(x), f.k(math.sqrt(2.0)))))


def _derive_ilogb_from_exponent_field(f, x):
    """The unbiased binary exponent, out of binary32's field. Zero and subnormals are
    NOT special-cased: zero answers -127 rather than C's FP_ILOGB0."""
    shift, mask, bias = f.exponent_field("ilogb")
    return f.subi(f.andi(f.shri(f.bits(x), shift), mask), bias)


def _derive_logb_from_ilogb(f, x):
    """logb(x) = ilogb(x) as a float."""
    return f.itof(_derive_ilogb_from_exponent_field(f, x))


def _derive_scalbn_from_exponent_field(f, x, n):
    """x * 2^n, by BUILDING binary32's exponent field. Valid for |n| inside the
    exponent range; beyond it the field wraps, as ldexpf's undefined land does."""
    shift, _, bias = f.exponent_field("scalbn")
    return f.mul(x, f.unbits(f.shli(f.addi(n, bias), shift)))


def _derive_signbit_from_sign_bit(f, x):
    """1 if the sign bit is set, as an integer the width of the bit pattern."""
    return f.andi(f.shri(f.bits(x), f.sign_bit()), f.kbits(1))


def _derive_isnan_from_unordered_compare(f, x):
    """isnan(x): x is unordered with itself."""
    return f.cmp("UNO", x, x)


def _derive_isinf_from_compare(f, x):
    """isinf(x): |x| == inf."""
    return f.cmp("OEQ", f.absf(x), f.k(_INF))


def _derive_nextafter_from_bit_step(f, x, y):
    """The next float from x toward y by stepping the bit pattern: +1 away from zero
    and -1 toward it. x == y gives y, and x == 0 gives y's tiniest."""
    b = f.bits(x)
    up = f.cmp("OLT", x, y)
    away = f.cmpi("eq", f.cmp("OGE", x, f.k(0.0)), up)
    stepped = f.sel(away, f.addi(b, f.kbits(1)), f.subi(b, f.kbits(1)))
    r = f.unbits(stepped)
    tiny = f.copysign(f.unbits(f.kbits(1)), y)
    r = f.sel(f.cmp("OEQ", x, f.k(0.0)), tiny, r)
    return f.sel(f.cmp("OEQ", x, y), y, r)


def _derive_erfinv_from_w(f, w, x):
    """Giles' two minimax branches split at w = 5, given w and the argument."""
    wc = f.sub(w, f.k(2.5))
    pc = f.poly(wc, [2.81022636e-08, 3.43273939e-07, -3.5233877e-06,
                     -4.39150654e-06, 0.00021858087, -0.00125372503,
                     -0.00417768164, 0.246640727, 1.50140941])
    wt = f.sub(f.sqrt(w), f.k(3.0))
    pt = f.poly(wt, [-0.000200214257, 0.000100950558, 0.00134934322,
                     -0.00367342844, 0.00573950773, -0.0076224613,
                     0.00943887047, 1.00167406, 2.83297682])
    p = f.sel(f.cmp("OLT", w, f.k(5.0)), pc, pt)
    return f.mul(p, x)


def _derive_erfinv_by_giles_minimax(f, x):
    """erfinv(x), with w = -log(1 - x*x)."""
    return _derive_erfinv_from_w(
        f, f.neg(f.log(f.sub(f.k(1.0), f.mul(x, x)))), x)


def _derive_erfcinv_from_erfinv(f, y):
    """erfcinv(y) = erfinv(1-y) with w = -log(y*(2-y)): the same value without the
    1-y cancellation. The UPPER tail, y near 2, still cancels and is not claimed."""
    w = f.neg(f.log(f.mul(y, f.sub(f.k(2.0), y))))
    return _derive_erfinv_from_w(f, w, f.sub(f.k(1.0), y))


def _derive_normcdfinv_from_erfcinv(f, p):
    """Phi^-1(p) = -sqrt2 * erfcinv(2p), NOT sqrt2*erfinv(2p-1): 2p-1 rounds to -1.0
    below p = 2^-25. The upper tail, p near 1, still cancels."""
    return f.neg(f.mul(f.k(math.sqrt(2.0)),
                       _derive_erfcinv_from_erfinv(f, f.mul(f.k(2.0), p))))


_GAMMLN_COF = (76.18009172947146, -86.50532032941677, 24.01409824083091,
               -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _derive_log_gamma_by_lanczos(f, z):
    """log Gamma(z) for z >= 0.5, Lanczos g=5. Not g=7: its coefficients run to 1259
    against a series summing to O(10), so cancellation costs more than it buys."""
    tmp = f.add(z, f.k(5.5))
    tmp = f.sub(tmp, f.mul(f.add(z, f.k(0.5)), f.log(tmp)))
    ser = f.k(1.000000000190015)
    y = z
    for c in _GAMMLN_COF:
        y = f.add(y, f.k(1.0))
        ser = f.add(ser, f.div(f.k(c), y))
    return f.add(f.neg(tmp), f.log(f.div(f.mul(f.k(_SQRT_2PI), ser), z)))


def _derive_lgamma_from_lanczos(f, x):
    """log|Gamma(x)| over the whole real line, reflecting below 0.5. BOTH branches are
    evaluated -- there is no control flow here, only select."""
    lo = f.cmp("OLT", x, f.k(0.5))
    big = f.sel(lo, f.sub(f.k(1.0), x), x)
    lg = _derive_log_gamma_by_lanczos(f, big)
    refl = f.sub(f.log(f.div(f.k(_PI), f.absf(f.sin(f.mul(f.k(_PI), x))))), lg)
    return f.sel(lo, refl, lg)


def _derive_tgamma_from_lanczos(f, x):
    """Gamma(x). The reflection is taken on GAMMA and not on its log, so the
    alternating sign rides along. The poles are checked: f32 sin(-2pi) is 1e-7."""
    lo = f.cmp("OLT", x, f.k(0.5))
    big = f.sel(lo, f.sub(f.k(1.0), x), x)
    g = f.exp(_derive_log_gamma_by_lanczos(f, big))
    refl = f.div(f.k(_PI), f.mul(f.sin(f.mul(f.k(_PI), x)), g))
    r = f.sel(lo, refl, g)
    pole = f.andi(f.cmp("OLE", x, f.k(0.0)), f.cmp("OEQ", f.roundeven(x), x))
    return f.sel(pole, f.nan(), r)


_J0_NEAR = (1.0, -2.2499997, 1.2656208, -0.3163866, 0.0444479, -0.0039444,
            0.00021)
_J0_AMP = (0.79788456, -0.00000077, -0.00552740, -0.00009512, 0.00137237,
           -0.00072805, 0.00014476)
_J0_PHASE = (-0.78539816, -0.04166397, -0.00003954, 0.00262573, -0.00054125,
             -0.00029333, 0.00013558)
_Y0_NEAR = (0.36746691, 0.60559366, -0.74350384, 0.25300117, -0.04261214,
            0.00427916, -0.00024846)
_J1_NEAR = (0.5, -0.56249985, 0.21093573, -0.03954289, 0.00443319, -0.00031761,
            0.00001109)
_J1_AMP = (0.79788456, 0.00000156, 0.01659667, 0.00017105, -0.00249511,
           0.00113653, -0.00020033)
_J1_PHASE = (-2.35619449, 0.12499612, 0.00005650, -0.00637879, 0.00074348,
             0.00079824, -0.00029166)
_Y1_NEAR = (-0.6366198, 0.2212091, 2.1682709, -1.3164827, 0.3123951,
            -0.0400976, 0.0027873)
_BESS_SPLIT = 3.0
_TWO_OVER_PI = 2.0 / math.pi


def _derive_bessel_far_field(f, a, amp_c, phase_c):
    """The far-field amplitude and phase in u = 3/a, shared by J and Y of one order:
    the SAME sinusoid a quarter period apart (A&S 9.4.3 / 9.4.6)."""
    u = f.div(f.k(_BESS_SPLIT), a)
    return f.poly_asc(u, amp_c), f.add(a, f.poly_asc(u, phase_c))


def _derive_j0_from_series_and_far_field(f, x):
    """J0: A&S 9.4.1 near the origin, 9.4.3 beyond 3. Even, so it runs on |x|."""
    a = f.absf(x)
    near = f.poly_asc(f.mul(f.div(a, f.k(_BESS_SPLIT)),
                            f.div(a, f.k(_BESS_SPLIT))), _J0_NEAR)
    amp, th = _derive_bessel_far_field(f, a, _J0_AMP, _J0_PHASE)
    far = f.mul(f.div(amp, f.sqrt(a)), f.cos(th))
    return f.sel(f.cmp("OLT", a, f.k(_BESS_SPLIT)), near, far)


def _derive_j1_from_series_and_far_field(f, x):
    """J1: A&S 9.4.4 near, 9.4.6 far. ODD, and the sign is a NEGATION, not a
    copysign: copysign is a different number everywhere past the first root."""
    a = f.absf(x)
    near = f.mul(a, f.poly_asc(f.mul(f.div(a, f.k(_BESS_SPLIT)),
                                     f.div(a, f.k(_BESS_SPLIT))), _J1_NEAR))
    amp, th = _derive_bessel_far_field(f, a, _J1_AMP, _J1_PHASE)
    far = f.mul(f.div(amp, f.sqrt(a)), f.cos(th))
    r = f.sel(f.cmp("OLT", a, f.k(_BESS_SPLIT)), near, far)
    return f.sel(f.cmp("OLT", x, f.k(0.0)), f.neg(r), r)


def _derive_j0_near_field(f, x):
    """J0's near-field polynomial ALONE, so Y0 can reuse it below 3 where they agree."""
    return f.poly_asc(f.mul(f.div(x, f.k(_BESS_SPLIT)),
                            f.div(x, f.k(_BESS_SPLIT))), _J0_NEAR)


def _derive_j1_near_field(f, x):
    """J1's near-field polynomial alone, times x."""
    return f.mul(x, f.poly_asc(f.mul(f.div(x, f.k(_BESS_SPLIT)),
                                     f.div(x, f.k(_BESS_SPLIT))), _J1_NEAR))


def _derive_y0_from_series_and_far_field(f, x):
    """Y0: A&S 9.4.2 near, 9.4.3 far. The near field carries a log singularity, so Y0
    is real only for x > 0 and everything at or below zero is NaN."""
    near = f.add(f.mul(f.mul(f.k(_TWO_OVER_PI),
                             f.log(f.mul(x, f.k(0.5)))),
                       _derive_j0_near_field(f, x)),
                 f.poly_asc(f.mul(f.div(x, f.k(_BESS_SPLIT)),
                                  f.div(x, f.k(_BESS_SPLIT))), _Y0_NEAR))
    amp, th = _derive_bessel_far_field(f, x, _J0_AMP, _J0_PHASE)
    far = f.mul(f.div(amp, f.sqrt(x)), f.sin(th))
    r = f.sel(f.cmp("OLT", x, f.k(_BESS_SPLIT)), near, far)
    return f.sel(f.cmp("OGT", x, f.k(0.0)), r, f.nan())


def _derive_y1_from_series_and_far_field(f, x):
    """Y1: A&S 9.4.5 near, 9.4.6 far. 9.4.5 tabulates x*Y1(x), so the near field is
    divided back out. NaN at and below zero, like Y0."""
    near = f.div(
        f.add(f.mul(f.mul(f.k(_TWO_OVER_PI), f.log(f.mul(x, f.k(0.5)))),
                    f.mul(x, _derive_j1_near_field(f, x))),
              f.poly_asc(f.mul(f.div(x, f.k(_BESS_SPLIT)),
                               f.div(x, f.k(_BESS_SPLIT))), _Y1_NEAR)), x)
    amp, th = _derive_bessel_far_field(f, x, _J1_AMP, _J1_PHASE)
    far = f.mul(f.div(amp, f.sqrt(x)), f.sin(th))
    r = f.sel(f.cmp("OLT", x, f.k(_BESS_SPLIT)), near, far)
    return f.sel(f.cmp("OGT", x, f.k(0.0)), r, f.nan())


_I0_NEAR = (1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768,
            0.0045813)
_I0_FAR = (0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281,
           -0.02057706, 0.02635537, -0.01647633, 0.00392377)
_I1_NEAR = (0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532,
            0.00032411)
_I1_FAR = (0.39894228, -0.03988024, -0.00362018, 0.00163801, -0.01031555,
           0.02282967, -0.02895312, 0.01787654, -0.00420059)
_I_SPLIT = 3.75


def _derive_bessel_i_far_field(f, a, coeffs):
    """A&S 9.8. The modified Bessels GROW, so the far field is exp(a)/sqrt(a) times a
    polynomial in 3.75/a -- no trig, and the split is at 3.75."""
    return f.mul(f.div(f.exp(a), f.sqrt(a)),
                 f.poly_asc(f.div(f.k(_I_SPLIT), a), coeffs))


def _derive_i0_from_series_and_far_field(f, x):
    """I0: A&S 9.8.1 near, 9.8.2 far. Even, so it runs on |x|."""
    a = f.absf(x)
    t = f.mul(f.div(a, f.k(_I_SPLIT)), f.div(a, f.k(_I_SPLIT)))
    return f.sel(f.cmp("OLT", a, f.k(_I_SPLIT)),
                 f.poly_asc(t, _I0_NEAR),
                 _derive_bessel_i_far_field(f, a, _I0_FAR))


def _derive_i1_from_series_and_far_field(f, x):
    """I1: A&S 9.8.3 near, 9.8.4 far. ODD, so the sign is a negation."""
    a = f.absf(x)
    t = f.mul(f.div(a, f.k(_I_SPLIT)), f.div(a, f.k(_I_SPLIT)))
    r = f.sel(f.cmp("OLT", a, f.k(_I_SPLIT)),
              f.mul(a, f.poly_asc(t, _I1_NEAR)),
              _derive_bessel_i_far_field(f, a, _I1_FAR))
    return f.sel(f.cmp("OLT", x, f.k(0.0)), f.neg(r), r)


DERIVE = {
    "exp10":      (1, _derive_exp10_from_exp),
    "log2":       (1, _derive_log2_from_log),
    "log10":      (1, _derive_log10_from_log),
    "log1p":      (1, _derive_log1p_from_log),
    "expm1":      (1, _derive_expm1_from_exp),
    "asin":       (1, _derive_asin_from_atan),
    "acos":       (1, _derive_acos_from_asin),
    "asinh":      (1, _derive_asinh_from_log),
    "erfc":       (1, _derive_erfc_from_erf_and_tail),
    "erfcx":      (1, _derive_erfcx_from_scaled_erfc),
    "cbrt":       (1, _derive_cbrt_from_exp_log),
    "rcbrt":      (1, _derive_rcbrt_from_cbrt),
    "sinpi":      (1, _derive_sinpi_from_sin),
    "cospi":      (1, _derive_cospi_from_cos),
    "saturate":   (1, _derive_saturate_from_minmax),
    "normcdf":    (1, _derive_normcdf_from_erfc),
    "normcdfinv": (1, _derive_normcdfinv_from_erfcinv),
    "erfinv":     (1, _derive_erfinv_by_giles_minimax),
    "erfcinv":    (1, _derive_erfcinv_from_erfinv),
    "ilogb":      (1, _derive_ilogb_from_exponent_field),
    "logb":       (1, _derive_logb_from_ilogb),
    "lgamma":     (1, _derive_lgamma_from_lanczos),
    "tgamma":     (1, _derive_tgamma_from_lanczos),
    "j0":         (1, _derive_j0_from_series_and_far_field),
    "j1":         (1, _derive_j1_from_series_and_far_field),
    "y0":         (1, _derive_y0_from_series_and_far_field),
    "y1":         (1, _derive_y1_from_series_and_far_field),
    "i0":         (1, _derive_i0_from_series_and_far_field),
    "i1":         (1, _derive_i1_from_series_and_far_field),
    "signbit":    (1, _derive_signbit_from_sign_bit),
    "isnan":      (1, _derive_isnan_from_unordered_compare),
    "isinf":      (1, _derive_isinf_from_compare),
    "absi":       (1, _derive_absi_from_compare),
    "atan2":      (2, _derive_atan2_from_atan_and_signbit),
    "powf":       (2, _derive_pow_from_exp_log),
    "fmod":       (2, _derive_fmod_from_trunc),
    "remainder":  (2, _derive_remainder_from_roundeven),
    "hypot":      (2, _derive_hypot_from_sqrt),
    "rhypot":     (2, _derive_rhypot_from_rsqrt),
    "fdim":       (2, _derive_fdim_from_max),
    "scalbn":     (2, _derive_scalbn_from_exponent_field),
    "nextafter":  (2, _derive_nextafter_from_bit_step),
}
