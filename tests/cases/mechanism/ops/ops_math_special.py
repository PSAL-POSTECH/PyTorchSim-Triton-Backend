"""The special functions -- the ones that are a coefficient table, not an identity."""
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
    """float64 torch.special, then cast; tgamma from math.gamma.

    NOT the f32 path: torch.distributions.Normal.cdf is (1 + erf(x/sqrt2))/2 and
    returns 0 for x < -5, the very defect this kernel pins."""
    import math
    import torch
    p, w, t = (inputs[k].double() for k in ("p", "w", "t"))
    s = torch.special
    tg = torch.tensor([math.gamma(v) for v in p.tolist()], dtype=torch.float64)
    gg = s.gammaln(p) + tg
    bb = (s.bessel_j0(w) + s.bessel_j1(w) + s.bessel_y0(p) + s.bessel_y1(p)
          + s.i0(w) + s.i1(w))
    ee = s.erfcx(t) + s.erfc(t) + s.ndtr(-t)
    return {"g": gg.float().contiguous(), "b": bb.float().contiguous(),
            "e": ee.float().contiguous()}


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
