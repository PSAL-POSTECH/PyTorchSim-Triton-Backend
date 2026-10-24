"""The math ops built out of the BIT PATTERN, and the two that need a polynomial."""
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
