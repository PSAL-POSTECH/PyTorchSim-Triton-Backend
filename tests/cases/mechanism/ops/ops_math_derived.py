"""The math ops built OUT OF the seven the vector unit has."""
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
