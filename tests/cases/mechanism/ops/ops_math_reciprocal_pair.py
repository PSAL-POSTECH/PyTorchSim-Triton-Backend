"""The two derivations nothing in the corpus was reaching -- rcbrt and rhypot.

FOUND BY COUNTING, not by a failure.  `_DERIVE` has 49 entries and a walk of the
487-kernel corpus reaches 47 of them: every other derivation is pinned by some
kernel that would go red without it, and these two were pinned by nothing.  Rule
2 says a capability ships with a kernel that fails without it, so this is the
kernel that was owed.

WHY THEY ARE A PAIR.  Both are a RECIPROCAL the target computes without a
divide.  rhypot is one rsqrt of the sum of squares, which is native; rcbrt is
one divide over the cbrt derivation, which is exp/log.  They sit at opposite
ends of the same idea, and a route that lost either would still pass every other
math kernel here.

WHAT WOULD BREAK IT.  Delete "rcbrt" or "rhypot" from _DERIVE and this kernel
stops at convert_unsupported_ops, naming the symbol -- that is the refusal arm
for a __nv_* symbol neither upstream nor this pass knows.  Route them through a
divide instead of rsqrt and the answer stays close but the assembly grows one
vfdiv per element, which is what the derivation exists to avoid.

TWO OUTPUTS AND NOT ONE SUM.  ops_math_special.py records why: summing two
derivations lets a wrong sign in one cancel against the other, and a kernel that
can pass while half broken pins nothing.

ARITHMETIC.  `w` spans both signs and stays off zero -- rcbrt(-8) is -0.5, so
the sign has to survive the exp/log route, and rcbrt(0) is an infinity that
torch.allclose would not compare.  `x`/`y` are away from zero for the same
reason on the hypotenuse.

POSITIVE KERNEL: every stage succeeds and the result matches torch.

convert_unsupported_ops: the two _DERIVE entries no other kernel reaches -- rcbrt (one divide over the exp/log cbrt) and rhypot (one native rsqrt of the sum of squares). Found by counting the table against the corpus, not by a failure. POSITIVE.
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
    def ops_math_reciprocal_pair_kernel(w_ptr, x_ptr, y_ptr, c_ptr, h_ptr,
                                        BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        w = tl.load(w_ptr + r)        # (-6, -0.5) U (0.5, 6), both signs
        x = tl.load(x_ptr + r)        # (0.5, 4.5)
        y = tl.load(y_ptr + r)        # (0.5, 4.5)

        tl.store(c_ptr + r, libdevice.rcbrt(w))
        tl.store(h_ptr + r, libdevice.rhypot(x, y))

    return ops_math_reciprocal_pair_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    #: Off zero on both sides: rcbrt(0) is an infinity and allclose has no
    #: equal_nan, so the pole is the sweep's business rather than this file's.
    mag = torch.rand(N, generator=g, dtype=torch.float32) * 5.5 + 0.5
    sign = torch.where(torch.arange(N) % 2 == 0, 1.0, -1.0).to(torch.float32)
    x = torch.rand(N, generator=g, dtype=torch.float32) * 4.0 + 0.5
    y = torch.rand(N, generator=g, dtype=torch.float32) * 4.0 + 0.5
    return {"w": (mag * sign).contiguous(), "x": x.contiguous(),
            "y": y.contiguous()}


def reference(inputs):
    """In float64, then cast. torch has no rcbrt or rhypot, so both are spelled
    out -- and spelling them in f32 would compare one rounding against another."""
    import torch
    w = inputs["w"].double()
    x, y = inputs["x"].double(), inputs["y"].double()
    cbrt = torch.sign(w) * w.abs().pow(1.0 / 3.0)
    return {"c": (1.0 / cbrt).to(torch.float32).contiguous(),
            "h": (1.0 / torch.hypot(x, y)).to(torch.float32).contiguous()}


SPEC = KernelSpec(
    name="ops_math_reciprocal_pair_kernel",
    kernel=kernel,
    signature={"w_ptr": "*fp32", "x_ptr": "*fp32", "y_ptr": "*fp32",
               "c_ptr": "*fp32", "h_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("w", IN, "float32", (N,)), Arg("x", IN, "float32", (N,)),
          Arg("y", IN, "float32", (N,)),
          Arg("c", OUT, "float32", (N,)), Arg("h", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # rcbrt goes through exp(log|w|/3) and carries that route's f32 error;
    # rhypot is one native rsqrt and sits at noise. Relative is the claim.
    rtol=1e-4, atol=1e-6,
)
