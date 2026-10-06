"""The arms of `(rm + K*rn) % K == rm % K`: when it fires and when it must not.

A term of a remainder's dividend that is itself a MULTIPLE of the modulus
contributes nothing, so PtrAnalysis leaves the axis carrying it standing still
-- which is what turns a sum wrap into a wrap on one axis, the only kind
`tts.make_tptr` has a slot for. The question sits beside the two already there:
a size-1 axis holds one element, a stride-0 axis does not move, and an axis
whose walk is whole laps adds nothing the modulo can act on.

FOUR LOADS, ONE PER DECISION, and three of them are the identity NOT firing:

  load 1  (t + 24) % 8              I  24 is a multiple of 8, dropped
  load 2  (t - 3 + 16) % 8          D  `t - 3` is an arith.subi, so the sign is
                                       unknown and the guard refuses
  load 3  (t*268435456 + 16) % 8    F  the bound is 15*2^28 + 16, over i32's
                                       sign bit, so a wrapped dividend has a
                                       remainder no identity predicts
  load 4  (8*t + 16) % 8            H  EVERY term is dead; the dividend is 0

WHY D IS THE ONE THAT COSTS MONEY. `remsi` truncates toward zero: `13 rem 8` is
5 and `-3 rem 8` is -3, so dropping the 16 from `(t - 3 + 16) % 8` changes every
element with t < 3 -- and changes an ADDRESS, silently. Measured over the suite,
this guard refuses 23 times and the identity fires once, so the pass is mostly
the guard.

WHAT IS NOT HERE. The `E unbounded` cell -- provably non-negative with no
readable bound -- has exactly one door, `arith.maxsi`. A kernel through that
door miscompiles before it arrives: see ptr_maximum_clamp_dropped.py. And
`arith.remui` never appears in this corpus; Triton spells an integer remainder
signed, which is why the unsigned spelling is pinned in lit instead
(unsigned_div_rem.mlir).

One load per decision of the whole-laps rule. The D arm is the evidence: drop the sign guard and its first three elements read a different address, with no diagnostic anywhere.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_modulus_multiples_arms_kernel(x_ptr, o_ptr, N: tl.constexpr):
        t = tl.arange(0, N)
        i = (t + 24) % 8
        d = (t - 3 + 16) % 8
        f = (t * 268435456 + 16) % 8
        h = (8 * t + 16) % 8
        tl.store(o_ptr + 0 * N + t, tl.load(x_ptr + 16 + i))
        tl.store(o_ptr + 1 * N + t, tl.load(x_ptr + 16 + d))
        tl.store(o_ptr + 2 * N + t, tl.load(x_ptr + 16 + f))
        tl.store(o_ptr + 3 * N + t, tl.load(x_ptr + 16 + h))

    return ptr_modulus_multiples_arms_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(64, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    t = torch.arange(N, dtype=torch.int64)

    def rem(a, b):
        # torch.fmod IS C's truncating remainder, which is arith.remsi. A
        # float divide loses bits at 2**28 and gets the F arm wrong.
        return torch.fmod(a, b)

    def i32(a):
        return ((a + 2 ** 31) % 2 ** 32) - 2 ** 31

    x = inputs["x"]
    i = rem(t + 24, 8)
    d = rem(t - 3 + 16, 8)
    f = rem(i32(i32(t * 268435456) + 16), 8)
    h = rem(8 * t + 16, 8)
    return {"o": torch.cat([x[16 + i], x[16 + d], x[16 + f], x[16 + h]])}


SPEC = KernelSpec(
    name="ptr_modulus_multiples_arms_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (64,)),
          Arg("o", OUT, "float32", (4 * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
