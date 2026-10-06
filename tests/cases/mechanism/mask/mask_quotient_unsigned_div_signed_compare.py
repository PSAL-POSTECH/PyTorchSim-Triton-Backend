"""An UNSIGNED quotient compared SIGNED -- the two halves disagree about the
dividend, so the bound cannot travel through the division.

foldBoundThroughDiv reads `(x // K) < c` as `x < c*K`.  That is one statement
about x only when the division and the compare read x the same way.  Here they
do not:

    idx = -1     unsigned -> 4294967295, // 4 -> 1073741823, `< 8` FALSE
                 rewritten to `idx slt 32` -> `-1 < 32` TRUE

so every lane holding a negative selector flips from `other` to a real load.
_SOUND is a set of PAIRS for exactly this reason, and until this file nothing
in the suite held either mismatched pair -- measured, the pass fires ten times
over the whole suite and widening _SOUND to all four pairs changed nothing.

WHAT WOULD BREAK IT: adding ("arith.divui", _SLT) to _SOUND, or writing the
guard as two lists (a division list and a predicate list) and taking their
product.

torch equivalent: x where (sel viewed as unsigned) // 4 < 8, else -1.

foldBoundThroughDiv's sound PAIRS. divui + slt is the combination a product of two lists would fold and this one must not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64
K = 4
C = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_quotient_unsigned_div_signed_compare_kernel(
            sel_ptr, x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
            C: tl.constexpr):
        i = tl.arange(0, N)
        sel = tl.load(sel_ptr + i)
        # divui, then a SIGNED compare of its result -- the mismatched pair.
        q = (sel.to(tl.uint32) // K).to(tl.int32)
        v = tl.load(x_ptr + i, q < C, other=-1.0)
        tl.store(o_ptr + i, v)

    return mask_quotient_unsigned_div_signed_compare_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    sel = torch.arange(N, dtype=torch.int32)
    sel[::2] = -1                      # half the lanes read unsigned-huge
    return {"sel": sel,
            "x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    u = inputs["sel"].to(torch.int64) & 0xFFFFFFFF          # unsigned view
    q = (u // K).to(torch.int32)
    keep = q < C
    return {"o": torch.where(keep, inputs["x"], torch.full((N,), -1.0))}


SPEC = KernelSpec(
    name="mask_quotient_unsigned_div_signed_compare_kernel", kernel=kernel,
    signature={"sel_ptr": "*i32", "x_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "K": "constexpr", "C": "constexpr"},
    constexprs={"N": N, "K": K, "C": C},
    args=[Arg("sel", IN, "int32", (N,)), Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
