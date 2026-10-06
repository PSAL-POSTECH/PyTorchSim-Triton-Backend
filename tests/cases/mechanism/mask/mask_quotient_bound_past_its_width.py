"""A quotient bound whose `c*K` does not fit the compare's own width.

`(x // K) < c` becomes `x < c*K`, and the new bound is minted in the compare's
type. IntegerAttr TRUNCATES SILENTLY, so a bound past that width comes back
negative and the mask inverts -- every element masked off.

torch equivalent: x (the mask is true everywhere)

WHAT BREAKS IT: guarding psto-bound-through-div's product against a hardcoded INT32_MAX instead of the compare's own width. 200*256 is inside i32 and outside i16, so the guard passes and the bound is minted as dense<-14336> -- every element masked off, all 64 wrong. The sibling is mask_bound_on_quotient.py, the i32 case the rewrite is for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 64
K = 256                 # C*K = 51200, past i16's 32767 and inside i32's
C = 200


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_quotient_bound_past_its_width_kernel(x_ptr, o_ptr,
                                                  BLOCK: tl.constexpr,
                                                  K: tl.constexpr,
                                                  C: tl.constexpr):
        i = tl.arange(0, BLOCK)
        # BOTH SIDES ARE i16 TILES, and they have to be: against an i32 literal
        # Triton widens the quotient back and the compare lands at i32, where
        # the product fits and this kernel pins nothing.
        k16 = tl.full((BLOCK,), K, tl.int16)
        c16 = tl.full((BLOCK,), C, tl.int16)
        keep = (i.to(tl.int16) // k16) < c16          # 0 < 200, true for all i
        v = tl.load(x_ptr + i, keep, other=0.0)
        tl.store(o_ptr + i, v)

    return mask_quotient_bound_past_its_width_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].clone()}


SPEC = KernelSpec(
    name="mask_quotient_bound_past_its_width_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr",
               "K": "constexpr", "C": "constexpr"},
    constexprs={"BLOCK": BLOCK, "K": K, "C": C},
    args=[Arg("x", IN, "float32", (BLOCK,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
