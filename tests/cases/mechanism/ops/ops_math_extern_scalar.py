"""An unmatched libdevice symbol applied to a SCALAR, not a tile.

ops_math_bits.py already covers `__nv_signbitf` and fifteen of its neighbours,
but every one of them arrives there on a `tensor<128xf32>`. This kernel is the
same symbol in the other shape:

    %31 = "tt.extern_elementwise"(%28) <{symbol = "__nv_signbitf"}>
              : (tensor<128xf32>) -> tensor<128xi32>      <- ops_math_bits
    %33 = "tt.extern_elementwise"(%6)  <{symbol = "__nv_signbitf"}>
              : (f32) -> i32                              <- here

Both are in 07-regularized.mlir of this kernel, which is the point: a floor-modulo
by a CONSTANT asks the sign question of the tile and of the constant, and only
the first of those has a shape.

WITHOUT `_emit_scalar_extern` this dies in stage `adapt` with

    ValueError: Cannot cast type to ShapedType (from Type(i32))

raised from `ShapedType(res.type)` -- the pass assumed every extern it had to
complete was a tile, because upstream's tensor-level converter is where they
had all come from. Verified against the tree before that arm existed.

WHERE IT CAME FROM. MoonViT's 2D rope (Kimi-VL) computes its position table as
`(flat_pos % max_width)` in f32, and Inductor writes that decomposition out
exactly as below. It is not a synthetic shape.

WHAT WOULD BREAK IT. Completing the scalar extern by SPLATTING it to a tile and
building the linalg.generic anyway: the result is correct but the kernel then
carries a vector register for one lane's worth of constant, and
`_issue_uniform_scalars` is the pass whose whole subject is that this does not
happen. Also a rewrite that folds the constant's signbit at the triton level --
the op would never reach the backend and this kernel would stop pinning
anything, so the 03 check above is part of the claim.

tt.extern_elementwise on an f32 SCALAR, not a tile: the sign question a floor-modulo asks of its constant.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128
MOD = 512.0


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_extern_scalar_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr,
                                      MOD: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)

        # Floor-modulo, written the way Inductor writes it: truncated
        # remainder, then one correction step when the signs disagree. The
        # second signbit takes the CONSTANT, so it is the scalar form.
        rem = libdevice.fmod(x, MOD)
        signs_differ = libdevice.signbit(rem) != libdevice.signbit(MOD)
        nonzero = rem != 0.0
        tl.store(o_ptr + r, tl.where(signs_differ & nonzero, rem + MOD, rem))

    return ops_math_extern_scalar_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Both signs and both sides of the modulus, so the correction step runs on
    # some lanes and not on others -- a kernel where it never fires would pass
    # with the whole `where` deleted.
    return {"x": (torch.rand(N, generator=g, dtype=torch.float32) * 4096.0) - 2048.0}


def reference(inputs):
    import torch
    return {"o": torch.remainder(inputs["x"], MOD)}


SPEC = KernelSpec(
    name="ops_math_extern_scalar_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr", "MOD": "constexpr"},
    constexprs={"BLOCK": BLOCK, "MOD": MOD},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
