"""A bf16 softmax of 8 rows of 384 in a 512 block, in Inductor's form: the tile is loaded as bf16 and
widened to f32 for the max and the exp. The v6e unpacks a bf16 vreg where it is used; here too the
widening is made in each reader (the max's in-lane step, the exp) as it reads the bf16 tile, and no
f32 copy of the tile is written. Breaks if the cast is a walk of its own again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, B = 8, 384, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_bf16_tile_cast_in_its_readers_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
        c = tl.arange(0, B)[None, :]
        mask = c < C
        at = tl.arange(0, R)[:, None] * C + c
        x = tl.load(x_ptr + at, mask, other=0.0).to(tl.float32)
        m = tl.max(tl.where(mask, x, float("-inf")), axis=1)[:, None]
        s = tl.sum(tl.where(mask, tl.exp(x - m), 0.0), axis=1)[:, None]
        y = tl.load(x_ptr + at, mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + at, (tl.exp(y - m) / s).to(tl.bfloat16), mask)

    return reduce_bf16_tile_cast_in_its_readers_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": torch.softmax(x.float(), 1).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="reduce_bf16_tile_cast_in_its_readers_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*bf16", "R": "constexpr", "C": "constexpr", "B": "constexpr"},
    constexprs={"R": R, "C": C, "B": B},
    args=[Arg("x", IN, "bfloat16", (R, C)), Arg("o", OUT, "bfloat16", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-2, atol=1e-2,
    # THE CAST IS THE CLAIM: two widenings, each in the walk that reads the bf16 tile, and six stores.
    expect_ir={"08-transformed.mlir": {"arith.extf": 2, "affine.vector_store": 6}},
)
