"""A masked index carried through NESTED loops, folded into the box like a flat one.

WHAT IT PINS: that the loop-carried mask's bound is found however deep the nest
is. The width of a mask does not change from iteration to iteration; only where
it starts does, and the loop carries that as an iter-arg. So the width is taken
where the state was MADE -- outside the outermost loop -- and the position is
read from THIS loop, whichever level it is.

WHAT IT USED TO DO. `parseLoopIterArg` asked whether the loop had an enclosing
scf.for and gave up if it did, under a TODO that said the implementation "does
not work with nested loops". Two things stood in the way, not one: that guard,
and the fact that an inner loop's init-arg IS the enclosing loop's iter-arg, so
the state op that says what the value is has no defining op to be found at.
Walking the init-arg chain outward answers both.

WHAT IT COST, measured. The mask fell to one bit per element, which is an
address per element: this kernel emitted 2 indirect descriptors where the
identical mask in loop_11's single loop emits 0. It emits 0 now.

PtrAnalysis never had the problem -- both loops carried (tensor, offset, stride)
and the pointers were analysed all along. It was only the MASK, and that split
is what made the file worth keeping.

A loop-carried mask two levels deep, folded into the box. The width comes from outside the nest, the position from this loop.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
NBOUND = 96
ITERS = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop15_mask_nested_kernel(x_ptr, o_ptr, n, iters, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        idx = tl.arange(0, BLOCK)
        for _ in range(iters):
            for _ in range(iters):          # nested: parseLoopIterArg refuses
                v = tl.load(x_ptr + r, mask=idx < n, other=0.0)
                tl.store(o_ptr + r, v, mask=idx < n)
                idx += BLOCK

    return loop15_mask_nested_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(BLOCK, dtype=torch.float32)
    r = torch.arange(BLOCK)
    idx = torch.arange(BLOCK)
    for _ in range(ITERS):
        for _ in range(ITERS):
            m = idx < NBOUND
            o[m] = x[:BLOCK][m]
            idx = idx + BLOCK
    return {"o": o}


SPEC = KernelSpec(
    name="loop15_mask_nested_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "iters": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # THE NUMBERS ARE RIGHT EITHER WAY -- a per-element mask reads the same
    # values -- so the claim lives in the IR: the box carries it and no
    # descriptor is indirect.
    expect_ir={"04-transformed.mlir": {"offset_stride": 0}},
    extra={"scalar_args": [("n", "int32_t"), ("iters", "int32_t")],
           "scalar_values": {"n": NBOUND, "iters": ITERS}},
)
