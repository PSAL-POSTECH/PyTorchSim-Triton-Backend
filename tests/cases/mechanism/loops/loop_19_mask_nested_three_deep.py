"""The same loop-carried mask three levels deep -- the walk out has no depth.

loop_15 is this at two levels, which is the depth that used to be refused. This
one is the general claim: `parseLoopIterArg` finds the state op by walking the
init-arg chain outward, and nothing in that walk counts levels.

WHY THREE AND NOT TWO. Two levels reaches the state op in ONE step, so it cannot
tell a walk from a special case for "my parent". Three needs the step taken
twice, and it needs the offset and base-offset positions mapped back in through
a loop that did not make them -- which is the part a two-level fix can leave
wrong and still pass.

WHAT IT PINS, in the IR because the numbers cannot say it: the mask rides the
box, so no descriptor is indirect. Falling back to one bit per element reads the
same values and would score green.

Three nested loops carrying one mask. The width comes from outside the nest, the position from the innermost loop.
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
    def loop19_mask_nested_three_kernel(x_ptr, o_ptr, n, iters,
                                        BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        idx = tl.arange(0, BLOCK)
        for _ in range(iters):
            for _ in range(iters):
                for _ in range(iters):     # three deep: the walk takes two steps
                    v = tl.load(x_ptr + r, mask=idx < n, other=0.0)
                    tl.store(o_ptr + r, v, mask=idx < n)
                    idx += BLOCK

    return loop19_mask_nested_three_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(BLOCK, dtype=torch.float32)
    idx = torch.arange(BLOCK)
    for _ in range(ITERS):
        for _ in range(ITERS):
            for _ in range(ITERS):
                m = idx < NBOUND
                o[m] = x[:BLOCK][m]
                idx = idx + BLOCK
    return {"o": o}


SPEC = KernelSpec(
    name="loop19_mask_nested_three_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "iters": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect_ir={"04-transformed.mlir": {"offset_stride": 0}},
    extra={"scalar_args": [("n", "int32_t"), ("iters", "int32_t")],
           "scalar_values": {"n": NBOUND, "iters": ITERS}},
)
