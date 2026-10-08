"""Two loop-carried splat states multiplied together -- mulState's scalar*scalar arm.

`o1 * o2` where both operands are splats of a program id and BOTH are advanced
inside the loop. mulState then sees two states that each have a `scalar` and no
structured dimension, and has to emit an arith.muli for the product.

THE GRID MUST BE 2-D. The kernel is launched at grid=(2, 2) and o1/o2 are built
from program_id(0) and program_id(1). With one program id, or with either
operand loop-invariant, triton's own reorder_broadcast pass sinks the multiply
onto the scalars before the linalg lowering ever sees it, mulState is handed a
constant, and this arm never runs. Both states being genuine loop-carried block
arguments is what prevents that.

WHAT IT PINS. That the product is of TWO ITER-ARGS. The scf.for carries SEVEN
iter-args here -- (tensor, index, index) for each of the two splat states plus
the f32 accumulator -- and the multiply inside the body has to read the
loop-carried offset of each side:

    %21 = arith.muli %arg9, %arg10 : index      both loop-carried
    %22 = tts.make_tptr %{...} offsets: [%21] ...

WHAT IT USED TO PIN, and why the number below is the whole point.
`reconcileLoopPtrState` rebuilt a rank>0 state's offsets, strides and base
offset from the loop's iter-args and left `scalar` holding the PRE-LOOP value.
`scalar` is set only where the state describes ONE number, so the state then
named that number twice, from two different iterations -- and mulState scales
one side by the OTHER side's `scalar`:

    %19 = arith.muli %arg4, %3 : index          iter-arg times pre-loop init

so the address was (px+k)*py instead of (px+k)*(py+2k). Measured bases
0,1,2,3 against 0,3,10,21; the baseline recorded it as an 11.9956 numerical
error under `extra["upstream_miscompile"]` until 2026-09-06.

POSITIVE. Two loop-carried splat states multiplied together, the only shape in which mulState sees a scalar on BOTH sides -- and the one that catches a scalar left behind by the loop.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
K = 4
BLOCK = 128
GX = 2
GY = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop10_splat_muli_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                 BLOCK: tl.constexpr, GY: tl.constexpr):
        px = tl.program_id(0)
        py = tl.program_id(1)
        r = tl.arange(0, BLOCK)
        o1 = tl.zeros((BLOCK,), dtype=tl.int32) + px      # splat state #1
        o2 = tl.zeros((BLOCK,), dtype=tl.int32) + py      # splat state #2
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            acc += tl.load(x_ptr + o1 * o2 + r)           # mulState(scalar, scalar)
            o1 = o1 + 1
            o2 = o2 + 2
        tl.store(o_ptr + (px * GY + py) * BLOCK + r, acc)

    return loop10_splat_muli_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(GX * GY * BLOCK, dtype=torch.float32)
    for px in range(GX):
        for py in range(GY):
            acc = torch.zeros(BLOCK, dtype=torch.float32)
            for k in range(K):
                base = (px + k) * (py + 2 * k)
                acc += x[base:base + BLOCK]
            lin = px * GY + py
            o[lin * BLOCK:(lin + 1) * BLOCK] = acc
    return {"o": o}


SPEC = KernelSpec(
    name="loop10_splat_muli_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr",
               "BLOCK": "constexpr", "GY": "constexpr"},
    constexprs={"K": K, "BLOCK": BLOCK, "GY": GY},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (GX * GY * BLOCK,)),
    ],
    grid=(GX, GY),
    reference=reference,
    make_inputs=make_inputs,
    # THE NUMBER SAYS IT HERE, unusually: a stale scalar is a different
    # address, and the reference walks the intended (px+k)*(py+2k). The op
    # count says WHICH shape -- one muli inside the loop body, not two.
    expect_ir={"03-linalg.mlir": {"tts.make_gather_scatter_tptr": 0}},
)
