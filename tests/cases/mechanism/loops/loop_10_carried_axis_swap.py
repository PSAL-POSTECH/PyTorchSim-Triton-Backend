"""A loop-carried tile wanted on one axis and produced on the other.

WHAT IT IS FOR. `_loop_slots` relates the four spellings of a loop-carried buffer
-- init, block argument, result, yielded value -- as ONE axis class, so a value
that goes round the loop has one lane axis and not four. Nothing in the suite
made that equation do any work: of the 158 round trips the suite emits, 155 are
outside every loop and the three inside are broadcasts, so a pass that broke the
back edge could stay green. This kernel is the missing cell.

THE SHAPE. `tl.trans` inside the loop is absorbed by absorb_layout_ops into the
dot's maps, which leaves A as [k, m]:

    linalg.matmul indexing_maps = [(d0,d1,d2) -> (d2,d0),      <- A is [k, m]
                                   (d0,d1,d2) -> (d2,d1),
                                   (d0,d1,d2) -> (d0,d1)]

so `matmul_axis(op, 0, MATMUL_K)` answers axis 0 for the carried accumulator,
while the same accumulator arrives from the previous iteration as a matmul's C,
which `matmul_axis(op, 2, MATMUL_N)` puts on axis 1. The carried value is
therefore WANTED ON 0 AND PRODUCED ON 1, and the back edge says those are the
same axis.

WHAT A PASS MAY DO WITH IT, and the three are not equally priced:

    undo it before the yield        two moves per iteration
    enter the loop on the other axis  one move, outside the loop
    carry both layouts              one move per iteration, twice the spad

A pass that answers by moving data INSIDE the loop pays the trip count; one that
answers outside pays once. Nothing here says which is right -- the kernel exists
so that a pass which quietly picks the expensive one, or which forgets the back
edge entirely and lets the two spellings drift apart, stops being invisible.

M == N ON PURPOSE. The accumulator has to come back the shape it left or the
loop does not close, and a square tile is the smallest thing that transposes and
still closes. The scale on the inputs keeps three chained matmuls inside f32.

The back edge is an equation: a loop-carried tile has one lane axis across init, block argument, result and yield. Here the body wants it on axis 0 and hands it back on axis 1, so the equation has to be paid for -- inside the loop per iteration, or once outside it.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, T = 16, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_carried_axis_swap_kernel(a_ptr, w_ptr, o_ptr,
                                      S: tl.constexpr, T: tl.constexpr):
        r = tl.arange(0, S)
        acc = tl.load(a_ptr + r[:, None] * S + r[None, :])        # (S, S)
        w = tl.load(w_ptr + r[:, None] * S + r[None, :])          # (S, S)
        for _ in range(T):
            # The transpose is absorbed into the dot's maps, so the carried tile
            # is read on axis 0 and written back on axis 1.
            acc = tl.dot(tl.trans(acc, 1, 0), w)
        tl.store(o_ptr + r[:, None] * S + r[None, :], acc)

    return loop_carried_axis_swap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (S ** 0.5)
    return {"a": torch.randn(S * S, generator=g, dtype=torch.float32) * s,
            "w": torch.randn(S * S, generator=g, dtype=torch.float32) * s}


def reference(inputs):
    acc = inputs["a"].reshape(S, S)
    w = inputs["w"].reshape(S, S)
    for _ in range(T):
        acc = acc.t() @ w
    return {"o": acc.reshape(-1)}


SPEC = KernelSpec(
    name="loop_carried_axis_swap_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "T": "constexpr"},
    constexprs={"S": S, "T": T},
    args=[Arg("a", IN, "float32", (S * S,)), Arg("w", IN, "float32", (S * S,)),
          Arg("o", OUT, "float32", (S * S,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
