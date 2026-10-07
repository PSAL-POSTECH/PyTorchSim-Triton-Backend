"""A reduce whose RESULT type moves when the reshape above it is absorbed.

WHAT IT PINS. `absorb_layout_ops` folds a reshape into a `linalg.reduce` by renaming
its `dimensions`, and for a long time it did that only where the reduce's RESULT
came out the shape it already had. Everything else it declined -- an axis that
appears or goes away always moves the result, and moving it makes the init, the
result type and every reader of the result part of ONE rewrite. That is the
group, and declining it left the reshape standing below the pass that declares
the layout family may not survive it (rule 8).

    tensor.expand_shape %reduced [[0, 1]] output_shape [1, 32] : 32 -> 1x32
    linalg.reduce ins(%expanded : 1x32) outs(%6 : 1) dimensions = [1]

    absorbed:  linalg.reduce ins(%reduced : 32) outs(%6 : f32) dimensions = [0]
               the `linalg.fill` behind %6 and its `tensor.empty` retyped with
               it, and the generic below -- which read the scalar with
               `(d0) -> (0)` -- now reading it with `(d0) -> ()`

`extra["forbids"]` IS WHAT MAKES THIS FAIL BEFORE AND PASS AFTER. Without the
group rewrite the kernel still compiles and still answers correctly -- the
`tensor.expand_shape` survives into bufferize and the passes below cope -- so
nothing but the declaration turns it red, and a kernel that passes either way
pins nothing (rule 2). Measured on the tree before the change:

    absorb_layout_ops left 1 op(s) it declared would not survive it:
    tensor.expand_shape x1. First at %expanded = tensor.expand_shape %reduced
    [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>

and after it, all eight stages green with max_abs_err 9.5e-06.

WHAT WOULD BREAK IT.

  * refusing a move with a GAP at the reduce (`_reduce_takes` used to do that
    first thing) or refusing one that reorders the survivors
  * deriving the result move and not WALKING it: the init here is a
    `linalg.fill`, so a walk that stops at the producer of the init leaves
    `outs(%6 : 1)` typed for a shape the reduce no longer has -- which
    `linalg.reduce` rejects at verify rather than silently
  * `fit_layout_to_lanes` losing the rank-zero spelling of a one-bank row again.
    The generic below the reduce reads the scalar with `(d0) -> ()` here where
    a rank-1 init gives it `(d0) -> (0)`, and only the second was answered
    until e8722f5 -- which is why this absorption was derived long before it
    could ship.

THE SHAPE IS dl/convnextv2_grn'S TWO FOLDS WITH EVERYTHING ELSE TAKEN AWAY. The
tile is loaded transposed so the first fold runs along the inner axis; that
leaves a rank-1 value living on what was the lane axis, and the only way to fold
IT in lane is to present it as a `[1, C]` row -- which is where the
`expand_shape` comes from. So this is not a contrivance: it is how a triton
program folds along its own lane axis, and the second fold's `[1]` result is
then read back across the whole tile.

Two folds and not one: the first is what makes the second's input a value the
backward walk cannot restate, so the absorption has to go FORWARDS into the
reduce. Written as a single `tl.sum(tl.load(...)[None, :], axis=1)` the tile
comes straight off a descriptor and absorb_layout_ops takes it backwards instead, absorbing
on the tree before the change too.

torch equivalent: gx = x.reshape(HW, C).t().sum(1); gx - gx.sum()

A linalg.reduce whose result type moves when the expand_shape above it is absorbed. Fails on a tree that declines that group, with absorb_layout_ops naming the tensor.expand_shape it left standing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, HW = 32, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_result_moves_kernel(x_ptr, o_ptr, C: tl.constexpr,
                                   HW: tl.constexpr):
        c = tl.arange(0, C)
        s = tl.arange(0, HW)
        # Stored [HW, C], read as [C, HW]: the transpose is in the addressing,
        # so the first fold is over the INNER axis.
        xt = tl.load(x_ptr + s[None, :] * C + c[:, None])   # [C, HW]
        gx = tl.sum(xt, axis=1)                             # [C]
        # [None, :] is the expand_shape: C -> 1xC, so this fold is over the
        # inner axis too and its result is [1] rather than a scalar.
        gm = tl.sum(gx[None, :], axis=1)                    # [1]
        tl.store(o_ptr + c, gx - gm)

    return reduce_result_moves_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(HW * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    gx = inputs["x"].reshape(HW, C).t().contiguous().sum(dim=1)
    return {"o": gx - gx.sum()}


SPEC = KernelSpec(
    name="reduce_result_moves_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "C": "constexpr",
               "HW": "constexpr"},
    constexprs={"C": C, "HW": HW},
    args=[Arg("x", IN, "float32", (HW * C,)),
          Arg("o", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
