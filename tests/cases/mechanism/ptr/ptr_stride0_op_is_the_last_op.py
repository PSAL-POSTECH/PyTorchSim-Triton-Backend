"""splat and expand_dims standing LAST in a pointer chain, which used to break it.

A pointer op that does not MOVE the address -- tt.splat, tt.broadcast,
tt.expand_dims -- adds a stride-0 axis and nothing else. PtrAnalysis models all
three in `visitOperand`, but only `rewriteOp` puts a result into `ptrMap`, and
`rewriteLoadOp` reads `ptrMap`. So one of them standing last left a hole there
and the load fell out of the structured path:

    splat        `tl.load(x_ptr + tl.zeros((N,), tl.int32))` -- triton folds the
                 zero addptr away, so tt.splat IS the last pointer op. Read one
                 scalar into every lane: sizes [N], strides [0]. It became a
                 tts.gather with an N-element index tensor for ONE address.
    expand_dims  `tl.load(p[None, :])` -- sizes [1, N], strides [0, 1]. Worse
                 than slow: the pointer reached TritonToUnstructured, whose
                 clone of splat/broadcast/expand_dims passes name, operands and
                 result type but no ATTRIBUTES, so the axis was dropped and the
                 pass built an op its own verifier refuses:

                     error: 'tt.expand_dims' op requires attribute 'axis'

tt.broadcast was the third and was fixed alone (3ebcd06, tile_bias_row_deeper_
than_one_per_lane.py). This file is the other two, and it is deliberately ONE
file: the three share a sentence, and splitting them hides that the repair is
one arm rather than three.

THE NUMBER CANNOT SAY IT for the splat half -- a gather returns the right
answer, slowly -- so both halves are pinned by expect_ir. The expand_dims half
also refuses to compile without the fix, and that is the stronger witness of
the two.

RUNNABLE. o = x[0] broadcast over N, then x itself through a unit axis.

A stride-0 pointer op standing last in the chain: splat and expand_dims. Both must reach tts.make_tptr, not the gather path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_stride0_op_is_the_last_op_kernel(x_ptr, b_ptr, o_ptr,
                                             N: tl.constexpr):
        offs = tl.arange(0, N)

        # SPLAT LAST. The zero offset folds away in the frontend, so the load's
        # pointer is a bare tt.splat -- every lane reads the same address.
        head = tl.load(x_ptr + tl.zeros((N,), tl.int32))

        # EXPAND_DIMS LAST. p is a rank-1 pointer tensor; [None, :] puts a
        # tt.expand_dims on it and the load consumes that.
        p = x_ptr + offs
        rows = tl.load(p[None, :])

        tl.store(b_ptr + offs, head)
        tl.store((o_ptr + offs)[None, :], rows)

    return ptr_stride0_op_is_the_last_op_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"b": x[0].expand(N).clone(), "o": x.clone()}


SPEC = KernelSpec(
    name="ptr_stride0_op_is_the_last_op_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("b", OUT, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference, make_inputs=make_inputs,
    expect_ir={"03-linalg.mlir": {"tts.gather": 0}},
)
