"""A transpose whose only reader is a SLICE, absorbed into the slice's arrays.

WHAT IT PINS. `absorb_layout_ops`'s slice arm, forward (`_absorb_into_slice`). The
transposed value is read by nothing but `tensor.extract_slice`, and before that
arm existed `_plan_into` had no case for it: it returned None, the forward walk
refused, the backward walk had nothing to reach either, and the transpose rode
down to `vectorize` -- which refuses any linalg op still touching a
scratchpad buffer.

WHY A SLICE CAN TAKE ONE, and why the module used to say it could not. Its
docstring blamed "no reader states which iteration axis each of its axes is,
and no producer does either". A slice states it THREE times over:

    tensor.extract_slice %t[0, 0] [8, 1] [1, 1]
                          offsets   sizes  strides      one entry per axis

which is exactly the shape of a transfer's `dram_stride`, and
`_absorb_into_transfer` already says why a permutation is free against that:
"reading it in another order IS the transpose". So the rewrite is that one,
three times, through the same `_renamed`.

WHAT MAKES IT END THERE. This slice is rank-reducing, and the axes it drops are
the ones the permutation moves, so the RESULT type does not change -- nothing
downstream has to be told and the absorption stops at the slice. When the result
would move, the arm declines rather than retyping and recursing, because that is
the group turn `fit_layout_to_lanes` owns (rule 4, rule 8).

WITHOUT THE ARM. Measured on the tree before it:

    adapt: NotVectorisable: 'linalg.transpose' still touches a scratchpad
    buffer after banking

WHAT WOULD BREAK IT. Making the slice keep both moved axes -- take [8, 2]
instead of [8, 1] -- so the result type turns with the permutation. The arm
declines that on purpose and the kernel goes back to red, which is the honest
outcome rather than a half-absorption.

A slice states offsets, sizes and strides per axis, so a permutation is absorbed into it the way one is absorbed into a dram_stride.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 8, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_trans_through_slice_kernel(x_ptr, o_ptr, R: tl.constexpr,
                                       C: tl.constexpr):
        rr = tl.arange(0, R)
        rc = tl.arange(0, C)
        x = tl.load(x_ptr + rr[:, None] * C + rc[None, :])        # (R, C)
        xt = tl.trans(x, 1, 0)                                     # (C, R)
        # One column of the transposed tile: the slice keeps R and drops C, so
        # the permutation moves only axes the result does not have.
        col = tl.sum(xt * (rc[:, None] == 0), axis=0)              # (R,)
        tl.store(o_ptr + rr, col)

    return ops_trans_through_slice_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    # Row 0 of the transposed tile is column 0 of the original.
    return {"o": inputs["x"].reshape(R, C)[:, 0].contiguous()}


SPEC = KernelSpec(
    name="ops_trans_through_slice_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R * C,)), Arg("o", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
