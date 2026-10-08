"""`tl.cumprod(x, axis=0)` on a RANK-2 tile -- the one scan shape _unroll_scan
refuses, and the refusal is an implementation limit rather than a machine one.

WHAT IT PINS. `_unroll_scan` opens a prefix scan into log2(n) Hillis-Steele steps
and declines any scan whose axis is not the innermost:

    if axis != rank - 1:
        raise _Fatal("a prefix scan is only laid out along the innermost axis")

The reason given for it in the module docstring is that the shift runs ALONG the
scanned axis, so that axis has to stay inside a bank -- but that constraint is
already carried, on every step it emits, as `torchsim.no_lane_axis = axis`. What
actually forces the innermost axis is three hard-coded `-1`s in the emitter:
`n = shape[-1]`, `wide = shape[:-1] + [pad + n]`, and the slice offset going
into `[0] * (rank - 1) + [offset]`.

WHY THIS SHAPE AND NOT tl.cumsum. `ttx.cumsum` cannot reach the refusal: its
verifier already requires `axis == rank-1` (the linalg lowering CumSum.cpp:63-70), so
a rank-2 axis-0 cumsum is rejected upstream and never arrives. A NON-ADD
combiner does not become a `ttx.cumsum` at all -- isCumSum() matches a lone
arith.AddFOp/AddIOp -- so `tl.cumprod` stays a `tt.scan` with its region and its
`axis` intact, and lands here. That is the only door to this refusal.

AND THAT IS WHAT WAS DONE. The three now read `axis`: `n = shape[axis]`, the
pad going into the same slot (`shape[:axis] + [pad + n] + shape[axis + 1:]`),
and the offset into `static_offsets[axis]`. So this file no longer pins a limit
-- it pins the general emitter, and it is here because it FAILED before that
change, with the message quoted above.

WHAT WOULD BREAK IT. Widening the refusal to cover the cumsum path too: that one
is upstream's and refusing it here would hide a change in the linalg lowering behind our
own message.

A rank-2 scan on axis 0 -- the shape _unroll_scan used to refuse. It is the only door to that refusal: ttx.cumsum is stopped upstream by its own verifier (axis == rank-1), so a non-add combiner is what carries an outer-axis scan this far. Pins that the ladder is built along `axis` rather than along the last dim.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scan_outer_axis_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                   N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        idx = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + idx)
        # axis 0 of a RANK-2 tile: not the innermost, and cumprod keeps this a
        # tt.scan rather than a ttx.cumsum, so it reaches _unroll_scan.
        tl.store(o_ptr + idx, tl.cumprod(x * 0.01 + 1.0, axis=0))

    return ops_scan_outer_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": (inputs["x"] * 0.01 + 1.0).cumprod(dim=0)}


SPEC = KernelSpec(
    name="ops_scan_outer_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
