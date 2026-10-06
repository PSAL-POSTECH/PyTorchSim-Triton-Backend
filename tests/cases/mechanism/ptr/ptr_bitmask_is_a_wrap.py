"""`x & (2**k - 1)` is a WRAP, and PtrAnalysis reads it as one.

WHAT IT PINS. A power-of-two modulo written as a bit mask is what a kernel
writes when it knows the extent is a power of two -- swin_window's window roll
is `(wy*M + m + SH) & (H - 1)` on BOTH axes. With no arm for `arith.andi` the
whole access lost its structure and fell to a per-element `tts.gather`: one
index per element, every DRAM stride 0, and no burst anywhere.

`visitOperandAnd` hands it to the same wrap machinery a `%` takes. Measured on
swin_window, which does it on two axes at once:

    before   8 tts.gather + 8 tts.scatter, 0 tts.load
    after    4 tts.load + 4 tts.store + 8 tts.make_tptr, 0 gather

and across the suite 101 rank>=2 gather/scatter sites become 93.

THE SIGN IS THE SOUNDNESS. `-3 & 7` is 5 and `-3 rem 8` is -3, so the identity
holds only where the dividend cannot go negative -- `isNonNegative` is the
guard, the same one the whole-laps rule rests on. This kernel's second load has an
offset that CAN go negative and keeps its `andi`; it becomes a
`make_gather_scatter_tptr`, ONE unstructured axis with the rest still striding,
which is what upstream gives an offset it cannot read on a single axis. The
loss this identity prevents is the other one -- a bit mask on TWO axes, where
that op cannot help and the whole access falls to a per-element gather.

WHAT WOULD BREAK IT. Reading the `and` as a failure rather than as the same
unreadable operand it was before the arm existed: an `and` this cannot restate
must still leave by the index-table exit, or the second load below drops from
two rungs to three.

Both arms live: two axes restated into wraps, and one that cannot be because its dividend goes negative.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H = W = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_bitmask_is_a_wrap_kernel(x_ptr, o_ptr, p_ptr,
                                     H: tl.constexpr, W: tl.constexpr):
        r = tl.arange(0, H)
        c = tl.arange(0, W)
        # NON-NEGATIVE on both axes: the mask is a wrap and both are restated.
        ys = (r + 3) & (H - 1)
        xs = (c + 5) & (W - 1)
        tl.store(o_ptr + r[:, None] * W + c[None, :],
                 tl.load(x_ptr + ys[:, None] * W + xs[None, :]))
        # CAN GO NEGATIVE: `_nonneg` refuses, so this one keeps its bit mask.
        neg = (c - 5) & (W - 1)
        tl.store(p_ptr + c, tl.load(x_ptr + neg))

    return ptr_bitmask_is_a_wrap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].view(H, W)
    r = torch.arange(H)
    c = torch.arange(W)
    o = x[(r + 3) % H][:, (c + 5) % W]
    return {"o": o.reshape(-1), "p": inputs["x"][(c - 5) & (W - 1)]}


SPEC = KernelSpec(
    name="ptr_bitmask_is_a_wrap_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "p_ptr": "*fp32",
               "H": "constexpr", "W": "constexpr"},
    constexprs={"H": H, "W": W},
    args=[Arg("x", IN, "float32", (H * W,)), Arg("o", OUT, "float32", (H * W,)),
          Arg("p", OUT, "float32", (W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"03-ttshared.mlir": {"tts.gather": 0, "tts.make_tptr": 3,
                                    "tts.make_gather_scatter_tptr": 1}},
)
