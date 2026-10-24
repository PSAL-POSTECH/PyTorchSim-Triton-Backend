"""Squeeze-Excitation -- channel attention, and EfficientNet's only new op.

    squeeze   z[c]   = mean over the plane of x[c, :]
    excite    g      = sigmoid(W2 @ swish(W1 @ z))
    scale     out[c] = x[c, :] * g[c]

WHAT IT IS FOR. A convolution is local, so it cannot see how channels relate
across the whole image. The pool summarises each channel to one number, a narrow
bottleneck mixes them, and the result comes back as a per-channel GATE. Hence
"channel attention".

  The bottleneck (C -> C/4 -> C) is not only about parameters. A channel has to
  pass through a narrower layer than itself, so it cannot decide its own gate
  independently of the others -- the mixing is the point.

  And the gate is a SIGMOID, not a softmax. It has to live in [0, 1] so that
  multiplying means "turn down"; a softmax would tie the channels' total to 1
  and make them compete, when several of them can be important at once.

WHY IT IS THE INTERESTING BLOCK FOR THIS BACKEND. Three different layout
questions appear in order, and one of them is the case that used to be refused:

    squeeze   fold [C, HW] along the INNER axis      each lane owns a row: free
    excite    z read as a ROW across [R, C]          needs the vector in every
                                                     lane -- a replicating DMA
              then folded along axis 1               free again
    scale     g read as a COLUMN down [C, HW]        splat inside the lane: free

AND IT DOES NOT BUILD YET. The reason is one step further back than the
broadcasts: SE both FOLDS over the channels (for the fc) and BROADCASTS over
them (for the scale), so C is axis 0 of the [C, HW] tile and axis 1 of the
[R, C] one. Those are one component, and no single lane axis means the same
thing in both:

    Fatal: constraint on axis 1 (linalg.reduce folds dim 1) cannot be recorded
           -- that axis has no image in the component's widest buffer

  A DRAM round trip is inherent here, not an implementation detail -- one tile
  wants the channels on the lanes and the other wants them in the vector, and
  DRAM is the only shared medium. pytorchsim.relayout is exactly that, and
  _relayout_conflicts already inserts one when two demands collide. What fails
  is EARLIER: the union drops the demand instead of reporting a collision, so
  the cut is never planned.

  Round-tripping `z` through DRAM by hand does not rescue it either -- that
  moves the failure to a verifier error in the second fold, so the shape is not
  simply "add a transfer".

BATCH ONE MAKES THE FCs MATVECS, so they are folds and not `tl.dot`: `W1 @ z` is
`sum_c W1[r, c] * z[c]`, which is a reduction along the inner axis of an [R, C]
tile. That also puts them on the vector lanes rather than the matrix unit,
which is right -- SE's channel counts are far too small to be worth the array.

torch equivalent: x * torch.sigmoid(W2 @ F.silu(W1 @ x.mean(-1)))[:, None]

THE GAP HAS CLOSED, and both halves of it are worth keeping here. SE folds over the channels and broadcasts over them, so C is axis 0 of one tile and axis 1 of another; the forest lost that demand until lib_lane_axis._add's lost-NOT rule, and then vectorize refused one step later with `operand [1, 32] broadcasts along dim 0 into a tile [8, 32] and it was staged one element per lane`. That message named a lane axis this tile does not have: the excitation vector comes out of pool/fc/relu/fc/sigmoid and the whole component ends up ONE_LANE, where nothing is banked and nothing needs replicating. The row is read at its own index in the one bank that holds it. What would break it again: counting a unit-axis stamp as a banking, or asking the dim-0 broadcast question of a tile with no lane axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, HW, R = 32, 64, 8            # channels, plane, squeezed width (C/4)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def squeeze_excite_kernel(x_ptr, w1_ptr, w2_ptr, o_ptr, g_ptr,
                              C: tl.constexpr, HW: tl.constexpr,
                              R: tl.constexpr):
        c = tl.arange(0, C)
        p = tl.arange(0, HW)
        r = tl.arange(0, R)
        x = tl.load(x_ptr + c[:, None] * HW + p[None, :])

        z = tl.sum(x, axis=1) / HW                       # squeeze
        w1 = tl.load(w1_ptr + r[:, None] * C + c[None, :])
        s = tl.sum(w1 * z[None, :], axis=1)              # W1 @ z
        s = s * tl.sigmoid(s)                            # swish
        w2 = tl.load(w2_ptr + c[:, None] * R + r[None, :])
        g = tl.sigmoid(tl.sum(w2 * s[None, :], axis=1))  # excite
        tl.store(g_ptr + c, g)
        tl.store(o_ptr + c[:, None] * HW + p[None, :], x * g[:, None])

    return squeeze_excite_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * HW, generator=g, dtype=torch.float32),
            "w1": torch.randn(R * C, generator=g, dtype=torch.float32) * 0.2,
            "w2": torch.randn(C * R, generator=g, dtype=torch.float32) * 0.2}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(C, HW)
    w1 = inputs["w1"].reshape(R, C)
    w2 = inputs["w2"].reshape(C, R)
    z = x.mean(-1)
    s = w1 @ z
    s = s * torch.sigmoid(s)
    g = torch.sigmoid(w2 @ s)
    return {"o": (x * g[:, None]).reshape(-1), "g": g}


SPEC = KernelSpec(
    name="squeeze_excite_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w1_ptr": "*fp32", "w2_ptr": "*fp32",
               "o_ptr": "*fp32", "g_ptr": "*fp32", "C": "constexpr",
               "HW": "constexpr", "R": "constexpr"},
    constexprs={"C": C, "HW": HW, "R": R},
    args=[Arg("x", IN, "float32", (C * HW,)), Arg("w1", IN, "float32", (R * C,)),
          Arg("w2", IN, "float32", (C * R,)),
          Arg("o", OUT, "float32", (C * HW,)), Arg("g", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
