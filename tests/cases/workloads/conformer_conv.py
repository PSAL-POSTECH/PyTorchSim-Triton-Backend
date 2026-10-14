"""NeMo's Conformer convolution module, whole: pointwise up, GLU, causal
depthwise, BatchNorm, Swish, pointwise down.

nemo/collections/asr/parts/submodules/conformer_modules.py, in order:

    x = pointwise_conv1(x)          # [C, L] -> [2C, L]
    x = glu(x, dim=channels)        # a * sigmoid(b), the two halves
    x = depthwise_conv(x)           # groups == C, kernel 31, causal padding
    x = batch_norm(x)
    x = activation(x)               # Swish
    x = pointwise_conv2(x)          # [C, L] -> [C, L]

This is one of the two things a Conformer block has that a transformer block does
not (the other is `rel_shift_attn.py`), and it is why Conformer beat pure
attention on ASR: attention is permutation-equivariant and has to LEARN that
audio is local and ordered, while a depthwise kernel of 31 frames -- about
300 ms at NeMo's subsampled 40 ms rate -- knows it. The block sandwiches this
between two half-step feed-forwards, so it runs once per layer, 16 to 24 times.

WHAT IS DIFFERENT FROM depthwise_conv.py AND causal_conv1d.py. Both of those ARE
the middle line of this kernel. What is here and not there is everything wrapped
around it, and the wrapping is not separable:

  THE GATE COMES FIRST. `pointwise_conv1` produces 2C channels and GLU consumes
  them in pairs, half as value and half as gate, so the depthwise conv never
  sees the projection's output -- it sees a product of two halves of it. The
  split is expressed in the WEIGHT'S ADDRESSING (`w1[0:C]` and `w1[C:2C]` are two
  loads and two `tl.dot`s) because a register tile has no row slice; one matmul
  to `[2C, L]` followed by a slice is not something this language can say. Two
  dots against the same right operand, and only one of them goes through a
  sigmoid.

  THE NORM COMES AFTER. `causal_conv1d.py` follows its conv with SiLU and stops.
  Here the conv result is normalised per channel first, and the normalisation is
  four rank-1 vectors reduced to two -- `s = gamma * rsqrt(var + eps)`,
  `b = beta - mean * s` -- so there is an rsqrt on a rank-1 tensor whose result
  broadcasts across a `[C, L]` tile. A transcendental at rank 1 feeding a rank-2
  broadcast is a shape nothing else here has; `bn_relu_add.py` takes `s` and `b`
  ALREADY FOLDED, because in a ResNet nothing sits between the BN and the conv
  it belongs to. Here something does, and leaving the fold undone is the point:
  `s` is per channel and the depthwise tap is per channel, so BN folds backwards
  into the tap and the whole normalisation can vanish. A compiler that finds that
  has done something real; it cannot look for it if the kernel arrives folded.

  THE ACTIVATION IS SWISH, `x * sigmoid(x)`, the same function `causal_conv1d.py`
  calls SiLU. Two names, one op, and both papers are cited in this suite.

THE GATED ACTIVATION GOES THROUGH MEMORY, and it is the depthwise conv that puts
it there. The conv reads `g[c, t - (K-1) + k]` for K different k, and `g` is a
value in registers with no dynamic index. `causal_conv1d.py` gets away with
shifted loads because its input is a POINTER; here the input is the result of a
matmul and a sigmoid, so it is stored (it is an output anyway) and read back at K
shifted offsets with the mask `t - (K-1) + k >= 0`. That store-to-load is the
real cost of fusing a conv behind a matmul, and it is visible here rather than
hidden in a library.

CAUSAL, because streaming ASR is the reason to use a Conformer over a
transformer at all: NeMo's `conv_context_size = [K-1, 0]` puts the whole
receptive field in the past so the same weights serve offline and streaming.
The kernel size is 5 here and 31 in the config; the boundary behaviour is
identical and the IR is readable.

WHY IT DIES ON SPIKE, AND IT IS NOT THIS KERNEL'S BUG. It reached the suite as
`spike: command failed with exit code 255` and nothing else -- the harness threw
away the failing command's own output, and spike writes its whole trap report to
STDOUT, which was never carried. It now reads:

    User load segfault @ 0x00000004d080b980

READ THE ADDRESS. Strip the high bits and it is 0xd080b980 -- an
address inside the DRAM the kernel was handed. What is set on top is exactly
2^34, and 34 is 32 + log2(4), where 4 is the byte scale the DMA applies to one
indirect offset (`offset_stride` x element size). That is the whole diagnosis:

    address = base + zext_i64(i32 offset) * scale
                     ^^^^ should be sext

An indirect (gather) transfer takes its per-element offsets from an i32 tile in
the scratchpad, and the DMA adds them to the DRAM base AS UNSIGNED. A NEGATIVE
offset therefore lands 2^32 * scale above where it belongs instead of below the
base -- the right address with one spurious bit.

WHERE THE NEGATIVE OFFSET COMES FROM. The depthwise causal branch reads `x[c, t - (W-1) + k]`, negative for the
first W-1 positions of every sequence.

THE SAME FAULT, THE SAME SHAPE, IN FIVE KERNELS: causal_conv1d (bit 34),
conformer_conv (bit 34), rwkv_token_shift (bit 34), ms_deform_attn (bit 37,
offset_stride 8) and unsupported/mask_negative_offset (bit 34), which is the
minimal reproducer of it. convert_accesses_to_transfer already clamps the DMA's BOX so
an out-of-tile position never reads the index buffer past its extent; nothing
clamps the index VALUE. Clamping it into [0, numel) at the point the index tile
is built would be sound -- an in-bounds gather is unaffected, and an
out-of-bounds one was already reading undefined data that the mask discards --
and it is the fix all five want. Not attempted here: it changes the IR of every
indirect transfer in the suite.

GREEN SINCE 2026-09-05, err 7.15e-07, AND THE FIX IS NOT IN THIS KERNEL'S SHAPE.
`fuse` was absorbing this kernel's elementwise producers INTO a
`linalg.matmul` -- three matmuls in, two out, and a generic with
`iterator_types = [parallel, parallel, reduction, reduction]` where the third
had been. That generic is still a contraction (d2 runs 32 and the maps spell
`out[d0,d1] = sum_d2 A[d0,d2] * B[d2,d1]`), but it no longer carries the NAME
this backend reaches the matrix unit through, so vectorize refused it --
correctly, since the alternative is the vector lanes doing the array's work.

A matmul is not an elementwise op and so is not a participant in a pass called
fuse. It is fenced for the duration of the upstream fusion now.

The Conformer's non-attention half: a GLU built from two dots on two row ranges of one weight, a causal depthwise conv whose input is a stored register tile, a rank-1 rsqrt broadcast over the tile, and a Swish, in one kernel. `g` is the gate's output, so a mismatch before and after the conv separate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, L = 32, 64        # d_model, frames
K = 5                # depthwise kernel (NeMo: 31)
EPS = 1e-5           # BatchNorm1d default


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def conformer_conv_kernel(x_ptr, w1_ptr, wd_ptr, mean_ptr, var_ptr,
                              gam_ptr, beta_ptr, w2_ptr, g_ptr, y_ptr,
                              C: tl.constexpr, L: tl.constexpr,
                              K: tl.constexpr, EPS: tl.constexpr):
        c = tl.arange(0, C)[:, None]
        t = tl.arange(0, L)[None, :]
        d = tl.arange(0, C)[None, :]

        x = tl.load(x_ptr + c * L + t)                       # [C, L]

        # POINTWISE UP + GLU. The 2C output channels are never materialised:
        # the split is the two row ranges of w1, so it costs addressing, not a
        # slice. Value half times sigmoid of gate half.
        wv = tl.load(w1_ptr + c * C + d)                     # rows 0..C-1
        wg = tl.load(w1_ptr + (C + c) * C + d)               # rows C..2C-1
        g = tl.dot(wv, x) * tl.sigmoid(tl.dot(wg, x))
        tl.store(g_ptr + c * L + t, g)

        # CAUSAL DEPTHWISE. The taps read g at K shifted times, so g comes back
        # from memory; the pad is entirely on the left and the last tap is the
        # only unmasked one.
        acc = tl.zeros((C, L), dtype=tl.float32)
        for k in tl.static_range(K):
            pos = t - (K - 1) + k
            v = tl.load(g_ptr + c * L + pos,
                        mask=tl.broadcast_to(pos >= 0, (C, L)), other=0.0)
            acc += tl.load(wd_ptr + c * K + k) * v

        # BATCHNORM, unfolded: an rsqrt at rank 1, broadcast over the tile.
        ch = tl.arange(0, C)
        s = tl.load(gam_ptr + ch) * tl.rsqrt(tl.load(var_ptr + ch) + EPS)
        b = tl.load(beta_ptr + ch) - tl.load(mean_ptr + ch) * s
        h = acc * s[:, None] + b[:, None]

        # SWISH, then pointwise back down.
        sw = h * tl.sigmoid(h)
        w2 = tl.load(w2_ptr + c * C + d)
        tl.store(y_ptr + c * L + t, tl.dot(w2, sw))

    return conformer_conv_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    return {"x": r(C * L),
            "w1": r(2 * C * C) * (C ** -0.5),
            "wd": r(C * K) * 0.5,
            "mean": r(C) * 0.1,
            "var": torch.rand(C, generator=g, dtype=torch.float32) + 0.5,
            "gam": r(C) * 0.5 + 1.0,
            "beta": r(C) * 0.1,
            "w2": r(C * C) * (C ** -0.5)}


def reference(inputs):
    import torch
    F = torch.nn.functional
    x = inputs["x"].reshape(C, L)
    w1 = inputs["w1"].reshape(2 * C, C)
    up = w1 @ x                                        # [2C, L]
    g = F.glu(up, dim=0)                               # a * sigmoid(b)
    dw = F.conv1d(F.pad(g[None], (K - 1, 0)),
                  inputs["wd"].reshape(C, 1, K), groups=C)[0]
    s = inputs["gam"] * (inputs["var"] + EPS).rsqrt()
    b = inputs["beta"] - inputs["mean"] * s
    h = dw * s[:, None] + b[:, None]
    y = inputs["w2"].reshape(C, C) @ F.silu(h)
    return {"g": g.reshape(-1).contiguous(), "y": y.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="conformer_conv_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w1_ptr": "*fp32", "wd_ptr": "*fp32",
               "mean_ptr": "*fp32", "var_ptr": "*fp32", "gam_ptr": "*fp32",
               "beta_ptr": "*fp32", "w2_ptr": "*fp32", "g_ptr": "*fp32",
               "y_ptr": "*fp32", "C": "constexpr", "L": "constexpr",
               "K": "constexpr", "EPS": "constexpr"},
    constexprs={"C": C, "L": L, "K": K, "EPS": EPS},
    args=[Arg("x", IN, "float32", (C * L,)),
          Arg("w1", IN, "float32", (2 * C * C,)),
          Arg("wd", IN, "float32", (C * K,)),
          Arg("mean", IN, "float32", (C,)),
          Arg("var", IN, "float32", (C,)),
          Arg("gam", IN, "float32", (C,)),
          Arg("beta", IN, "float32", (C,)),
          Arg("w2", IN, "float32", (C * C,)),
          Arg("g", OUT, "float32", (C * L,)),
          Arg("y", OUT, "float32", (C * L,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
