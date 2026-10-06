"""Mamba's depthwise causal conv1d, width 4, with the SiLU fused in.

    conv[c, t] = b[c] + sum_{k=0..3} w[c, k] * x[c, t - 3 + k]      x[.., <0] = 0
    y[c, t]    = conv * sigmoid(conv)

WHAT IT IS DOING IN THE BLOCK. A Mamba block is projection -> conv1d -> SiLU ->
selective scan, and `selective_scan.py` is the step after this one. The conv is
there because the recurrence is bad at exactly the thing a convolution is good
at: a short, shift-invariant, strictly local mixer. The SSM has to spend state
capacity to remember "the previous token was a space"; a width-4 filter knows it
for free. Width 4 and not 3 -- `d_conv=4` is the value every Mamba and Mamba-2
config ships, and the extra tap is cheap because the filter is depthwise.

DEPTHWISE BECAUSE THE MIXING IS ALREADY PAID FOR. The in-projection immediately
before it is a full [D, 2D] matmul and the out-projection after it is another, so
a channel-mixing conv here would be a third one buying nothing. Each of the
d_inner channels gets its own 4 taps: `groups == C`, the same extreme
`depthwise_conv.py` covers.

WHAT IS DIFFERENT FROM depthwise_conv.py, AND IT IS THE BOUNDARY. That kernel is
a VALID convolution -- it computes only the interior, so no tap ever leaves the
input and there is no mask anywhere in it. This one is CAUSAL and same-length:
the model is autoregressive, so a tap may only look backwards, and the three
zeros a width-4 causal filter needs are all on the LEFT. They cannot be dropped,
because dropping them would shorten the sequence. So

  * the tile is rank 2 -- channels on the lanes, the sequence in-lane -- rather
    than rank 3, and the shift is along the vector axis instead of a middle loop;
  * three of the four taps carry a PREDICATE, `t - 3 + k >= 0`, that varies along
    the vector axis and is constant along the lane axis, and it is a different
    predicate per tap. Four loads at four offsets with three different masks,
    all folded into one accumulator.

WHY THE SiLU IS IN HERE. `causal_conv1d_fn(..., activation="silu")` is one kernel
in mamba-ssm, and the reason is that the conv output has no other consumer -- it
goes straight into the scan's input projection. Writing it out and reading it
back would double the traffic of the widest tensor in the block for one
elementwise op. Both are stored here so the conv and the activation can be told
apart when they disagree.

NOT MODELLED: the decode-time form. At inference this conv becomes a 3-element
ring buffer per channel that is shifted once per token, which is a completely
different kernel with the same weights. This is the prefill form.

FAILS TODAY, AND NO LONGER FOR THE REASON IT USED TO. Until legalize
started charging splats and widened values their real registers, this died at
stage 6 with `llc spilled a vector register GROUP` (4 x vs4r.v). It no longer
spills; it now runs and spike stops it:

    pc 0000000000010670   va/inst 00000004d0801004
    User load segfault @ 0x00000004d0801004

with a0 = 0xd0801010. The faulting address is the right DRAM address plus
0x4_00000000 -- ONE BIT AT POSITION 34 -- so this is an address computation, not
a spill and not a register-pressure problem. The guess at the time was a lost
width in our own arithmetic; it has since been chased, and it is not (see
below). The pointer this time is the right one: the indirect MVIN takes its
offsets from a buffer this loop writes.

WHY IT DIES ON SPIKE, AND IT IS NOT THIS KERNEL'S BUG. It reached the suite as
`spike: command failed with exit code 255` and nothing else -- the harness threw
away the failing command's own output, and spike writes its whole trap report to
STDOUT, which was never carried. It now reads:

    User load segfault @ 0x00000004d0801004

READ THE ADDRESS. Strip the high bits and it is 0xd0801004 -- an
address inside the DRAM the kernel was handed. What is set on top is exactly
2^34, and 34 is 32 + log2(4), where 4 is the byte scale the DMA applies to one
indirect offset (`offset_stride` x element size). That is the whole diagnosis:

    address = base + zext_i64(i32 offset) * scale
                     ^^^^ should be sext

An indirect (gather) transfer takes its per-element offsets from an i32 tile in
the scratchpad, and the DMA adds them to the DRAM base AS UNSIGNED. A NEGATIVE
offset therefore lands 2^32 * scale above where it belongs instead of below the
base -- the right address with one spurious bit.

WHERE THE NEGATIVE OFFSET COMES FROM. Three of the four taps read `x[c, t - 3 + k]` with `t - 3 + k < 0` on the
left edge. The predicate says do not use it; the DMA still addresses it.

THE SAME FAULT, THE SAME SHAPE, IN FIVE KERNELS: causal_conv1d (bit 34),
conformer_conv (bit 34), rwkv_token_shift (bit 34), ms_deform_attn (bit 37,
offset_stride 8) and unsupported/mask_negative_offset (bit 34), which is the
minimal reproducer of it. All five are FIXED: convert_accesses_to_transfer widens every
indirect index tile to i64 (_Ctx.index_buffer), where the DMA's zero extension is
the identity and the address wraps to the one the program meant.

AND IT STILL FAILS, ON A SECOND BUG THIS ONE WAS HIDING. Nothing traps now; the
numbers are wrong instead:

    indirect_index2 = c*64 + t + 1     instead of    c*64 + t - 1
    (the simulator's own runtime/indirect_access/indirect_index2.raw)

diff exactly +2 on 48 of the 64 columns of every row. The three shifted taps each
build `t - 3 + k` as a [1, 64] row buffer banked on axis 1 -- one element per
lane, reserved as 8 -- and vectorize fills it with a vector<64> store, which
runs 224 bytes past the bank into the `c * 64` tile that sits above it in `.spad`
and adds 2 to the first 48 of its elements. Taps 0 and 1 were exact in the same
dump: the overrun only corrupts when a live buffer is downstream of it.
tile/tile_lane_axis_row_overrun.py is that bug on its own, without the filter.

Mamba's conv1d: four shifted in-lane reads, three of them masked at the left edge, into one accumulator. depthwise_conv.py is the same op with no boundary; here the boundary is the op.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, L = 64, 64                   # channels (d_inner) and sequence length
W = 4                           # d_conv -- the value Mamba actually uses


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def causal_conv1d_kernel(x_ptr, w_ptr, b_ptr, c_ptr, y_ptr,
                             C: tl.constexpr, L: tl.constexpr,
                             W: tl.constexpr):
        c = tl.arange(0, C)[:, None]
        t = tl.arange(0, L)[None, :]

        acc = tl.zeros((C, L), dtype=tl.float32)
        for k in tl.static_range(W):
            # CAUSAL: tap k reads t-(W-1)+k, so every tap but the last runs off
            # the front of the sequence and the pad is entirely on the left.
            pos = t - (W - 1) + k
            v = tl.load(x_ptr + c * L + pos,
                        mask=tl.broadcast_to(pos >= 0, (C, L)), other=0.0)
            # one scalar per channel per tap -- no channel sum anywhere
            acc += tl.load(w_ptr + c * W + k) * v

        acc += tl.load(b_ptr + tl.arange(0, C))[:, None]
        idx = c * L + t
        tl.store(c_ptr + idx, acc)
        tl.store(y_ptr + idx, acc * tl.sigmoid(acc))

    return causal_conv1d_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * L, generator=g, dtype=torch.float32),
            "w": torch.randn(C * W, generator=g, dtype=torch.float32) * 0.5,
            "b": torch.randn(C, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, C, L)
    w = inputs["w"].reshape(C, 1, W)
    # left pad of W-1 keeps the length and lets no tap see the future
    xp = torch.nn.functional.pad(x, (W - 1, 0))
    conv = torch.nn.functional.conv1d(xp, w, bias=inputs["b"], groups=C)
    return {"c": conv.reshape(-1).contiguous(),
            "y": torch.nn.functional.silu(conv).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="causal_conv1d_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "b_ptr": "*fp32",
               "c_ptr": "*fp32", "y_ptr": "*fp32", "C": "constexpr",
               "L": "constexpr", "W": "constexpr"},
    constexprs={"C": C, "L": L, "W": W},
    args=[Arg("x", IN, "float32", (C * L,)),
          Arg("w", IN, "float32", (C * W,)),
          Arg("b", IN, "float32", (C,)),
          Arg("c", OUT, "float32", (C * L,)),
          Arg("y", OUT, "float32", (C * L,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
