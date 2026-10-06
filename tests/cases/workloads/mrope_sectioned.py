"""Qwen2-VL's M-RoPE -- one head dim cut into THREE contiguous sections, each
driven by its own position id.

    channels        0 .. ST            angle from the TEMPORAL position
                   ST .. ST+SH         angle from the HEIGHT position
                ST+SH .. HALF          angle from the WIDTH position
    and the same partition again over the second half, because `emb` is
    `cat(freqs, freqs)` and rotate_half pairs k with k + HALF.

ONE SEQUENCE, THREE COORDINATES. A VLM flattens frames, rows and columns of a
video into a single token stream, and a plain RoPE index would say that the
patch below and the patch to the right are the same distance away. Qwen2-VL
gives every token a position TRIPLE `(t, h, w)` -- text tokens get `t = h = w =
position`, so text degenerates to ordinary RoPE and the two live in the same
attention -- and then has to fit three positions into one head dim. It does it
by PARTITIONING the frequencies: `mrope_section = [16, 24, 24]` over the 64
half-channels, so the lowest 16 frequency bands carry time and the rest split
between the two spatial axes. The section widths are a budget, and giving time
the fewest bands is a statement that a video moves slowly compared to how much
detail is in a frame.

    cos = cat([m[i % 3] for i, m in enumerate(cos.split(section * 2, -1))], -1)

  is the whole thing in transformers: `cos` arrives with a leading axis of 3,
  one per position axis, and the split-and-select keeps only the section that
  belongs to each channel range.

WHAT IS STRUCTURALLY NEW HERE. `rope_2d.py` is the nearest kernel and it is a
different shape: two axes, split down the MIDDLE, and the coordinates are
DERIVED (`n // W`, `n % W`) so nothing is loaded. This one has THREE axes, at
UNEQUAL widths that are not a bisection, and the positions come from memory as
a `[3, T]` int table because a VLM's token stream interleaves text and image
and the mapping is not a formula.

  So the tile it builds is a per-lane broadcast SELECTED BY AN IN-LANE
  PREDICATE: three rank-1 position vectors, each broadcast along the channel
  axis, and a `tl.where` on the channel index picking between them. Every other
  broadcast in this suite pins one tensor to one axis; here three tensors
  pinned to the LANE axis are multiplexed by a comparison pinned to the OTHER
  axis, and the result feeds a transcendental. `softmax_causal.py` builds a
  mask that way but only ever uses it as a mask.

WHAT IS PINNED. The section widths are constants -- they are a config field in
the real model too, so nothing is lost. The three position ids are the data.

torch equivalent:
    ang  = inv_freq[None, :] * pos[section_of_channel, :].T
    o    = cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

Qwen2-VL M-RoPE: three position axes owning three CONTIGUOUS sections of the head dim, selected by an in-lane predicate over three lane-axis broadcasts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
import math

T = 32                 # tokens
D = 32                 # head dim
HALF = D // 2          # rotate_half pairs k with k + HALF
ST, SH, SW = 4, 6, 6   # the sections: t / h / w, summing to HALF.
STH = ST + SH          # ...in the same 16:24:24 ratio Qwen2-VL ships.
THETA = 10000.0
# inv_freq[k] = THETA ** (-2k/D), written as exp(-k * INVSTEP) so the jit body
# needs one transcendental and no pow.
INVSTEP = (2.0 / D) * math.log(THETA)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mrope_sectioned_kernel(x_ptr, pos_ptr, o_ptr,
                               T: tl.constexpr, D: tl.constexpr,
                               HALF: tl.constexpr, ST: tl.constexpr,
                               STH: tl.constexpr, INVSTEP: tl.constexpr):
        t = tl.arange(0, T)
        k = tl.arange(0, HALF)

        inv = tl.exp(-k.to(tl.float32) * INVSTEP)

        # The position TRIPLE, [3, T] in memory: one rank-1 tensor per axis.
        pt = tl.load(pos_ptr + t).to(tl.float32)
        ph = tl.load(pos_ptr + T + t).to(tl.float32)
        pw = tl.load(pos_ptr + 2 * T + t).to(tl.float32)

        # THE SECTIONS. Three lane-axis broadcasts multiplexed by a predicate
        # on the CHANNEL axis -- the boundaries are contiguous, which is the
        # whole difference from mrope_interleaved.py.
        kk = k[None, :]
        pos = tl.where(kk < ST, pt[:, None],
                       tl.where(kk < STH, ph[:, None], pw[:, None]))

        ang = pos * inv[None, :]
        c = tl.cos(ang)
        s = tl.sin(ang)

        # rotate_half's pairing is (k, k + HALF), so the halves are separate
        # tiles from the start -- rope.py's rule, for the same lane reason.
        base = t[:, None] * D
        x1 = tl.load(x_ptr + base + k[None, :])
        x2 = tl.load(x_ptr + base + (HALF + k)[None, :])

        tl.store(o_ptr + base + k[None, :], x1 * c - x2 * s)
        tl.store(o_ptr + base + (HALF + k)[None, :], x2 * c + x1 * s)

    return mrope_sectioned_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # [3, T]: t / h / w. Deliberately not a ramp -- a VLM's grid ids repeat and
    # jump, and a compiler that folded them into an affine index would still
    # look right on arange.
    pos = torch.randint(0, 24, (3, T), generator=g, dtype=torch.int32)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32),
            "pos": pos.reshape(-1).contiguous()}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    pos = inputs["pos"].reshape(3, T).float()
    inv = torch.exp(-torch.arange(HALF).float() * INVSTEP)[None, :]

    k = torch.arange(HALF)[None, :]
    sel = torch.where(k < ST, pos[0][:, None],
                      torch.where(k < STH, pos[1][:, None], pos[2][:, None]))
    ang = sel * inv
    c, s = ang.cos(), ang.sin()

    x1, x2 = x[:, :HALF], x[:, HALF:]
    out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=1)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="mrope_sectioned_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "pos_ptr": "*i32", "o_ptr": "*fp32",
               "T": "constexpr", "D": "constexpr", "HALF": "constexpr",
               "ST": "constexpr", "STH": "constexpr", "INVSTEP": "constexpr"},
    constexprs={"T": T, "D": D, "HALF": HALF, "ST": ST, "STH": STH,
                "INVSTEP": INVSTEP},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("pos", IN, "int32", (3 * T,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
