"""RWKV-6/7's token shift -- a one-token lookback fused into four mixes.

    xx      = x[t-1] - x[t]                  x[-1] = 0
    x_r     = x + xx * mu_r
    x_w     = x + xx * mu_w                  and the same for k and v
                                             mu_* are per-channel, learned

so each of R, W, K, V is projected from its own learned interpolation between
this token and the previous one. `fused_addcmul_rwkv7` in the reference repo is
literally this: one read of x, one read of the shifted x, and six `addcmul`s.

WHY A MODEL WOULD DO THIS. RWKV has no attention and, in the time-mixing block,
no convolution either -- the only mechanism that lets a channel see anything but
the current token is the WKV recurrence, which is diagonal and cannot compare
adjacent positions. The token shift is the whole of RWKV's local receptive
field, and giving R, W, K and V DIFFERENT mixing ratios is what lets the key look
mostly at the previous token while the value looks mostly at this one. It is one
learned scalar per channel per gate, and it is a large part of why RWKV needs no
positional encoding.

WHAT IS STRUCTURALLY DIFFERENT FROM causal_conv1d.py, WHICH IS THE NEAREST THING.
Both are a short causal lookback. Everything else is opposite:

  * THE SHIFT IS ON THE LANE AXIS, NOT THE VECTOR AXIS. The Mamba conv puts
    channels on the lanes and the sequence in-lane, so its taps are shifts along
    the vector axis. RWKV's tensors are [T, C] and the mix is per channel across
    the whole row, so the natural tile is tokens on the lanes and channels
    in-lane -- and then the lookback is a shift along the LANE axis. That is the
    same arithmetic reaching for a different port: an in-lane shift is a slide
    within a register, a lane shift is a different set of banks. Neither is a
    permute here -- both come out as a second load at a different base, which is
    exactly what makes this worth having beside the conv rather than instead
    of it.

  * THE WEIGHT DOES NOT MULTIPLY A TAP, IT INTERPOLATES. A conv accumulates
    `sum_k w_k x_{t-k}`; this is `x + mu*(x_{t-1} - x)`, an affine blend that is
    the identity at mu = 0 and a pure shift at mu = 1. The difference is formed
    FIRST and shared by all four gates, which is the entire reason to fuse: `xx`
    is computed once and consumed four times, so the fused form reads x twice
    and writes four tensors where the unfused form reads x eight times.

  * FOUR OUTPUTS FROM ONE LOADED TILE, all at the same depth in the chain --
    unlike `fused_add_rmsnorm.py`, whose two outputs straddle a reduction. There
    is no reduction anywhere in this kernel. It is pure bandwidth: the pressure
    is on holding `x` and `xx` live while four independent per-channel vectors
    stream past them.

THE BOUNDARY IS ONE ELEMENT AND IT IS AT THE FRONT. Token 0 has no predecessor,
so `xx[0] = -x[0]` and the first row of every output is `x*(1-mu)`. That single
masked row is the only control flow in the kernel, and it is the thing an
implementation that shifts with a wraparound gets wrong while every other row
still matches.

`xx` IS AN OUTPUT. All four mixes are the same tensor scaled differently, so if
the shift is wrong they are all wrong together and the four give no more
information than one. `xx` isolates the shift from the interpolation.

WHY IT DIES ON SPIKE, AND IT IS NOT THIS KERNEL'S BUG. It reached the suite as
`spike: command failed with exit code 255` and nothing else -- the harness threw
away the failing command's own output, and spike writes its whole trap report to
STDOUT, which was never carried. It now reads:

    User load segfault @ 0x00000004d0800f90

READ THE ADDRESS. Strip the high bits and it is 0xd0800f90 -- an
address inside the DRAM the kernel was handed. What is set on top is exactly
2^34, and 34 is 32 + log2(4), where 4 is the byte scale the DMA applies to one
indirect offset (`offset_stride` x element size). That is the whole diagnosis:

    address = base + zext_i64(i32 offset) * scale
                     ^^^^ should be sext

An indirect (gather) transfer takes its per-element offsets from an i32 tile in
the scratchpad, and the DMA adds them to the DRAM base AS UNSIGNED. A NEGATIVE
offset therefore lands 2^32 * scale above where it belongs instead of below the
base -- the right address with one spurious bit.

WHERE THE NEGATIVE OFFSET COMES FROM. The shift itself: `x[t - 1]` with `x[-1] = 0`. One negative index, the
first row, and it is the whole kernel.

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

A one-row lookback along the LANE axis, fanned into four interpolations. causal_conv1d.py shifts along the vector axis and accumulates; this shifts across banks and blends. Exact arithmetic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, C = 32, 32                   # tokens on the lanes, channels in-lane


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rwkv_token_shift_kernel(x_ptr, mur_ptr, muw_ptr, muk_ptr, muv_ptr,
                                xx_ptr, xr_ptr, xw_ptr, xk_ptr, xv_ptr,
                                T: tl.constexpr, C: tl.constexpr):
        t = tl.arange(0, T)[:, None]            # lane axis
        c = tl.arange(0, C)[None, :]            # in-lane axis
        off = t * C + c
        rc = tl.arange(0, C)

        x = tl.load(x_ptr + off)
        # THE SHIFT, along the LANE axis: one row back, zero before the start.
        prev = tl.load(x_ptr + off - C,
                       mask=tl.broadcast_to(t > 0, (T, C)), other=0.0)
        xx = prev - x                           # formed once, used four times
        tl.store(xx_ptr + off, xx)

        # Four interpolations, one per gate, each with its own per-channel ratio.
        tl.store(xr_ptr + off, x + xx * tl.load(mur_ptr + rc)[None, :])
        tl.store(xw_ptr + off, x + xx * tl.load(muw_ptr + rc)[None, :])
        tl.store(xk_ptr + off, x + xx * tl.load(muk_ptr + rc)[None, :])
        tl.store(xv_ptr + off, x + xx * tl.load(muv_ptr + rc)[None, :])

    return rwkv_token_shift_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # mu is a sigmoid-parameterised ratio in the model, so keep it in (0, 1).
    def mu():
        return torch.rand(C, generator=g, dtype=torch.float32)
    return {"x": torch.randn(T * C, generator=g, dtype=torch.float32),
            "mur": mu(), "muw": mu(), "muk": mu(), "muv": mu()}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, C)
    prev = torch.cat([torch.zeros(1, C, dtype=torch.float32), x[:-1]], dim=0)
    xx = prev - x
    out = {"xx": xx.reshape(-1).contiguous()}
    for name, key in (("xr", "mur"), ("xw", "muw"),
                      ("xk", "muk"), ("xv", "muv")):
        out[name] = (x + xx * inputs[key][None, :]).reshape(-1).contiguous()
    return out


SPEC = KernelSpec(
    name="rwkv_token_shift_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "mur_ptr": "*fp32", "muw_ptr": "*fp32",
               "muk_ptr": "*fp32", "muv_ptr": "*fp32", "xx_ptr": "*fp32",
               "xr_ptr": "*fp32", "xw_ptr": "*fp32", "xk_ptr": "*fp32",
               "xv_ptr": "*fp32", "T": "constexpr", "C": "constexpr"},
    constexprs={"T": T, "C": C},
    args=[Arg("x", IN, "float32", (T * C,)),
          Arg("mur", IN, "float32", (C,)),
          Arg("muw", IN, "float32", (C,)),
          Arg("muk", IN, "float32", (C,)),
          Arg("muv", IN, "float32", (C,)),
          Arg("xx", OUT, "float32", (T * C,)),
          Arg("xr", OUT, "float32", (T * C,)),
          Arg("xw", OUT, "float32", (T * C,)),
          Arg("xk", OUT, "float32", (T * C,)),
          Arg("xv", OUT, "float32", (T * C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
