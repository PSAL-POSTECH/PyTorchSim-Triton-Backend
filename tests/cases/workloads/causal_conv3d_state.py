"""The video VAE's CausalConv3d -- a 3-D convolution padded on ONE SIDE OF TIME
ONLY, with the boundary carried in a persistent state across temporal tiles.

    o[t, y, x] = b + sum_{kt,kh,kw} w[kt,kh,kw] * f[t + kt - (KT-1), y+kh, x+kw]

    f[s] =  x[s]            s >= 0        this tile's frames
            state[s + S]    s <  0        the LAST S frames of the PREVIOUS tile
    S    =  KT - 1

    new_state = x[T-S : T]                what the next tile will read

WHERE IT LIVES. This is the convolution in the 3-D VAE of Wan 2.1, HunyuanVideo
and CogVideoX -- `CausalConv3d` in all three trees, and it is the encoder's and
the decoder's only conv. A video VAE compresses time as well as space, so its
convolutions have to reach across frames; making them CAUSAL in time is what lets
the same weights encode a 5-second clip and a 5-minute one, and what makes the
first frame reconstruct as an image rather than as a smear of frames that do not
exist yet.

TWO THINGS HERE HAVE NO PRECEDENT IN THIS SUITE.

  1. ASYMMETRIC PADDING ON ONE AXIS OF THREE. Height and width get the ordinary
     symmetric `same` pad; time gets KT-1 on the LEFT and ZERO on the right, so
     the output has as many frames as the input and no frame ever reads the
     future. causal_conv1d.py has the same one-sided pad but the tensor is
     [C, L] and the causal axis IS the vector axis. Here the causal axis is a
     third axis that is neither on the lanes nor in-lane -- it is the frame
     stride -- so the predicate is uniform across the whole tile and varies only
     from tap to tap. That is a different shape of mask entirely: causal_conv1d
     masks WITHIN a vector, this masks a WHOLE TILE in or out.

  2. STATE THAT OUTLIVES THE LAUNCH. A 5-second 720p clip does not fit, so
     inference runs frame-tiled and every tile must produce bit-identical output
     to the untiled run. The left pad of tile n is therefore not zeros -- it is
     the last KT-1 frames of tile n-1, kept in `conv_cache` between calls. Every
     other kernel in the three suites is a pure function of its inputs; this one
     reads a buffer written by its own previous invocation and writes the buffer
     its next invocation will read. Zeros are correct only for tile 0, and using
     them everywhere is exactly the bug that puts a visible seam at every tile
     boundary in a long generation.

     paged_attention.py and kv_cache_write.py also touch a cache, but that cache
     is an accumulating log indexed by position. This is a RING OF FIXED DEPTH,
     KT-1 frames, and its contents are a copy of a slice of the input -- so the
     kernel both consumes and produces it, and the two must agree on the offset.

HOW THE TWO SOURCES ARE READ. There is no way to address a value that lives in
two different buffers, so each tap issues TWO masked loads at complementary
predicates and adds them -- `mask=(src >= 0)` against x, `mask=(src < 0)` against
the state, `other=0.0` on both. One of the two is always entirely dead. That is
also what `torch.cat([cache, x], dim=2)` costs in the reference implementation,
except the cat materialises a whole extra tensor and this does not.

SHAPES. The input arrives ALREADY PADDED in height and width (HP = H+2, WP = W+2)
the way conv2d.py's does, because the spatial pad is not what this kernel is
about and masking it too would bury the time predicate in noise. One channel:
the real op is [B, C, T, H, W] with a full channel contraction, and the channel
axis would be an outer program dimension; nothing about the time boundary changes
with it. KT = KH = KW = 3 is what all three VAEs ship.

The spatial taps run in RUNTIME loops and the time taps in `tl.static_range`,
which is the split that matters: the time index decides WHICH BUFFER is read, so
it has to be visible per tap, while a spatial offset is just an address.

A third, non-tiled axis carrying a one-sided pad, and the pad's contents coming from a state buffer the kernel also rewrites. `ns` is a pure copy, so a mismatch there is addressing; a mismatch in `o` at t < KT-1 only is the state path, and everywhere is the conv.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T = 4                       # frames in this temporal tile
H, W = 8, 8                 # output height, width
KT, KH, KW = 3, 3, 3        # the kernel every 3-D video VAE ships
S = KT - 1                  # 2 -- the depth of the carried boundary state
HP, WP = H + KH - 1, W + KW - 1     # 10, 10: spatially pre-padded input
PP = 16                     # HP/WP rounded up: arange needs a power of two
FR = HP * WP                # frame stride, in both x and the state


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def causal_conv3d_state_kernel(x_ptr, st_ptr, w_ptr, b_ptr,
                                   o_ptr, ns_ptr,
                                   T: tl.constexpr, H: tl.constexpr,
                                   W: tl.constexpr, KT: tl.constexpr,
                                   KH: tl.constexpr, KW: tl.constexpr,
                                   S: tl.constexpr, HP: tl.constexpr,
                                   WP: tl.constexpr, PP: tl.constexpr,
                                   FR: tl.constexpr):
        rh = tl.arange(0, H)
        rw = tl.arange(0, W)
        bias = tl.load(b_ptr)

        for t in tl.static_range(T):
            acc = tl.zeros((H, W), dtype=tl.float32) + bias

            for kt in tl.static_range(KT):
                # CAUSAL IN TIME: tap kt reads t - (KT-1) + kt, so the earliest
                # taps of the earliest frames fall off the FRONT of the tile.
                src = t + kt - KT + 1
                srcb = src + tl.zeros((H, W), dtype=tl.int32)
                from_x = srcb >= 0          # uniform over the tile, per tap
                from_st = srcb < 0

                for kh in range(KH):
                    for kw in range(KW):
                        off = (rh[:, None] + kh) * WP + (rw[None, :] + kw)
                        # The same element, from whichever buffer holds it. One
                        # of these two loads is always entirely masked off.
                        v = tl.load(x_ptr + srcb * FR + off,
                                    mask=from_x, other=0.0)
                        v += tl.load(st_ptr + (srcb + S) * FR + off,
                                     mask=from_st, other=0.0)
                        acc += tl.load(w_ptr + (kt * KH + kh) * KW + kw) * v

            tl.store(o_ptr + t * (H * W) + rh[:, None] * W + rw[None, :], acc)

        # THE STATE THE NEXT TILE WILL READ: the last S frames of this one,
        # copied out at the offset the load above expects to find them at. The
        # padded frame is 10 x 10, so the copy tile is the next power of two
        # with the tail masked -- the same reason flux_single_block pads its
        # free axis.
        ph = tl.arange(0, PP)
        pw = tl.arange(0, PP)
        pm = (ph[:, None] < HP) & (pw[None, :] < WP)
        poff = ph[:, None] * WP + pw[None, :]
        for s in tl.static_range(S):
            tl.store(ns_ptr + s * FR + poff,
                     tl.load(x_ptr + (T - S + s) * FR + poff, mask=pm,
                             other=0.0),
                     mask=pm)

    return causal_conv3d_state_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * FR, generator=g, dtype=torch.float32),
            "st": torch.randn(S * FR, generator=g, dtype=torch.float32),
            "w": torch.randn(KT * KH * KW, generator=g, dtype=torch.float32),
            "b": torch.randn(1, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, HP, WP)
    st = inputs["st"].reshape(S, HP, WP)
    w = inputs["w"].reshape(1, 1, KT, KH, KW)

    # The reference CONCATENATES -- state then input, the `torch.cat` the real
    # implementation does -- and then runs a plain valid conv3d over it. That
    # the kernel gets the same answer without building the joined tensor is the
    # point of the two masked loads.
    joined = torch.cat([st, x], dim=0)[None, None]          # [1,1,T+S,HP,WP]
    o = torch.nn.functional.conv3d(joined, w, bias=inputs["b"])
    return {"o": o.reshape(-1).contiguous(),
            "ns": x[T - S:T].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="causal_conv3d_state_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "st_ptr": "*fp32", "w_ptr": "*fp32",
               "b_ptr": "*fp32", "o_ptr": "*fp32", "ns_ptr": "*fp32",
               "T": "constexpr", "H": "constexpr", "W": "constexpr",
               "KT": "constexpr", "KH": "constexpr", "KW": "constexpr",
               "S": "constexpr", "HP": "constexpr", "WP": "constexpr",
               "PP": "constexpr", "FR": "constexpr"},
    constexprs={"T": T, "H": H, "W": W, "KT": KT, "KH": KH, "KW": KW,
                "S": S, "HP": HP, "WP": WP, "PP": PP, "FR": FR},
    args=[Arg("x", IN, "float32", (T * FR,)),
          Arg("st", IN, "float32", (S * FR,)),
          Arg("w", IN, "float32", (KT * KH * KW,)),
          Arg("b", IN, "float32", (1,)),
          Arg("o", OUT, "float32", (T * H * W,)),
          Arg("ns", OUT, "float32", (S * FR,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
