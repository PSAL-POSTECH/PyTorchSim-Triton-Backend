"""Swin's shifted window -- a cyclic roll, then a partition into blocks.

    roll      (y, x) -> ((y + S) mod H, (x + S) mod W)
    partition the rolled plane into MxM windows
    attend    inside each window only
    unroll    write back through the same map

ATTENTION THAT IS LINEAR IN THE IMAGE. A ViT attends every patch to every patch,
which is quadratic in the number of patches and hopeless at high resolution.
Swin attends only within an MxM window, which is linear -- and then has the
problem every windowed scheme has: the window boundaries never move, so
information cannot cross them.

**THE SHIFT IS THE FIX, AND IT IS A ROLL RATHER THAN A DIFFERENT PARTITION.**
Alternate layers roll the whole plane by half a window before partitioning, so a
boundary in one layer is the middle of a window in the next. Two layers together
therefore mix everything, and neither one pays for global attention. Doing it by
rolling rather than by re-partitioning keeps every window exactly MxM -- the
patches that wrap around the edge form their own window, and Swin masks them
apart so a top-row patch does not attend to a bottom-row one.

  It is the same trick as `channel_shuffle`, one axis over. There the factored
  transpose mixes channel groups; here the roll mixes spatial windows. Both
  exist because a cheap operation partitioned something and the partition then
  had to be broken.

WHAT IT ASKS OF THIS MACHINE. The window partition is index arithmetic --
`(y + S) mod H` with H a power of two is a mask, so every lane computes its own
address and nothing is looked up. The windows are a static loop for the usual
reason: a window axis in the tile would make it rank 3.

  The fold inside a window is per ROW here, not over the whole MxM block.
  Folding both axes of one tile forbids both as the lane axis (see
  `groupnorm_film`), and a window is not contiguous in memory so the flat read
  that kernel uses is not available. What is being tested is the partition, not
  the softmax.

torch equivalent: torch.roll(x, (-S, -S), (0, 1)), then view/permute into
windows, attend, and roll back.

Swin: a cyclic roll folded into the window addresses, which is channel_shuffle's trick one axis over -- a cheap partition, then the partition broken. It was red on `operand [1, 8] broadcasts along dim 0 into a tile [8, 8], and it was staged one element per lane` -- a refusal that names a lane axis the tile does not have, since its component resolves to ONE_LANE. Nothing is banked there, so the row is read at its own index rather than replicated.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W = 16, 16
M = 8                  # window edge
SH = M // 2            # the shift, half a window
NW = H // M            # windows per axis


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def swin_window_kernel(x_ptr, o_ptr, H: tl.constexpr, W: tl.constexpr,
                           M: tl.constexpr, SH: tl.constexpr,
                           NW: tl.constexpr):
        m = tl.arange(0, M)

        for wy in tl.static_range(NW):
            for wx in tl.static_range(NW):
                # THE ROLL, folded into the window's addresses. `mod H` is a
                # mask because H is a power of two, so this is arithmetic each
                # lane does for itself rather than a table.
                ys = (wy * M + m + SH) & (H - 1)
                xs = (wx * M + m + SH) & (W - 1)
                idx = ys[:, None] * W + xs[None, :]

                v = tl.load(x_ptr + idx)
                # Attention stands in as a per-row softmax over the window.
                mx = tl.max(v, axis=1)
                ex = tl.exp(v - mx[:, None])
                p = ex / tl.sum(ex, axis=1)[:, None]
                # UNROLL by writing back through the same map.
                tl.store(o_ptr + idx, p)

    return swin_window_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(H, W)
    out = torch.empty_like(x)
    m = torch.arange(M)
    for wy in range(NW):
        for wx in range(NW):
            ys = (wy * M + m + SH) % H
            xs = (wx * M + m + SH) % W
            v = x[ys[:, None], xs[None, :]]
            out[ys[:, None], xs[None, :]] = torch.softmax(v, dim=-1)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="swin_window_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "H": "constexpr",
               "W": "constexpr", "M": "constexpr", "SH": "constexpr",
               "NW": "constexpr"},
    constexprs={"H": H, "W": W, "M": M, "SH": SH, "NW": NW},
    args=[Arg("x", IN, "float32", (H * W,)),
          Arg("o", OUT, "float32", (H * W,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
