"""2-D axial RoPE -- Flux and SD3's position code for a grid of patches.

    channels 0 .. D/2      rotated by the ROW index
    channels D/2 .. D      rotated by the COLUMN index

    token n at grid (r, c) = (n // W, n % W)

AN IMAGE HAS TWO POSITIONS AND RoPE ENCODES ONE. `rope.py` rotates by a single
index, which is what a sentence has. A patch grid needs both coordinates, and
the axial answer is to SPLIT THE CHANNELS: half the head dimension carries the
row, half carries the column. Attention between two patches then sees
`R(dr)` on one half and `R(dc)` on the other, so relative position in both
directions falls out of the same dot product.

  Why not add the two rotations. Composing them on the same channels would give
  `R(dr + dc)`, which cannot tell (1, 0) from (0, 1) -- the diagonal collapses.
  Splitting keeps them independent, at the cost of halving the frequency
  resolution each gets.

  Flux and SD3 both do this, with the grid coordinates supplied per token rather
  than derived, because their sequences interleave image patches with text
  tokens and the text ones get position (0, 0).

WHAT IT ASKS OF THIS MACHINE, and why it is cheap. `rope.py` already established
the shape: the two halves a rotation mixes are kept as SEPARATE TILES from the
start, because `rotate_half` inside one tile is a lane-crossing move. Here that
splitting happens twice -- four quarter-width tiles, two rotations -- and the
grid coordinates are `linalg.index` arithmetic (`n // W`, `n % W`), which every
lane does for itself.

  The `//` and `%` are what make the index non-affine, the same thing
  `upsample`'s nearest path hits. Here they only feed the ANGLE, not an address,
  so nothing becomes a gather.

torch equivalent: rotate the first D/2 channels by row and the rest by column,
each in its own rotate_half pairing.

Flux/SD3: axial RoPE -- the channels split so half carry the row and half the column, because composing the two rotations collapses the diagonal.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

GH, GW = 8, 8              # the patch grid
T = GH * GW                # tokens
D = 32                     # head dim; D/2 for rows, D/2 for columns
Q = D // 4                 # quarter: each axis's rotation pairs Q with Q


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rope_2d_kernel(x_ptr, o_ptr, T: tl.constexpr, GW: tl.constexpr,
                       D: tl.constexpr, Q: tl.constexpr):
        n = tl.arange(0, T)
        q = tl.arange(0, Q)

        # THE TWO COORDINATES, from the token index. Non-affine, and it does not
        # matter: they feed the angle rather than an address.
        row = (n // GW).to(tl.float32)
        col = (n % GW).to(tl.float32)

        # Frequencies, shared by both axes -- each gets Q of them.
        inv = tl.exp(-q.to(tl.float32) * (2.0 / Q) * 2.302585092994046)

        ar = row[:, None] * inv[None, :]
        ac = col[:, None] * inv[None, :]

        # FOUR QUARTER TILES. The pairing a rotation mixes is (k, k+Q) within
        # each half, so the halves are read as separate tiles the way rope.py
        # does -- rotate_half inside one tile would cross lanes.
        base = n[:, None] * D
        r1 = tl.load(x_ptr + base + q[None, :])
        r2 = tl.load(x_ptr + base + (Q + q)[None, :])
        c1 = tl.load(x_ptr + base + (2 * Q + q)[None, :])
        c2 = tl.load(x_ptr + base + (3 * Q + q)[None, :])

        cr, sr = tl.cos(ar), tl.sin(ar)
        cc, sc = tl.cos(ac), tl.sin(ac)

        tl.store(o_ptr + base + q[None, :], r1 * cr - r2 * sr)
        tl.store(o_ptr + base + (Q + q)[None, :], r2 * cr + r1 * sr)
        tl.store(o_ptr + base + (2 * Q + q)[None, :], c1 * cc - c2 * sc)
        tl.store(o_ptr + base + (3 * Q + q)[None, :], c2 * cc + c1 * sc)

    return rope_2d_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    n = torch.arange(T)
    row = (n // GW).float()[:, None]
    col = (n % GW).float()[:, None]
    inv = torch.exp(-torch.arange(Q).float() * (2.0 / Q)
                    * 2.302585092994046)[None, :]
    ar, ac = row * inv, col * inv
    r1, r2 = x[:, :Q], x[:, Q:2 * Q]
    c1, c2 = x[:, 2 * Q:3 * Q], x[:, 3 * Q:]
    out = torch.cat([r1 * ar.cos() - r2 * ar.sin(),
                     r2 * ar.cos() + r1 * ar.sin(),
                     c1 * ac.cos() - c2 * ac.sin(),
                     c2 * ac.cos() + c1 * ac.sin()], dim=1)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rope_2d_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "T": "constexpr",
               "GW": "constexpr", "D": "constexpr", "Q": "constexpr"},
    constexprs={"T": T, "GW": GW, "D": D, "Q": Q},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
