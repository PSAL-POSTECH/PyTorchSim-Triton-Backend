"""InternVL3 / Llama4 `pixel_shuffle` -- the vision-to-LLM connector, and it is
four reshapes with no arithmetic in it at all.

    [H, W, C]  view   -> [H,    W/2, 2C]
               permute-> [W/2,  H,   2C]
               view   -> [H/2,  W/2, 4C]
               permute-> [W/2,  H/2, 4C]

    scale_factor = 1/2:  a quarter of the patches, four times the channels.

WHY A VLM THROWS AWAY SPATIAL RESOLUTION ON PURPOSE. A 448x448 image at patch 14
is 1024 vision tokens, and those tokens go into the LLM's context, where
attention is quadratic and every one of them competes with the user's actual
text. InternVL's answer is that a 2x2 neighbourhood of patches carries less than
four patches' worth of independent information, so STACK the four along the
channel axis instead of keeping them as four positions: 1024 tokens become 256,
the MLP that follows absorbs the 4x-wide channel vector down to the LLM's hidden
size, and nothing is discarded -- only re-indexed. Llama4 ships the identical
function under the same name with `shuffle_ratio`, and both feed it straight
into a two-layer MLP projector.

  It is `nn.PixelShuffle` run backwards. The super-resolution operator moves
  channels INTO space; this moves space into channels, and that is why the
  permutes look inverted next to the torch op.

ZERO FLOPS, ZERO WEIGHTS, AND THAT IS THE POINT OF THE KERNEL. There is no
multiply-add anywhere in this file. Its entire cost is address arithmetic and
DRAM, so it is the purest bandwidth-bound layout transform in the suite and the
only thing a backend can get wrong is where the bytes land. The MLP after it is
an ordinary matmul and is deliberately not here -- `conv1x1.py` already is that,
and folding it in would hide the part with no arithmetic to check against.

  `channel_shuffle.py` is the nearest kernel and it is a strictly smaller
  problem: one factored transpose, INSIDE the channel axis, with the spatial
  axis carried along untouched. Here the two permutes each move a SPATIAL
  extent into the channel extent, so the axis that is contiguous at the end was
  strided at the start and the map is not a permutation of channels at all --
  `out[w, h, c]` reads a source element whose H, W and C coordinates are all
  functions of all three. Two reshapes and two transposes chained, with a
  reshape BETWEEN the transposes, which is what stops the pair collapsing into
  one permutation the way `reshape/transpose/reshape` does.

WHAT IT ASKS OF THIS MACHINE. Rank-3 tiles and `tl.trans(x, 1, 0, 2)` twice, the
same op `channel_shuffle` uses, but the second one operates on a tile whose
current shape came out of a reshape of a transpose -- so the backend cannot keep
a single affine view from load to store and has to decide where to materialise.
With the channels in-lane the whole thing is bank-local; with the patches in-lane
the first permute is lane-to-lane and has to become a DRAM round trip.

WHAT IS PINNED. The real thing derives `patch_size` from `sqrt(num_patches)` at
runtime, per image. Here the grid is a constant, because a data-dependent tile
shape is not expressible; the layout map is what is being tested and it is
unchanged.

torch equivalent:
    x = x.reshape(H, W // 2, C * 2).permute(1, 0, 2).contiguous()
    x = x.reshape(H // 2, W // 2, C * 4).permute(1, 0, 2).contiguous()

InternVL3/Llama4 pixel_shuffle: view/permute/view/permute trading 4x spatial for 4x channels. No arithmetic at all, so any mismatch is a layout fault and the tolerance is exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W, C = 8, 8, 8          # patch grid and channels, scale_factor = 1/2
WC = W * C                 # the flat in-lane extent of the loaded tile
W2, H2 = W // 2, H // 2    # every derived extent is its own constexpr: a shape
C2, C4 = C * 2, C * 4      # element must be a constexpr USED DIRECTLY
OC = H2 * C4               # ...and the flat in-lane extent of the stored tile


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def pixel_shuffle_merge_kernel(x_ptr, o_ptr, H: tl.constexpr,
                                   WC: tl.constexpr, W2: tl.constexpr,
                                   H2: tl.constexpr, C2: tl.constexpr,
                                   C4: tl.constexpr, OC: tl.constexpr):
        r = tl.arange(0, H)
        c = tl.arange(0, WC)
        x = tl.load(x_ptr + r[:, None] * WC + c[None, :])       # [H, W*C]

        # HALF OF W INTO THE CHANNELS, then the transpose that makes it stick.
        a = tl.reshape(x, (H, W2, C2))
        b = tl.trans(a, 1, 0, 2)                                # [W/2, H, 2C]

        # HALF OF H INTO THE CHANNELS. The reshape between the two transposes
        # is what stops them folding into one permutation.
        d = tl.reshape(b, (H2, W2, C4))
        e = tl.trans(d, 1, 0, 2)                                # [W/2, H/2, 4C]

        ro = tl.arange(0, W2)
        co = tl.arange(0, OC)
        tl.store(o_ptr + ro[:, None] * OC + co[None, :],
                 tl.reshape(e, (W2, OC)))

    return pixel_shuffle_merge_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(H * W * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(H, W, C)
    x = x.reshape(H, W // 2, C * 2).permute(1, 0, 2).contiguous()
    x = x.reshape(H // 2, W // 2, C * 4).permute(1, 0, 2).contiguous()
    return {"o": x.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="pixel_shuffle_merge_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "H": "constexpr",
               "WC": "constexpr", "W2": "constexpr", "H2": "constexpr",
               "C2": "constexpr", "C4": "constexpr", "OC": "constexpr"},
    constexprs={"H": H, "WC": WC, "W2": W2, "H2": H2, "C2": C2, "C4": C4,
                "OC": OC},
    args=[Arg("x", IN, "float32", (H * W * C,)),
          Arg("o", OUT, "float32", (H * W * C,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
