"""ConvNeXt V2's Global Response Normalization -- two folds, two different axes.

    Gx = ||x||_2 over the SPATIAL axes, per channel        -> [C]
    Nx = Gx / (mean of Gx over the CHANNEL axis + eps)     -> [C]
    y  = gamma * (x * Nx) + beta + x

torch equivalent: the GRN module in facebookresearch/ConvNeXt-V2, verbatim.

WHY IT EXISTS. V2 trains ConvNeXt with a masked autoencoder and the blocks
collapse: most channels of most feature maps go dead, because nothing in the
architecture makes channels compete. GRN is the competition. A channel's own
spatial energy is divided by the AVERAGE energy of all channels, so a channel
that is quiet relative to its neighbours gets scaled down further and a loud one
gets amplified -- and V2 deletes LayerScale, which was the thing V1 used to keep
deep stacks trainable, because GRN does that job too.

THE SHAPE IS THE POINT: TWO REDUCTIONS OVER DIFFERENT AXES, IN SEQUENCE, WITH
THE SECOND EATING THE FIRST'S OUTPUT.

  layernorm.py, rmsnorm.py, groupnorm_film.py all fold ONE axis and broadcast
  back along it. Whichever axis that is, put it in-lane and both halves are
  free. convnext_block.py says so explicitly: "the fold and the broadcast are
  both along the channel axis, and both stay inside one lane".

  Here the two folds want OPPOSITE layouts. The L2 norm folds HW, so HW must be
  the inner axis and the tile is [C, HW] -- which is not how a channels-last
  ConvNeXt stores it, hence the strided load that transposes on the way in, the
  same trick attention.py uses for K. Then the mean folds C, which in that tile
  is the LANE axis, and there is no lane-crossing primitive.

  The resolution is that the first fold has already collapsed HW away: `Gx` is
  rank 1, length C, so it is re-presented as a [1, C] tile and the second fold
  is again an inner-axis reduction, now over a single row. That is the whole
  trick, and it is the only thing in the suite that needs it -- a rank-1
  intermediate whose lane axis has to be reinterpreted between two reductions.

  It also means the second reduction is one lane deep and 32 wide, the shape
  pooling.py calls "the case to avoid". It is unavoidable here: there is
  genuinely one number for the whole layer.

`torch.norm(p=2)` AND NOT A MEAN OF SQUARES. There is no 1/HW anywhere -- the
scale of Gx cancels in the ratio, which is why the authors could leave it out,
and why a backend that "helpfully" normalises the reduction still gets the right
answer. The eps is on the DENOMINATOR outside the sqrt, not inside it the way
RMSNorm puts it, so it does not guard the sqrt at all; it guards an all-zero
feature map.

The residual `+ x` is outside gamma and outside beta. gamma and beta both init
to zero, so a fresh GRN is exactly the identity.

FAILS TODAY, AND NO LONGER FOR THE REASON IT USED TO. It used to die at stage 6
with `llc spilled a vector register GROUP` (1 x vs8r.v), so it never ran at all.
With legalize sizing the loop correctly it runs and comes back NaN in
1339 of 2048 elements.

WHAT IS WRONG IS THE PER-ROW BROADCAST, dumped straight out of the last loop by
storing the intermediate instead of the result:

    %9  = vector.broadcast %8 : vector<1xf32> to vector<32xf32>
          rows 0,1,2 -> [nan, nan, 0.0, 0.0]
    %12                          -> [-0.592, 0.746, 0.192, -0.178]  (all rows)

Constant DOWN the lane axis and varying ALONG the vector, which is the transpose
of what a per-row broadcast means. The loads themselves are exact (storing %7
reproduces the input bit for bit), so the tile is staged correctly and it is the
scalar-per-row read that lands in the wrong place.

NOT A GENERAL BROADCAST BUG: `o = x * s[:, None]` on the same 32x64 tile with s
coming from a DMA verifies exactly, and ops_reduce_broadcast and rmsnorm pass.
What is different here is that the per-row scalar is a REDUCTION RESULT read back
out of its own accumulator buffer. Not chased further.

The only kernel here that folds two DIFFERENT axes in sequence: an L2 over space per channel, then a mean over channels of that. The second fold's axis is the first tile's lane axis, so the rank-1 intermediate has to be re-laid-out between them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

HW, C = 64, 32                  # spatial positions, channels (channels-last)
EPS = 1e-6


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def convnextv2_grn_kernel(x_ptr, gam_ptr, beta_ptr, o_ptr,
                              HW: tl.constexpr, C: tl.constexpr,
                              EPS: tl.constexpr):
        c = tl.arange(0, C)
        s = tl.arange(0, HW)

        # x is stored [HW, C] (channels-last, as ConvNeXt keeps it) and read as
        # [C, HW], so the SPATIAL axis lands in-lane and the L2 fold is an
        # inner-axis reduction. The transpose is in the addressing, not a
        # tl.trans of a loaded tile.
        xt = tl.load(x_ptr + s[None, :] * C + c[:, None])

        # FOLD ONE: over the spatial axis, one norm per channel.
        gx = tl.sqrt(tl.sum(xt * xt, axis=1))                 # [C]

        # FOLD TWO: over the channel axis -- which is the lane axis of the tile
        # above. It only works because gx is rank 1 now, so presenting it as a
        # [1, C] row makes C the inner axis again.
        gm = tl.sum(gx[None, :], axis=1) / C                   # [1]
        nx = gx / (gm + EPS)                                   # [C]

        # gamma * (x * Nx) + beta + x, written back through the same transpose.
        y = (tl.load(gam_ptr + c)[:, None] * (xt * nx[:, None])
             + tl.load(beta_ptr + c)[:, None] + xt)
        tl.store(o_ptr + s[None, :] * C + c[:, None], y)

    return convnextv2_grn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Per-channel gains spread over two orders of magnitude, so Gx/mean(Gx) is
    # actually doing something rather than sitting at 1.
    scale = torch.rand(C, generator=g, dtype=torch.float32) * 2.0 + 0.05
    x = torch.randn(HW, C, generator=g, dtype=torch.float32) * scale
    return {"x": x.reshape(-1).contiguous(),
            "gam": torch.randn(C, generator=g, dtype=torch.float32),
            "beta": torch.randn(C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(HW, C)
    gx = torch.linalg.vector_norm(x, ord=2, dim=0)             # over spatial
    nx = gx / (gx.mean(dim=-1, keepdim=True) + EPS)            # over channels
    y = inputs["gam"] * (x * nx) + inputs["beta"] + x
    return {"o": y.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="convnextv2_grn_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "gam_ptr": "*fp32", "beta_ptr": "*fp32",
               "o_ptr": "*fp32", "HW": "constexpr", "C": "constexpr",
               "EPS": "constexpr"},
    constexprs={"HW": HW, "C": C, "EPS": EPS},
    args=[Arg("x", IN, "float32", (HW * C,)),
          Arg("gam", IN, "float32", (C,)),
          Arg("beta", IN, "float32", (C,)),
          Arg("o", OUT, "float32", (HW * C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
