"""A diffusion U-Net ResBlock's conditioning path -- GroupNorm, then FiLM.

    emb    = [sin(t*w), cos(t*w)]          sinusoidal, built from the STEP
    s, b   = Ws @ emb,  Wb @ emb           one scale and one shift per channel
    h      = GroupNorm(x, G)
    out    = h * (1 + s[:, None]) + b[:, None]

THE STEP IS AN INPUT TO THE NETWORK, and that is what makes a diffusion U-Net
different from a classifier. The same weights run at every noise level, so the
network has to be told which one it is at -- and it is told the way a
Transformer is told about position, with a sinusoidal code. Then a small MLP
turns that into a per-channel scale and shift which every ResBlock applies to
its normalised activations. That is FiLM: the condition does not enter as
another input tensor, it MODULATES the ones already there.

GROUPNORM IS THE OPERATION THIS SUITE DID NOT HAVE, and the reason is the shape
rather than the arithmetic. Every reduction here folds a WHOLE axis:

    BatchNorm    over the batch, which is not in the tile at all
    LayerNorm    over the features
    softmax      over a row
    GroupNorm    over a SLICE of the channel axis, together with the spatial
                 axes -- a PARTIAL-axis fold

  Why diffusion uses it: batch statistics are useless here. Training samples
  differ in noise level, so a batch mixes timesteps and its mean means nothing;
  and at sampling time the batch is 1 or 2. LayerNorm over all channels is too
  coarse for convolutional features. Groups of 32 channels is what the GroupNorm
  paper settled on and what every diffusion U-Net copies.

WRITTEN AS A LOOP OVER GROUPS, and that is not a stylistic choice. `[C, HW]`
viewed as `[G, C/G, HW]` and folded on axes 1 and 2 is a RANK-3 REDUCTION with a
broadcast back along the group axis, which this backend refuses (see the
rulebook's vectorize section). G is static, so the loop keeps every tile
rank 2 and costs nothing -- the same move `alibi` and `moe_router` make.

AND THE STATISTICS ARE TAKEN FLAT. Folding the group's [CG, HW] tile on both
axes forbids both of them as the lane axis and the forest has nothing left to
pick, so the block is read twice: once as a rank-1 vector for the mean and
variance, once shaped for the data. See the comment at the loop.

THE TIMESTEP IS A RUNTIME SCALAR, not a constexpr. It changes every sampler
step, and baking it in would mean recompiling the network once per step.

torch equivalent:
    emb = cat([sin(t * freqs), cos(t * freqs)])
    s, b = w_s @ emb, w_b @ emb
    h = group_norm(x, G)
    out = h * (1 + s[:, None]) + b[:, None]

Diffusion ResBlock conditioning: GroupNorm (a PARTIAL-axis fold), a sinusoidal step embedding built from a runtime scalar, and FiLM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, HW, G = 32, 64, 4
E = 16                      # embedding width; E/2 frequencies
CG = C // G
EPS = 1e-5
TSTEP = 137                 # the runtime timestep the test uses


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def groupnorm_film_kernel(x_ptr, ws_ptr, wb_ptr, o_ptr, t,
                              C: tl.constexpr, HW: tl.constexpr,
                              G: tl.constexpr, E: tl.constexpr,
                              CG: tl.constexpr, EPS: tl.constexpr):
        p = tl.arange(0, HW)
        e = tl.arange(0, E)

        # SINUSOIDAL EMBEDDING of the step. half the entries are sin, half cos,
        # over frequencies spaced geometrically -- the same code a Transformer
        # uses for position, applied to "how noisy is this".
        half: tl.constexpr = E // 2
        n: tl.constexpr = CG * HW
        CGHW: tl.constexpr = CG * HW
        k = e % half
        freq = tl.exp(-9.210340371976184 * k.to(tl.float32) / half)  # 10000^-k/half
        ang = t.to(tl.float32) * freq
        emb = tl.where(e < half, tl.sin(ang), tl.cos(ang))

        # ONE GROUP AT A TIME, so every tile stays rank 2 -- and the MLP goes
        # inside the loop rather than outside it, because a [C] vector cannot be
        # sliced to a group's channels once it is a value. Reading the group's
        # rows of the weight is the same work in a different order.
        for g in tl.static_range(G):
            cg = g * CG + tl.arange(0, CG)
            sg = tl.sum(tl.load(ws_ptr + cg[:, None] * E + e[None, :])
                        * emb[None, :], axis=1)
            bg = tl.sum(tl.load(wb_ptr + cg[:, None] * E + e[None, :])
                        * emb[None, :], axis=1)

            # GROUPNORM. The statistics are taken over a FLAT read of the
            # group's block and the data over a shaped one, which is two reads
            # of the same bytes and deliberately so:
            #
            #   folding a [CG, HW] tile on BOTH axes forbids both as the lane
            #   axis -- the second reduce's "not this axis" travels back through
            #   the rank-1 intermediate onto the tile -- and the forest then has
            #   nothing left to choose. Measured: "every axis is forbidden as
            #   the lane axis". A rank-1 read folds ONE axis of a buffer that
            #   has one, which is the ordinary reduce-to-a-scalar this backend
            #   already stages (_replicate_captured_scalars).
            #
            # It is also what a real GroupNorm does: two passes, or Welford.
            q = tl.arange(0, CGHW)
            flat = tl.load(x_ptr + g * CGHW + q)
            mu = tl.sum(flat, axis=0) / n
            fd = flat - mu
            var = tl.sum(fd * fd, axis=0) / n

            xt = tl.load(x_ptr + cg[:, None] * HW + p[None, :])
            # THE RSQRT STAYS IN THE VECTOR DOMAIN. `var` is rank 0, and a
            # transcendental on a scalar has nowhere to run here -- VCIX takes a
            # vector register and the LLVM tail would emit a libm call that does
            # not link. Broadcasting first computes the same number in every
            # lane, which is what a splat does anyway.
            rstd = tl.exp(-0.5 * tl.log(var + EPS + 0.0 * xt))
            h = (xt - mu) * rstd

            # FiLM: the condition MODULATES rather than adds. `1 + s` so a zero
            # scale is the identity, which is how these are initialised.
            out = h * (1.0 + sg[:, None]) + bg[:, None]
            tl.store(o_ptr + cg[:, None] * HW + p[None, :], out)

    return groupnorm_film_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * HW, generator=g, dtype=torch.float32),
            "ws": torch.randn(C * E, generator=g, dtype=torch.float32) * 0.3,
            "wb": torch.randn(C * E, generator=g, dtype=torch.float32) * 0.3}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(C, HW)
    half = E // 2
    k = torch.arange(E) % half
    freq = torch.exp(-9.210340371976184 * k.float() / half)
    ang = float(TSTEP) * freq
    emb = torch.where(torch.arange(E) < half, ang.sin(), ang.cos())
    s = inputs["ws"].reshape(C, E) @ emb
    b = inputs["wb"].reshape(C, E) @ emb
    h = torch.empty_like(x)
    for g in range(G):
        sl = x[g * CG:(g + 1) * CG]
        mu = sl.mean()
        d = sl - mu
        h[g * CG:(g + 1) * CG] = d * torch.rsqrt((d * d).mean() + EPS)
    out = h * (1.0 + s[:, None]) + b[:, None]
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="groupnorm_film_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "ws_ptr": "*fp32", "wb_ptr": "*fp32",
               "o_ptr": "*fp32", "t": "i32",
               "C": "constexpr", "HW": "constexpr", "G": "constexpr",
               "E": "constexpr", "CG": "constexpr", "EPS": "constexpr"},
    constexprs={"C": C, "HW": HW, "G": G, "E": E, "CG": CG, "EPS": EPS},
    args=[Arg("x", IN, "float32", (C * HW,)),
          Arg("ws", IN, "float32", (C * E,)),
          Arg("wb", IN, "float32", (C * E,)),
          Arg("o", OUT, "float32", (C * HW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("t", "int32_t")], "scalar_values": {"t": TSTEP}},
    rtol=1e-4, atol=1e-4,
)
