"""SAM2 / Hiera `MultiScaleAttention` -- the query is POOLED, K and V are not.

    q'   = maxpool2x2(q over the HxW token grid)      NQ = N / 4
    out  = softmax(q' @ k^T / sqrt(D)) @ v            + q'   (pooled shortcut)

torch equivalent: hieradet.MultiScaleAttention.forward with `q_pool` set --
`do_pool(q, nn.MaxPool2d(2, 2))`, then scaled_dot_product_attention, then the
MultiScaleBlock's `shortcut + attn(x)` where the shortcut went through the same
pool.

DOWNSAMPLING INSIDE THE ATTENTION, NOT BETWEEN BLOCKS. A hierarchical vision
backbone has to reduce resolution somewhere. A ViT does it once, in the patch
embed; a Swin does it in a separate `PatchMerging` layer between stages. Hiera
does it in the attention op itself: the stage-transition block pools ONLY the
query, so the output has a quarter of the tokens while the keys and values still
see the full-resolution map. The coarse tokens are therefore built by attending
over fine ones -- a learned, content-dependent downsample instead of a strided
concat -- and it is what makes Hiera a plain ViT with no window shifting, no
relative position bias and no cross-scale plumbing. SAM2 runs this as its image
encoder on every frame.

THE THING NOTHING ELSE IN THE SUITE DOES: THE TWO SEQUENCE LENGTHS INSIDE ONE
ATTENTION DIFFER. Every other attention here -- attention.py, sparse_attention,
swin_window, swinv2_cosine_attn, gqa_repeat_kv -- has square score tiles or
gets its rectangle from BLOCKING a square one, so `BM` and `BN` name pieces of
the same axis. Here (NQ, N) is rectangular because the OPERATOR is, 16 by 64 and
not by tiling. Then `p @ v` contracts the LONG axis back down, so the two
matmuls of one attention have their large dimension in different places: the
first produces it, the second consumes it. A layout choice that suits `q @ k^T`
is not automatically the one `p @ v` wants, and with a square score tile that
question never comes up.

THE POOL IS A GATHER WITH A DIVIDED INDEX. The tokens are a flat sequence that
is secretly an 8x8 grid, so pooled query `i` reads rows `(2*(i//4) + ky) * 8 +
2*(i%4) + kx`. `i // 4` is upsample.py's divided index run backwards -- there it
made four outputs share an input, here it makes four inputs share an output --
and it is the same place a structured-pointer analysis stops being affine. Four
shifted (NQ, D) tiles folded by `tl.maximum`, which is pooling.py's shape with
the window walking a SEQUENCE axis rather than a plane, so each tap is a row
gather of D contiguous channels rather than a strided plane read.

THE SHORTCUT IS POOLED TOO, and by the same pool -- the residual cannot be the
unpooled input because it no longer has the right length. So `q'` is live from
before the first matmul until after the second, across both dots and the
softmax, which is the register-pressure shape fused_add_rmsnorm.py isolates,
here stretched over the whole kernel.

One head: the head axis is a batch and gqa_repeat_kv covers what it does to
addressing. `dim_out != dim` (Hiera widens the channels at a stage transition)
is a projection, which conv1x1.py already is.

Attention whose Q and K have different sequence lengths: a 2x2 max pool on the query only, so the score tile is genuinely rectangular and the two matmuls disagree about where the long axis lives.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W = 8, 8                     # the token grid hiding inside the sequence
N = H * W                       # 64 keys / values
OH, OW = H // 2, W // 2
NQ = OH * OW                    # 16 queries after the 2x2 pool
D = 16                          # head dim
SCALE = 1.0 / (D ** 0.5)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sam2_qpool_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                               W: tl.constexpr, N: tl.constexpr,
                               NQ: tl.constexpr, OW: tl.constexpr,
                               D: tl.constexpr, SCALE: tl.constexpr):
        i = tl.arange(0, NQ)
        d = tl.arange(0, D)

        # 2x2 max pool over the grid the sequence really is. The divided index
        # is what takes this off the affine path.
        base = (i // OW) * (2 * W) + (i % OW) * 2
        qp = tl.load(q_ptr + base[:, None] * D + d[None, :])
        for ky in tl.static_range(2):
            for kx in tl.static_range(2):
                r = base + ky * W + kx
                qp = tl.maximum(qp, tl.load(q_ptr + r[:, None] * D
                                            + d[None, :]))

        # (NQ, N) -- rectangular because the OPERATOR is, not because of tiling.
        n = tl.arange(0, N)
        kt = tl.load(k_ptr + n[None, :] * D + d[:, None])
        s = tl.dot(qp, kt) * SCALE
        m = tl.max(s, axis=1)
        e = tl.exp(s - m[:, None])
        p = e / tl.sum(e, axis=1)[:, None]

        # The second dot contracts the long axis the first one produced.
        v = tl.load(v_ptr + n[:, None] * D + d[None, :])
        acc = tl.dot(p, v)

        # The residual is the POOLED query -- the unpooled input is the wrong
        # length -- so qp stays live across both matmuls and the softmax.
        tl.store(o_ptr + i[:, None] * D + d[None, :], acc + qp)

    return sam2_qpool_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(N * D, generator=g, dtype=torch.float32),
            "k": torch.randn(N * D, generator=g, dtype=torch.float32) * 0.5,
            "v": torch.randn(N * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    F = torch.nn.functional
    q = inputs["q"].reshape(1, H, W, D).permute(0, 3, 1, 2)
    qp = F.max_pool2d(q, 2, 2).permute(0, 2, 3, 1).reshape(NQ, D)
    k = inputs["k"].reshape(N, D)
    v = inputs["v"].reshape(N, D)
    p = torch.softmax(qp @ k.t() * SCALE, dim=-1)
    return {"o": (p @ v + qp).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="sam2_qpool_attn_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "o_ptr": "*fp32", "W": "constexpr", "N": "constexpr",
               "NQ": "constexpr", "OW": "constexpr", "D": "constexpr",
               "SCALE": "constexpr"},
    constexprs={"W": W, "N": N, "NQ": NQ, "OW": OW, "D": D, "SCALE": SCALE},
    args=[Arg("q", IN, "float32", (N * D,)),
          Arg("k", IN, "float32", (N * D,)),
          Arg("v", IN, "float32", (N * D,)),
          Arg("o", OUT, "float32", (NQ * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
