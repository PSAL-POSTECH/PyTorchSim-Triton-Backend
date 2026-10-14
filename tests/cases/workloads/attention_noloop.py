"""Attention with NO KV LOOP -- the score matrix fits, so nothing is tiled.

WHAT MAKES THIS DIFFERENT FROM dl/attention. That kernel walks K and V in an
`scf.for`, because its sequence does not fit in the scratchpad. Here it does:
64 queries against 64 keys is a 64x64 score matrix, and the whole of softmax
happens in registers with no loop anywhere.

That is not a smaller version of the same kernel, it is a different shape for
this backend. `_widen_unit_expand` -- the rewrite that turns
`expand_shape [N] -> [1, N]` into a broadcast, so a softmax row stops being
paired with the tile's UNIT axis -- is gated on `_rides_a_loop`, and a row that
rides no loop never reaches it. So the row was banked on axis 0 and the score
matrix on axis 1, and vectorize refused:

    operands of one elementwise op are banked on different axes [0, 1]
      ins(%6 : memref<64x64xf32>, %expand_shape : memref<1x64xf32>)

CROSS ATTENTION IS WHERE IT SHOWS UP and is not the cause. Stable Diffusion's
U-Net attends image tokens to text tokens, so SQ != SK, and the same failure
appears with SQ == SK -- the rectangle is not the trigger, the missing loop is.
The VAE decoder's attention block is the same shape again.

WHY IT IS FIXED NOW. The gate was a workaround for `_reshape_map` pairing a
unit-axis expand with the extent-1 axis of its group, and that convention has
been corrected. The rank-1 row now corresponds to the axis its data is actually
on, so the two operands agree without anything having to recognise a loop.

torch equivalent: F.scaled_dot_product_attention(q, k, v) with no mask.

attention whose score matrix fits, so there is no KV loop -- the shape cross attention and the VAE decoder both have.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

SQ = 64          # image tokens
SK = 32          # text tokens -- deliberately NOT equal to SQ
D = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def attention_noloop_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                                SQ: tl.constexpr, SK: tl.constexpr,
                                D: tl.constexpr, SCALE: tl.constexpr):
        rq = tl.arange(0, SQ)
        rk = tl.arange(0, SK)
        rd = tl.arange(0, D)

        q = tl.load(q_ptr + rq[:, None] * D + rd[None, :])       # [SQ, D]
        # K read TRANSPOSED, so the dot contracts D without a shuffle.
        kt = tl.load(k_ptr + rk[None, :] * D + rd[:, None])      # [D, SK]
        s = tl.dot(q, kt) * SCALE                                # [SQ, SK]

        m = tl.max(s, axis=1)
        e = tl.exp(s - m[:, None])
        p = e / tl.sum(e, axis=1)[:, None]                       # [SQ, SK]

        v = tl.load(v_ptr + rk[:, None] * D + rd[None, :])       # [SK, D]
        o = tl.dot(p, v)                                         # [SQ, D]
        tl.store(o_ptr + rq[:, None] * D + rd[None, :], o)

    return attention_noloop_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(SQ * D, generator=g, dtype=torch.float32),
            "k": torch.randn(SK * D, generator=g, dtype=torch.float32),
            "v": torch.randn(SK * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(SQ, D)
    k = inputs["k"].reshape(SK, D)
    v = inputs["v"].reshape(SK, D)
    s = (q @ k.T) * (D ** -0.5)
    p = torch.softmax(s, dim=1)
    return {"o": (p @ v).reshape(-1)}


SPEC = KernelSpec(
    name="attention_noloop_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "o_ptr": "*fp32", "SQ": "constexpr", "SK": "constexpr",
               "D": "constexpr", "SCALE": "constexpr"},
    constexprs={"SQ": SQ, "SK": SK, "D": D, "SCALE": D ** -0.5},
    args=[Arg("q", IN, "float32", (SQ * D,)), Arg("k", IN, "float32", (SK * D,)),
          Arg("v", IN, "float32", (SK * D,)),
          Arg("o", OUT, "float32", (SQ * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
