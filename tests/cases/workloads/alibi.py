"""ALiBi -- BLOOM's position scheme, which is not an embedding at all.

    bias[h, i, j] = -m_h * (i - j)          j <= i, and -inf above the diagonal
    p             = softmax(scores + bias)  over j

    m_h = 2 ** (-8 (h+1) / H)               geometric, head 0 steepest

NO POSITION INFORMATION IS ADDED TO THE TOKENS. Every other scheme in this suite
puts position into the VALUES -- a learned vector added to the embedding (GPT-2),
a rotation applied to q and k (RoPE). ALiBi puts it into the SCORES, as a
penalty proportional to how far back the key is, and the token representations
never learn where they are.

  Press et al. (2021) introduced it for length extrapolation: a model trained at
  1024 keeps working at 2048, because "further away matters less" is a fixed
  linear function rather than something learned only over the training range.

EVERY HEAD GETS A DIFFERENT SLOPE, and that is the whole design. The slopes are
a geometric series `2^-8/H, 2^-16/H, ...`, so head 0 can barely see past a few
tokens while the last head is nearly unbiased. One head is a local window and
another is global, without either being told to be.

THE BIAS IS NOT A TABLE. `-m_h * (i - j)` is computed from the indices, so there
is nothing to look up and nothing to train -- BLOOM-176B stores zero parameters
for position. The triangular distance matrix `i - j` is `linalg.index`
arithmetic, which this backend builds inside a lane for free.

WHAT IT ASKS OF THIS MACHINE, and why the head axis is a loop here. A head-major
tile `[H, S, S]` would put the per-head slope on the head axis -- a broadcast
along an axis that is neither the lane axis nor the vector axis, which
vectorize refuses (measured: the rank-3 outer product in moe_router hit the
same wall). With H a static loop each tile stays rank 2 and the slope is a
scalar, which is the shape that lowers. The cost is that the head axis is not
tested as a LAYOUT question, which it would be on a machine with more lanes than
this one has heads.

torch equivalent:
    d = torch.arange(S)[:, None] - torch.arange(S)[None, :]
    bias = torch.where(d >= 0, -m * d, -inf)
    p = softmax(scores + bias, dim=-1)

BLOOM: position as a per-head linear penalty on the SCORES, built from indices with no table and no parameters.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, H = 64, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def alibi_kernel(q_ptr, o_ptr, S: tl.constexpr, H: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)
        # THE DISTANCE MATRIX, from indices only -- no table, no parameters.
        dist = i[:, None] - j[None, :]
        causal = dist >= 0

        for h in tl.static_range(H):
            # m_h = 2 ** (-8 (h+1) / H), written as exp since the exponent is
            # not an integer and this triton has no float pow.
            m = tl.exp(-8.0 * (h + 1) / H * 0.6931471805599453)

            sc = tl.load(q_ptr + h * S * S + i[:, None] * S + j[None, :])
            b = sc - m * dist.to(tl.float32)
            # The mask is applied where the bias is, not before it: -inf plus a
            # finite penalty is still -inf, but a finite score times a masked
            # bias is not what softmax wants to see.
            b = tl.where(causal, b, float("-inf"))

            mx = tl.max(b, axis=1)
            ex = tl.exp(b - mx[:, None])
            p = ex / tl.sum(ex, axis=1)[:, None]
            tl.store(o_ptr + h * S * S + i[:, None] * S + j[None, :], p)

    return alibi_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(H * S * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    sc = inputs["q"].reshape(H, S, S)
    d = (torch.arange(S)[:, None] - torch.arange(S)[None, :]).float()
    out = torch.empty_like(sc)
    for h in range(H):
        m = 2.0 ** (-8.0 * (h + 1) / H)
        b = sc[h] - m * d
        b = torch.where(d >= 0, b, torch.full_like(b, float("-inf")))
        out[h] = torch.softmax(b, dim=-1)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="alibi_kernel",
    kernel=kernel,
    signature={"q_ptr": "*fp32", "o_ptr": "*fp32", "S": "constexpr",
               "H": "constexpr"},
    constexprs={"S": S, "H": H},
    args=[Arg("q", IN, "float32", (H * S * S,)),
          Arg("o", OUT, "float32", (H * S * S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
