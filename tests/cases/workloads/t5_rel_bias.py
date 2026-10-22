"""T5's relative position bias -- log-spaced buckets, then a table lookup.

    rel     = j - i                              memory position minus query
    bucket  = f(rel)                             LOG-SPACED, 32 of them
    bias    = table[h, bucket]                   learned, one row per head
    p       = softmax(scores + bias)

WHY BUCKETS AT ALL. A learned bias per (i, j) pair would be `S^2` parameters per
head and would not generalise past the training length. T5 learns one value per
BUCKET, and the bucketing is what makes distance 5 and distance 6 different
while distance 100 and distance 110 are the same:

    |rel| < 16        one bucket each -- exact, because near context is where
                      the precise offset matters
    |rel| >= 16       log-spaced up to max_distance, then saturating -- "far"
                      stops being resolved

Half the buckets go to positive `rel` and half to negative, which is what makes
it BIDIRECTIONAL: an encoder needs to tell "8 tokens before" from "8 after", and
ALiBi and RoPE both cannot.

    bucket = (rel > 0) * 16
           + (n < 8 ? n : 8 + log(n/8)/log(64/8) * 8)      n = |rel|, clamped

THE INDEX IS COMPUTED, NOT LOADED, AND IT IS NOT AFFINE. `log(n)` of an index
difference is where a structured pointer analysis gives up, so the table read is
gathered over both tile axes by the bucket itself (tts.make_gather_scatter_tptr,
gather_scatter_dims = [0, 1]) with rows of one element -- one index per tile
ELEMENT rather than one per row, read element by element in one transfer. That
is the shape this kernel is here to have: every other gather in the suite reads
rows.

  It is also entirely data-independent. The bucket of (i, j) is fixed by the
  shape, so a compiler that folded index arithmetic could build the table read
  as a constant -- nothing here does, and the general case (a real T5 runs this
  once per layer, cached) does not need it to.

FLOAT LOG ON AN INTEGER DISTANCE. The division and log are done in f32 and the
result truncated back, matching `torch`'s own implementation -- so the boundary
between two buckets lands where torch puts it rather than where exact integer
arithmetic would.

torch equivalent: T5Attention._relative_position_bucket, then an embedding
lookup and a softmax.

T5: a log-spaced bucket computed from indices, then a PER-ELEMENT table gather -- the expensive gather shape, unlike every other one in this suite.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, H = 32, 4
NB = 32            # num_buckets
MAXD = 64          # max_distance


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def t5_rel_bias_kernel(sc_ptr, tab_ptr, o_ptr,
                           S: tl.constexpr, H: tl.constexpr,
                           NB: tl.constexpr, MAXD: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, S)

        # THE BUCKET, exactly as T5 computes it. Half the buckets carry the
        # sign, so the bidirectional case works on |rel| and adds an offset.
        rel = j[None, :] - i[:, None]
        half: tl.constexpr = NB // 2
        exact: tl.constexpr = half // 2

        sign = tl.where(rel > 0, half, 0)
        n = tl.abs(rel)
        small = n < exact
        # log-spaced for the rest: exact + log(n/exact)/log(MAXD/exact) * (half-exact)
        fn = tl.maximum(n, exact).to(tl.float32)
        big = exact + (tl.log(fn / exact) / tl.log(MAXD * 1.0 / exact)
                       * (half - exact)).to(tl.int32)
        big = tl.minimum(big, half - 1)
        bucket = sign + tl.where(small, n, big)

        for h in tl.static_range(H):
            # PER-ELEMENT GATHER: one index per (i, j), not one per row.
            b = tl.load(tab_ptr + h * NB + bucket)
            sc = tl.load(sc_ptr + h * S * S + i[:, None] * S + j[None, :])
            z = sc + b
            m = tl.max(z, axis=1)
            ex = tl.exp(z - m[:, None])
            p = ex / tl.sum(ex, axis=1)[:, None]
            tl.store(o_ptr + h * S * S + i[:, None] * S + j[None, :], p)

    return t5_rel_bias_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"sc": torch.randn(H * S * S, generator=g, dtype=torch.float32),
            "tab": torch.randn(H * NB, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    sc = inputs["sc"].reshape(H, S, S)
    tab = inputs["tab"].reshape(H, NB)
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    rel = j - i
    half, exact = NB // 2, NB // 4
    sign = torch.where(rel > 0, torch.full_like(rel, half),
                       torch.zeros_like(rel))
    n = rel.abs()
    small = n < exact
    fn = torch.maximum(n, torch.full_like(n, exact)).float()
    big = exact + ((fn / exact).log() / torch.tensor(MAXD / exact).log()
                   * (half - exact)).int()
    big = torch.minimum(big, torch.full_like(big, half - 1))
    bucket = sign + torch.where(small, n, big)
    out = torch.empty_like(sc)
    for h in range(H):
        out[h] = torch.softmax(sc[h] + tab[h][bucket], dim=-1)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="t5_rel_bias_kernel",
    kernel=kernel,
    signature={"sc_ptr": "*fp32", "tab_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "H": "constexpr", "NB": "constexpr",
               "MAXD": "constexpr"},
    constexprs={"S": S, "H": H, "NB": NB, "MAXD": MAXD},
    args=[Arg("sc", IN, "float32", (H * S * S,)),
          Arg("tab", IN, "float32", (H * NB,)),
          Arg("o", OUT, "float32", (H * S * S,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
