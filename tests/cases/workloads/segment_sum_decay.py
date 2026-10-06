"""Mamba-2's `segsum` -- the per-chunk decay block, materialised on its own.

For one head with scalar A < 0 and one chunk of Q timesteps:

    da[t]    = softplus(dt_raw[t]) * A            # log-decay, non-positive
    cum[t]   = sum_{k<=t} da[k]                   # inclusive prefix sum
    S[t, s]  = cum[t] - cum[s]  = sum_{k=s+1..t} da[k]      for s <= t
    L[t, s]  = exp(S[t, s])                                  for s <= t, else 0
    dout[t]  = exp(cum[t])                        # weight on the INCOMING state
    dstate[t]= exp(cum[Q-1] - cum[t])             # weight of t in the OUTGOING state

WHAT THIS OBJECT IS. `L` is the whole causal structure of a linear SSM written
out as a matrix: entry (t, s) is how much of input s survives to output t. It is
attention's score matrix with the softmax replaced by a product of decays, which
is the entire content of "State Space Duality". Every chunked linear-attention
kernel in the field builds this block first and then multiplies by it -- Mamba-2,
Gated DeltaNet, RetNet and lightning attention all differ mainly in what goes
into `da`.

WHY IT IS A SEPARATE FILE FROM ssd_chunk.py, WHICH COMPUTES THE SAME MATRIX.
There it is an intermediate: built, immediately consumed by a `tl.dot`, never
written. So when that kernel disagrees with its reference there is no way to say
whether the decay block or the matmul moved. Here the block IS the output, and
with it the two boundary vectors ssd_chunk.py computes and throws away. Three
things are separated that the fused kernel fuses:

  * `S` IS STORED BEFORE THE EXPONENTIAL. S is a difference of prefix sums and
    every entry of it is <= 0 by construction; that is the invariant which makes
    the chunked form stable where a running product of gains is not. A sign
    error, an off-by-one in the prefix, or a lost `softplus` all show up in `S`
    as a structured error. After `exp` they show up as a number near 1 and the
    matmul buries them.
  * THE DECAY IS BUILT FROM A PER-HEAD SCALAR. ssd_chunk.py takes `a` already
    per-timestep. Real Mamba-2 has A as ONE learned scalar per head and dt per
    timestep, so `da` is an outer product of a [Q] vector with an [H] vector and
    the same dt drives H different decay matrices. That is a broadcast the fused
    kernel never sees, and it is why H is a loop axis here.
  * dout AND dstate ARE THE CHUNK BOUNDARY. `L` alone is only the interior. The
    two vectors are what couples chunk c to c-1 and to c+1, and they are the
    parts a chunked implementation gets wrong first, because they are the only
    places `cum` is used un-differenced.

THE PREFIX SUM IS A MASKED ROW SUM, for the reason ssd_chunk.py gives: `cum` is a
scan along the chunk axis, the chunk axis is in-lane, and there is no in-lane
scan. Folding it as `sum_s (s <= t) da[s]` over a [Q, Q] tile turns the scan into
a reduction on the inner axis, which is the only direction that folds. It costs
Q times the arithmetic and buys a form that exists. Mamba-2's own reference
`segsum` does the same thing for a different reason -- it wants the matrix
anyway.

THE MASK IS APPLIED AFTER THE EXPONENTIAL AND NOT BEFORE. Zeroing S first would
leave `exp(0) = 1` above the diagonal, i.e. a fully acausal block that still
looks plausible. The reference uses -inf there; this stores 0 in `S` as a
sentinel and masks `L` separately, so the two are checked independently.

The decay block ssd_chunk.py builds and immediately eats, written out instead: the segment sum before the exp, the masked matrix after it, and the two chunk-boundary vectors that couple the chunks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, NC, Q = 2, 2, 16             # heads, chunks, chunk length
LEN = NC * Q                    # sequence length
NB = H * NC                     # decay blocks produced


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def segment_sum_decay_kernel(dt_ptr, a_ptr, s_ptr, l_ptr, do_ptr, ds_ptr,
                                 H: tl.constexpr, NC: tl.constexpr,
                                 Q: tl.constexpr):
        t = tl.arange(0, Q)[:, None]            # output step within the chunk
        s = tl.arange(0, Q)[None, :]            # input step within the chunk
        rq = tl.arange(0, Q)

        for h in tl.static_range(H):
            Ah = tl.load(a_ptr + h)             # ONE scalar for the whole head
            for ck in tl.static_range(NC):
                # dt is shared across heads; A is not. da is their outer product.
                dtr = tl.load(dt_ptr + ck * Q + s)
                da = tl.log(1.0 + tl.exp(dtr)) * Ah         # [1, Q], <= 0

                # PREFIX SUM AS A MASKED ROW SUM -- the only foldable direction.
                cum = tl.sum(tl.where(s <= t, da, 0.0), axis=1)     # [Q]
                tot = tl.sum(da, axis=1)                            # [1]

                dif = cum[:, None] - cum[None, :]                   # [Q, Q]
                base = (h * NC + ck) * Q * Q
                # S: the segment sum itself, 0 outside the causal triangle.
                tl.store(s_ptr + base + t * Q + s, tl.where(s <= t, dif, 0.0))
                # L: masked AFTER the exp, or the acausal half would read as 1.
                tl.store(l_ptr + base + t * Q + s,
                         tl.where(s <= t, tl.exp(dif), 0.0))

                vb = (h * NC + ck) * Q
                tl.store(do_ptr + vb + rq, tl.exp(cum))         # incoming state
                tl.store(ds_ptr + vb + rq, tl.exp(tot - cum))   # outgoing state

    return segment_sum_decay_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"dt": torch.randn(LEN, generator=g, dtype=torch.float32),
            # A = -exp(A_log): one strictly negative scalar per head.
            "a": -torch.rand(H, generator=g, dtype=torch.float32).exp()}


def reference(inputs):
    import torch
    dt = torch.nn.functional.softplus(inputs["dt"]).reshape(NC, Q)
    da = dt[None, :, :] * inputs["a"][:, None, None]         # [H, NC, Q]
    cum = da.cumsum(dim=2)
    tot = cum[:, :, -1:]
    dif = cum[:, :, :, None] - cum[:, :, None, :]            # [H, NC, Q, Q]
    tri = torch.tril(torch.ones(Q, Q, dtype=torch.bool))
    S = torch.where(tri, dif, torch.zeros((), dtype=torch.float32))
    L = torch.where(tri, dif.exp(), torch.zeros((), dtype=torch.float32))
    return {"s": S.reshape(-1).contiguous(),
            "l": L.reshape(-1).contiguous(),
            "do": cum.exp().reshape(-1).contiguous(),
            "ds": (tot - cum).exp().reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="segment_sum_decay_kernel", kernel=kernel,
    signature={"dt_ptr": "*fp32", "a_ptr": "*fp32", "s_ptr": "*fp32",
               "l_ptr": "*fp32", "do_ptr": "*fp32", "ds_ptr": "*fp32",
               "H": "constexpr", "NC": "constexpr", "Q": "constexpr"},
    constexprs={"H": H, "NC": NC, "Q": Q},
    args=[Arg("dt", IN, "float32", (LEN,)),
          Arg("a", IN, "float32", (H,)),
          Arg("s", OUT, "float32", (NB * Q * Q,)),
          Arg("l", OUT, "float32", (NB * Q * Q,)),
          Arg("do", OUT, "float32", (NB * Q,)),
          Arg("ds", OUT, "float32", (NB * Q,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
