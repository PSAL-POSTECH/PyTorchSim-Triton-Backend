"""Liger's chunked DPO loss -- a PAIRED preference objective: two sequences, two
chunked logprob sweeps, and one pairwise sigmoid at the end.

    for each chunk of the vocabulary, for BOTH sequences at once:
        lg   = x @ w[chunk].T                  produced and dropped
        m, s = online logsumexp update
        z   += lg where this position's target lands in this chunk
    lp[r]  = z[r] - (m[r] + log s[r])          per-position logprob, 2T of them
    L      = [sum_t lp[chosen, t], sum_t lp[rejected, t]]
    u      = beta * ((L_c - L_r) - (ref_c - ref_r))
    loss   = -log sigmoid(u)

WHAT DPO IS AND WHY THE LOSS LOOKS LIKE THIS. RLHF used to mean training a reward
model on preference pairs and then running PPO against it. Rafailov et al. (2023)
showed the optimal policy for that objective has a closed form, and that
substituting it back turns the whole pipeline into ONE supervised loss on the
pairs -- no reward model, no sampling, no value head. Every open fine-tune that
says "DPO", "IPO", "ORPO" or "SimPO" is a variant of the expression above, and
TRL calls Liger's kernel for it by default.

  The `ref_c - ref_r` term is the frozen reference model's answer to the same
  pair, computed once and passed in. `beta` is how hard the policy is allowed to
  move away from it; 0.1 is the usual value.

WHY IT IS NOT chunked_cross_entropy.py WITH A DIFFERENT EPILOGUE. That kernel and
this one share the vocabulary-chunking machinery -- the whole reason both exist
is that [T, V] logits at Llama-3's V = 128256 do not fit -- and everything above
the last four lines is the same online logsumexp. The difference is that
cross-entropy's output is PER TOKEN and this one's is ONE SCALAR FOR TWO WHOLE
SEQUENCES, which changes the dataflow twice over:

  1. TWO SEQUENCES ARE LIVE IN ONE KERNEL. Chosen and rejected are stacked on the
     row axis as [2T, D] and swept together, because they share the head weight
     and sweeping them separately would read all of `w` twice -- the dominant
     cost. So the row axis is really (sequence, position) flattened, and every
     chunk's `tl.dot` is contracting two different sequences against one weight
     tile at the same time. sampler_cfg.py also has two batches, but it combines
     them elementwise at matching offsets; here they are combined only after a
     REDUCTION OVER DIFFERENT LENGTHS OF THE ROW AXIS.

  2. THE ROW AXIS HAS TO BE FOLDED, AND THE FOLD IS THE INTERESTING PART. Summing
     logprobs over a sequence is a reduction along the axis the banks hold, and
     there is no lane-crossing primitive. So the flat [2T] vector is RESHAPED to
     [2, T] -- putting the position axis INSIDE a lane and the sequence axis on
     the lanes -- and reduced on axis 1, the only direction available. That
     reshape is the kernel's one real relayout, and it is exactly the operation
     that turns "a batch of tokens" into "a pair of sequences".

  3. AND THEN THE PAIR HAS TO MEET. `L_c - L_r` needs the two ends of a length-2
     tile as separate scalars, which no value can be sliced into, so the pair
     round-trips through memory and comes back as two scalar loads. Storing it is
     not a workaround: TRL logs both sequence logprobs and both implicit rewards
     every step, so a real kernel writes them anyway.

THE FINAL NONLINEARITY IS WRITTEN AS A STABLE SOFTPLUS. `-log sigmoid(u)` is
`log(1 + exp(-u))`, which overflows for u very negative; the kernel computes
`max(-u, 0) + log(1 + exp(-|u|))`, the identical value with the exponent's
argument pinned at or below zero. u is a difference of two sums of T logprobs, so
it is routinely tens in magnitude on real data and the naive form genuinely does
blow up -- this is not defensive decoration.

THREE OUTPUTS ON PURPOSE. `lp` is the chunked sweep alone, `lps` adds the reshape
and the fold, `loss` adds the pairing. A disagreement can be placed exactly.

NOT MODELLED: the backward, and the NLL-on-chosen auxiliary term some recipes add
(that one is chunked_cross_entropy.py, unchanged, on half the rows).

GREEN SINCE 2026-09-05, err 8.34e-07, AND THE WAY OUT IS THROUGH MEMORY. The
refusal here used to say the way back from the lanes is `vfmv.f.s` and that this
target has not got one. It has: spike's `vfmv_f_s.h` reads
`P.VU.elt<uint32_t>(vs2, 0, 0)` -- element 0 of vector unit 0 -- into the scalar
file, and it is the INTEGER `vmv.x.s` whose body sits commented out under
`assert(0); // INVALID INSTRUCTION IN TPU`.

But llc does not emit it. Given an extract feeding a scalar store it keeps the
value in the lanes and writes it with a ONE-ELEMENT VECTOR STORE:

    vsetivli zero, 1, e32, mf2, ta, ma
    vfadd.vv v8, v9, v8
    vse32.v  v8, (a7)          a7 comes from `ld a7, 8(a2)`: an ARGUMENT

and VI_ST addresses the scratchpad only, so that exits with
`VI_ST ERROR: Access address overflow: 0xd8003070` -- 12400 bytes past the
`spad_bank_top` symbol.

So the value does not come back through the REGISTER FILE at all. It is written
where a vector store is addressable -- a one-element scratchpad buffer -- and
read back with a scalar load, whose flat address is bank 0's. Every lane holds
the same number, because the scalar this replaced was computed once.

Two sequences swept through one chunked logsumexp against one head weight, then folded apart by a [2T] -> [2, T] reshape so the per-sequence sum lands on the inner axis, then paired through a scalar round trip. `lp` isolates the sweep, `lps` the reshape and fold, `loss` the pairing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T = 16                  # positions per sequence
PT = 2 * T              # (sequence, position) flattened: chosen then rejected
D = 16                  # model width
V = 64                  # vocabulary
BV = 16                 # vocab chunk -- the only thing held at once
BETA = 0.1              # DPO's KL strength, TRL's default
NEG = -1e30             # the running max's identity, finite on purpose


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def chunked_dpo_loss_kernel(x_ptr, w_ptr, tok_ptr, ref_ptr,
                                lp_ptr, lps_ptr, loss_ptr,
                                T: tl.constexpr, PT: tl.constexpr,
                                D: tl.constexpr, V: tl.constexpr,
                                BV: tl.constexpr, BETA: tl.constexpr,
                                NEG: tl.constexpr):
        rr = tl.arange(0, PT)
        rd = tl.arange(0, D)

        # BOTH SEQUENCES AT ONCE: rows 0..T-1 are chosen, T..2T-1 rejected, and
        # the head weight below is read exactly once for the pair.
        x = tl.load(x_ptr + rr[:, None] * D + rd[None, :])        # [PT, D]
        tok = tl.load(tok_ptr + rr)[:, None]                      # [PT, 1]

        m = tl.full((PT,), NEG, dtype=tl.float32)    # running max
        s = tl.zeros((PT,), dtype=tl.float32)        # running sum of exp
        z = tl.zeros((PT,), dtype=tl.float32)        # the target's own logit

        for v0 in tl.static_range(0, V, BV):
            rv = v0 + tl.arange(0, BV)
            wt = tl.load(w_ptr + rv[None, :] * D + rd[:, None])   # [D, BV]
            lg = tl.dot(x, wt)                                    # [PT, BV]

            mn = tl.maximum(m, tl.max(lg, axis=1))
            s = s * tl.exp(m - mn) + tl.sum(tl.exp(lg - mn[:, None]), axis=1)
            m = mn

            # No buffer to address, so the target logit is a masked fold -- it
            # is non-zero in exactly one chunk. chunked_cross_entropy.py's move.
            z += tl.sum(tl.where(tok == rv[None, :], lg, 0.0), axis=1)

        lp = z - (m + tl.log(s))                                  # [PT]
        tl.store(lp_ptr + rr, lp)

        # THE FOLD. Summing over a sequence is a reduction along the lane axis,
        # which has no primitive -- so (sequence, position) is reshaped apart,
        # putting position INSIDE a lane, and reduced on the inner axis.
        lps = tl.sum(tl.reshape(lp, (2, T)), axis=1)              # [2]
        tl.store(lps_ptr + tl.arange(0, 2), lps)

        # THE PAIRING. A length-2 value cannot be sliced, so the two sequence
        # logprobs come back as scalars.
        lc = tl.load(lps_ptr + 0)
        lr = tl.load(lps_ptr + 1)
        u = BETA * ((lc - lr) - (tl.load(ref_ptr + 0) - tl.load(ref_ptr + 1)))

        # -log sigmoid(u), with the exponent pinned at or below zero.
        tl.store(loss_ptr,
                 tl.maximum(-u, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(u))))

    return chunked_dpo_loss_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(PT * D, generator=g, dtype=torch.float32) * 0.5,
            "w": torch.randn(V * D, generator=g, dtype=torch.float32) * 0.5,
            "tok": torch.randint(0, V, (PT,), generator=g, dtype=torch.int32),
            # the frozen reference model's summed logprobs for the same pair
            "ref": torch.tensor([-41.5, -44.25], dtype=torch.float32)}


def reference(inputs):
    import torch
    # The reference materialises the logits for the whole pair -- which is the
    # buffer the chunked form exists to avoid.
    logits = inputs["x"].reshape(PT, D) @ inputs["w"].reshape(V, D).t()
    lse = torch.logsumexp(logits, dim=1)
    picked = logits[torch.arange(PT), inputs["tok"].long()]
    lp = picked - lse
    lps = lp.reshape(2, T).sum(dim=1)

    u = BETA * ((lps[0] - lps[1]) - (inputs["ref"][0] - inputs["ref"][1]))
    loss = torch.nn.functional.softplus(-u)
    return {"lp": lp.contiguous(), "lps": lps.contiguous(),
            "loss": loss.reshape(1).contiguous()}


SPEC = KernelSpec(
    name="chunked_dpo_loss_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "tok_ptr": "*i32",
               "ref_ptr": "*fp32", "lp_ptr": "*fp32", "lps_ptr": "*fp32",
               "loss_ptr": "*fp32", "T": "constexpr", "PT": "constexpr",
               "D": "constexpr", "V": "constexpr", "BV": "constexpr",
               "BETA": "constexpr", "NEG": "constexpr"},
    constexprs={"T": T, "PT": PT, "D": D, "V": V, "BV": BV, "BETA": BETA,
                "NEG": NEG},
    args=[Arg("x", IN, "float32", (PT * D,)),
          Arg("w", IN, "float32", (V * D,)),
          Arg("tok", IN, "int32", (PT,)),
          Arg("ref", IN, "float32", (2,)),
          Arg("lp", OUT, "float32", (PT,)),
          Arg("lps", OUT, "float32", (2,)),
          Arg("loss", OUT, "float32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
