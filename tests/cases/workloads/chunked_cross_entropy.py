"""Liger's fused linear + cross-entropy -- the loss without the logit tensor.

    for each chunk of the vocabulary:
        lg   = x @ w[chunk].T                 [T, BV], produced and dropped
        m, s = online logsumexp update
        z   += lg where the target lands in this chunk
    loss = (m + log s) - z

WHAT IS BEING AVOIDED. `cross_entropy.py` is the same loss and it takes a
MATERIALISED [T, V] logit tile. That tile is the reason this kernel exists: at
Llama-3's V = 128256 and 8192 tokens in flight the fp32 logits are 4 GB, more
than every other activation in the layer put together, and all of it is
manufactured so that one scalar per token can be read out of it. Liger's
fused_linear_cross_entropy chunks the vocabulary, keeps only [T, BV] alive, and
turns O(T*V) into O(T*BV) -- which is what lets the same GPU take 4x the sequence
length at the same batch size. The projection is INSIDE the loss for exactly this
reason: keeping them apart is what forces the buffer to exist.

TWO CONSEQUENCES, AND THEY ARE THE KERNEL.

  NO SECOND PASS, SO THE REDUCTION HAS TO BE ONLINE. The standard logsumexp is
  max, then sum of exp, then log -- two folds over the same buffer. There is no
  buffer here, and the max is not known until the last chunk has been computed
  and thrown away. So the max and the sum-of-exp are carried as running state and
  the SUM IS REPAIRED whenever the max moves:

      m' = max(m, max(lg));   s = s * exp(m - m') + sum(exp(lg - m'))

  the same recurrence flash attention uses, but the state is two rank-1 vectors
  rather than an accumulator tile -- and note the coupling: `s`'s update reads
  the OLD m and the NEW one in one expression. A fusion that understood only
  elementwise chains would drop the rescale and be wrong by whatever the max
  moved by.

  THE GATHER TURNS INTO A MASK. `logits[t, target[t]]` was a `tts.gather` in
  cross_entropy.py -- an in-lane-unstructured address into a real buffer. Here
  there is no address to compute, because the element exists only while its chunk
  is live. So the target logit is picked out by comparing the target against the
  chunk's column ids and folding: `sum(where(target == v, lg, 0))`, which
  contributes in exactly one chunk and zero in the others. Same value, no
  indirection at all -- and it costs a compare and a fold over BV per chunk
  instead of one load.

WHAT IT ASKS OF THE BACKEND. A `tl.dot` inside a loop whose result never reaches
memory and is consumed by three different reductions before it dies; two rank-1
loop-carried states that are read, rescaled and rewritten each iteration; and the
head weight read transposed (`w[v, :]` down the contraction axis), the same
strided read lm_head_tied.py pays for.

BOTH lse AND loss ARE STORED. The online form is not the two-pass form
rearranged, it is a different order of roundings, so `lse` is where any drift
against a reference logsumexp will show and `loss` only inherits it.

NOT MODELLED: the backward. Liger fuses that too -- it recomputes each chunk's
logits to get the gradient rather than storing them, which is the same trade
again. The forward is where the buffer is.

A dot inside a loop whose result never reaches memory, an online logsumexp rescaled every chunk, and cross_entropy.py's gather replaced by a masked fold because there is nothing left to address.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D, V = 64, 32, 128           # tokens, model width, vocabulary
BV = 32                         # vocab chunk -- the only thing held at once
NEG = -1e30                     # the running max's identity, finite on purpose


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def chunked_cross_entropy_kernel(x_ptr, w_ptr, tok_ptr, lse_ptr, loss_ptr,
                                     T: tl.constexpr, D: tl.constexpr,
                                     V: tl.constexpr, BV: tl.constexpr,
                                     NEG: tl.constexpr):
        rt = tl.arange(0, T)
        rd = tl.arange(0, D)
        x = tl.load(x_ptr + rt[:, None] * D + rd[None, :])        # [T, D]
        tok = tl.load(tok_ptr + rt)[:, None]                      # [T, 1]

        m = tl.full((T,), NEG, dtype=tl.float32)     # running max
        s = tl.zeros((T,), dtype=tl.float32)         # running sum of exp
        z = tl.zeros((T,), dtype=tl.float32)         # the target's own logit

        for v0 in tl.static_range(0, V, BV):
            rv = v0 + tl.arange(0, BV)
            # The head weight, transposed: v along the free axis, d down the
            # contraction axis, the same strided read lm_head_tied.py makes.
            wt = tl.load(w_ptr + rv[None, :] * D + rd[:, None])   # [D, BV]
            lg = tl.dot(x, wt)                                    # [T, BV]

            # ONLINE LOGSUMEXP. The old sum is rescaled by how far the max moved;
            # the update reads both the old m and the new one.
            mn = tl.maximum(m, tl.max(lg, axis=1))
            s = s * tl.exp(m - mn) + tl.sum(tl.exp(lg - mn[:, None]), axis=1)
            m = mn

            # THE TARGET LOGIT, as a masked fold instead of a gather -- it is
            # non-zero in exactly one chunk and there is no buffer to address.
            z += tl.sum(tl.where(tok == rv[None, :], lg, 0.0), axis=1)

        lse = m + tl.log(s)
        tl.store(lse_ptr + rt, lse)
        tl.store(loss_ptr + rt, lse - z)

    return chunked_cross_entropy_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32) * 0.5,
            "w": torch.randn(V * D, generator=g, dtype=torch.float32) * 0.5,
            "tok": torch.randint(0, V, (T,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    # The reference DOES materialise the logits -- that is the whole point of
    # the comparison. Nothing else here is allowed to.
    logits = inputs["x"].reshape(T, D) @ inputs["w"].reshape(V, D).t()
    lse = torch.logsumexp(logits, dim=1)
    picked = logits[torch.arange(T), inputs["tok"].long()]
    return {"lse": lse.contiguous(), "loss": (lse - picked).contiguous()}


SPEC = KernelSpec(
    name="chunked_cross_entropy_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "tok_ptr": "*i32",
               "lse_ptr": "*fp32", "loss_ptr": "*fp32", "T": "constexpr",
               "D": "constexpr", "V": "constexpr", "BV": "constexpr",
               "NEG": "constexpr"},
    constexprs={"T": T, "D": D, "V": V, "BV": BV, "NEG": NEG},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("w", IN, "float32", (V * D,)),
          Arg("tok", IN, "int32", (T,)),
          Arg("lse", OUT, "float32", (T,)),
          Arg("loss", OUT, "float32", (T,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
