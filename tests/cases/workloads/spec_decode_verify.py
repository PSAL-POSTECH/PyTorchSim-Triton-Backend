"""Speculative decoding's acceptance test -- and the residual that makes it exact.

For each of the K drafted positions, with `tok[j]` the token the draft proposed,
`q` the draft's distribution and `p` the target's:

    accept j   iff  u[j] * q[j, tok[j]] <= p[j, tok[j]]
    n          =    length of the leading run of accepts
    resid      =    normalise(relu(p[n] - q[n]))       at the first rejection only

WHY THIS IS THE INTERESTING PART OF SPECULATIVE DECODING. Decode is memory bound:
a forward pass for ONE token reads every weight in the model, so the arithmetic
units idle. A small draft model proposes K tokens cheaply, the target scores all
K+1 positions in a SINGLE pass for barely more than the cost of one, and this
kernel decides how many of them survive. Two to three tokens per pass, for free.

  And "for free" is a theorem, not a heuristic. Accepting with probability
  min(1, p/q) and, on rejection, resampling from the normalised positive part of
  `p - q` gives a token distributed EXACTLY as p -- the two branches' masses add
  back to p at every token. That is why serving stacks turn it on by default and
  do not treat it as a quality knob. THE RESIDUAL LINE IS THE PROOF: resample
  from `p` instead and the output is quietly biased toward tokens the draft
  already liked, in a way no eval short of a KL test would catch.

  `u * q <= p` rather than `u <= p / q`. Same predicate, and q is routinely
  denormal-small on a token some other position's draft proposed -- the division
  is the only place this can produce an inf.

WHAT IS STRUCTURALLY NEW, AGAINST sampling.py AND top_p_sampling.py. Those reduce
and scan along the VOCAB axis, one row at a time, and the vocab axis is the only
axis they have. This kernel uses BOTH AXES OF THE SAME [K, V] BUFFER, in
different layouts, for different operations:

  * the acceptance test is a scan along the POSITION axis. `pre` is an inclusive
    cumsum of rejections compared against zero -- the run-length primitive, and
    it has to run across positions, which are the ROWS of the logit buffer. So
    that half is laid out [1, K]: positions in-lane, because a scan cannot cross
    lanes and the position axis is the one being scanned.
  * the residual is a reduction along the VOCAB axis, which needs positions on
    the lanes -- [K, V], the ordinary orientation.

Two orientations of one buffer inside one kernel, and a value crossing between
them: `n` comes out of the position-axis scan as a single number and then selects
a ROW of the vocab-axis tile. It is expressed as `row == n` rather than as an
address, because pulling `n` out to a scalar is exactly the lane extract this
machine refuses (rulebook fact 6) -- the comparison keeps it in a vector.

THE GATHER IS THE OTHER ORIENTATION TOO. `p[j, tok[j]]` is cross_entropy.py's
gather turned ninety degrees: there the index is one per LANE, here it is one per
element along the vector axis, and it addresses two different buffers at the same
place.

ALL-ACCEPTED IS A REAL CASE and it falls out: `n == K`, no row matches, and the
residual is all zero. That is correct -- when nothing is rejected the extra token
comes from the target's own distribution at position K and no correction exists.

FAILS TODAY, AND NO LONGER FOR THE REASON IT USED TO. It used to die at stage 6
with `llc spilled a vector register GROUP` (1 x vs8r.v). It no longer spills, and
now 43 of 1024 elements of `r` are wrong -- every one of them zero where the
reference has a value, i.e. the `where` never fires.

WHAT WAS MEASURED. With the spill refusal suppressed and every loop left at the
full 64, the kernel verifies EXACTLY (0 error on all three outputs). Narrowing
either of the two loops over the residual buffer to 32 -- and it does not matter
WHICH, each alone reproduces it -- gives the wrong answer, and the value that
goes wrong is the broadcast of the accepted-length scalar out of its 1-element
buffer. Making every loop over that buffer agree on one width does not fix it
either (tried at 8: same failure), so this is not the sub-byte addressing hazard
that mask_i1_buffer_two_widths pins.

Same shape as convnextv2_grn's: a scalar that is a REDUCTION RESULT, read back
per lane and broadcast along the vector axis, correct only when the loop reading
it runs at the tile's full width. Not chased further.

One [K,V] buffer read in both orientations: a scan across positions and a reduction across the vocab. `a` and `n` are integers and exact, so a mismatch there is the scan; drift can only be in `r`.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 04-passes/04-torchsim-materialize-layouts.mlir at 10; what that pins is remove-layout-conversions NOT rematerializing compute into the sum's one-bank layout (the patch's computesOnData); with the GPU cost model every 1x16 tile above the sum goes one-bank and the fold then crosses lanes at each, 20.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, V = 16, 64                   # drafted positions, vocabulary
FLOOR = 1e-9                    # guards the residual's normaliser


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def spec_decode_verify_kernel(p_ptr, q_ptr, tok_ptr, u_ptr,
                                  a_ptr, n_ptr, r_ptr,
                                  K: tl.constexpr, V: tl.constexpr,
                                  FLOOR: tl.constexpr):
        # LAYOUT ONE: positions in-lane, because the scan runs over positions.
        j = tl.arange(0, K)[None, :]
        tok = tl.load(tok_ptr + j)
        pt = tl.load(p_ptr + j * V + tok)          # target's mass on the draft
        qt = tl.load(q_ptr + j * V + tok)          # draft's own mass on it

        # Cross-multiplied, so a vanishing draft probability cannot divide.
        ok = tl.load(u_ptr + j) * qt <= pt
        # THE PREFIX: a position survives only if nothing before it failed.
        pre = tl.cumsum(tl.where(ok, 0.0, 1.0), axis=1) == 0.0
        tl.store(a_ptr + j, pre.to(tl.int32))
        n = tl.sum(tl.where(pre, 1.0, 0.0), axis=1)              # (1,)
        tl.store(n_ptr + tl.arange(0, 1), n.to(tl.int32))

        # LAYOUT TWO: positions on the lanes, because the reduction is over V.
        rk = tl.arange(0, K)[:, None]
        rv = tl.arange(0, V)[None, :]
        off = rk * V + rv
        d = tl.maximum(tl.load(p_ptr + off) - tl.load(q_ptr + off), 0.0)
        d = d / tl.maximum(tl.sum(d, axis=1), FLOOR)[:, None]

        # `n` selects a row by COMPARISON, never by address -- it has to stay in
        # a vector. Nothing matches when every position was accepted.
        tl.store(r_ptr + off, tl.where(rk == n[None, :], d, 0.0))

    return spec_decode_verify_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Real distributions on both sides, and a draft that is close but not equal
    # to the target -- which is the regime the acceptance rate is decided in.
    base = torch.randn(K, V, generator=g, dtype=torch.float32) * 2.0
    p = torch.softmax(base, dim=1)
    q = torch.softmax(base + torch.randn(K, V, generator=g,
                                         dtype=torch.float32) * 0.8, dim=1)
    return {"p": p.reshape(-1).contiguous(),
            "q": q.reshape(-1).contiguous(),
            # what the draft sampled: from q, so most positions are acceptable
            "tok": torch.multinomial(q, 1, generator=g).reshape(-1).to(
                torch.int32),
            "u": torch.rand(K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = inputs["p"].reshape(K, V)
    q = inputs["q"].reshape(K, V)
    j = torch.arange(K)
    t = inputs["tok"].long()
    ok = inputs["u"] * q[j, t] <= p[j, t]
    pre = (~ok).to(torch.int32).cumsum(0) == 0
    n = int(pre.sum())
    d = (p - q).clamp_min(0.0)
    d = d / d.sum(dim=1, keepdim=True).clamp_min(FLOOR)
    sel = (j == n)[:, None]
    return {"a": pre.to(torch.int32).contiguous(),
            "n": torch.tensor([n], dtype=torch.int32),
            "r": torch.where(sel, d, torch.zeros_like(d)).reshape(
                -1).contiguous()}


SPEC = KernelSpec(
    name="spec_decode_verify_kernel", kernel=kernel,
    signature={"p_ptr": "*fp32", "q_ptr": "*fp32", "tok_ptr": "*i32",
               "u_ptr": "*fp32", "a_ptr": "*i32", "n_ptr": "*i32",
               "r_ptr": "*fp32", "K": "constexpr", "V": "constexpr",
               "FLOOR": "constexpr"},
    constexprs={"K": K, "V": V, "FLOOR": FLOOR},
    args=[Arg("p", IN, "float32", (K * V,)),
          Arg("q", IN, "float32", (K * V,)),
          Arg("tok", IN, "int32", (K,)),
          Arg("u", IN, "float32", (K,)),
          Arg("a", OUT, "int32", (K,)),
          Arg("n", OUT, "int32", (1,)),
          Arg("r", OUT, "float32", (K * V,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
    expect_ir={"04-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 10}},
)
