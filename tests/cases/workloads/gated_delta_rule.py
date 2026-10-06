"""Qwen3-Next / Kimi Linear's gated delta rule -- a MATRIX state, updated rank-1.

Per timestep, with state S in R^{dk x dv}:

    S  <- g_t * S                                  # forget, one scalar gate
    u_t = beta_t * (v_t - S^T k_t)                 # what the state gets WRONG
    S  <- S + k_t u_t^T                            # rank-1 write, at k_t
    o_t = S^T q_t

which is the paper's `S_t = g_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`
with the parenthesis expanded so the Householder factor never has to be formed.

WHY IT IS NOT A SCAN. `selective_scan.py` and `ssd_chunk.py` both run
`h <- a_t * h + b_t` where `a_t` is a SCALAR multiplying every component of the
state. The transition is diagonal, so the state's components never talk to each
other and the whole thing factorises into N independent scalar recurrences. The
delta rule's transition is `g_t (I - beta_t k_t k_t^T)` -- identity minus a rank-1
projector. It is not diagonal, it is not even diagonalisable by a fixed basis,
and its action on the state depends on where in key space `k_t` points. That is
the entire modelling claim: a diagonal gate can only forget uniformly, while this
can overwrite the one direction `k_t` names and leave everything else alone,
which is what lets it do in-context associative recall that Mamba cannot.

WHAT THAT COSTS THE BACKEND, AND IT IS THE POINT OF THE FILE

  * THE CARRIED STATE IS A TILE, NOT A VECTOR. selective_scan.py carries [N] as
    an `scf.for` iteration argument; ssd_chunk.py carries [N]. Here the iteration
    argument is [DV, DK] -- 256 live values crossing every loop back-edge, read
    twice and written once per step. Register pressure, not arithmetic, is the
    constraint.

  * TWO REDUCTIONS AND AN OUTER PRODUCT PER STEP, ALL AGAINST THAT TILE. `S^T k`
    and `S^T q` contract the key axis; `k u^T` expands both. An outer product is
    the one shape that is pure broadcast in both directions at once, and it
    writes the whole state every step.

  * THE STATE IS STORED TRANSPOSED, [DV, DK] AND NOT [DK, DV], AND THAT IS
    FORCED. Both contractions run over dk. Whichever axis is on the lanes cannot
    be reduced, so dk has to be the in-lane axis, which puts dv on the lanes and
    turns `S^T k` into an ordinary inner-axis fold. `ssd_chunk.py` reads B
    transposed for exactly this reason at exactly this cost.

  * THE RECURRENCE IS DATA-DEPENDENT IN A WAY A CUMSUM IS NOT. `u_t` needs the
    state BEFORE step t, so the read of S and the write of S are one dependency
    apart with a reduction between them. There is no reassociation available.

WHAT IS NOT MODELLED, DELIBERATELY. The chunked form. Inside a chunk the products
of `(I - beta_j k_j k_j^T)` collapse into a WY representation whose middle factor
is `(I - tril(diag(beta) K K^T, -1))^{-1}` -- the inverse of a unit lower
triangular Q x Q matrix. That is a triangular solve, and forward substitution
over Q rows is a sequential lane-crossing update this target has no form for. So
the loop here is over timesteps, with the state CHECKPOINTED at every chunk
boundary the way the chunked kernel would checkpoint it: the boundary values are
the part the chunked version has to reproduce, so they are what is stored.
Recorded rather than attempted, as in selective_scan.py.

q AND k ARE L2-NORMALISED, and that is load-bearing rather than cosmetic. With
||k|| = 1 and beta in (0, 1), `I - beta k k^T` has singular values in {1, 1-beta}
and is a contraction, so the recurrence is stable; without it the update can
amplify and the state runs away in a few hundred steps. fla normalises inside the
kernel, so it is done inside this one.

NO DRAM ROUND TRIPS, AND IT ONCE CARRIED 24 WITH 8 POINTLESS (rule 8b) -- the
most of any kernel in the suite, and the reason this file used to end in a
findings section. Sixteen came from the cut search (`_cut_one_at_a_time`) and
eight from `_widen_lane_axis_rows` staging a row so a broadcast lands inside a
lane; the eight stamped `torchsim.relayout = array<i64: 0, 0>` were the pointless
ones -- a cut whose two classes then chose the SAME lane axis, so the DMA moved
bytes that did not change place. Measured today: 0 trips here, 0 pointless
anywhere in the suite, and `04-quality.json` plus tests/baselines/tpuv6e.txt are
what hold it there -- a trip reappearing here is a regression, not a finding.

WHAT IS STILL MISSING, and what made those eight cost DRAM at all: a cut that
needs no change of layout wants two scratchpad buffers and a bank-local copy,
not a round trip. This backend has no such copy -- every split goes out to DRAM
and back -- and building one is a different pass's business from choosing an
axis (rule 9).

A [DV,DK] tile as the loop-carried state, updated by a rank-1 outer product against its own prediction error. Every other recurrence in the suite has a diagonal transition and a vector state.
"""

#: WHY THIS REFUSES, TRACED 2026-09-23, AND THE GAP IS IN THE GRAPH. The op it
#: stops on is an identity-mapped `linalg.generic` whose ins is a loop's carried
#: block argument (`%arg14`, riding axis 1 through `buf2`) and whose outs is
#: `buf9`, riding axis 0. That is a cross-lane move wearing an elementwise op,
#: and vectorize is right to refuse it.
#:
#: NOTHING CONSTRAINED THE TWO TO AGREE because the graph cannot see the carry.
#: `_BufferOf` names a block argument `"arg%d" % arg_number`, and a loop body's
#: arguments are numbered in THEIR OWN block -- so the body's `%arg14` is
#: `"arg1"`, the same name as the KERNEL's second argument, and everything read
#: inside a loop resolves to whatever that parameter is.
#:
#: Resolving it instead to the loop's init operand (`FOR_INITS + k`, the two
#: numberings' meeting point, already stated in `lib_axis_maps`) makes the graph
#: see the carry -- and takes this kernel from a refusal to a WRONG ANSWER
#: (3.50791e+07, measured).
#:
#: UPSTREAM HAS THE ARM AND IT IS PORTED.
#: `AddWhileConstraints` (la.cc:833, called from `AddMandatoryConstraints` at
#: 1578) forces the body's RESULT layout onto the body's parameter, the while's
#: operand and the while's result, at `current_priority_ +
#: kNumberOfPropagationRounds` -- above every round, because there is no copy to
#: insert across a back edge. With it, `buf2` goes [1, 0] -> [0, 1], the carry
#: holds one order, no carried buffer is split (measured), and the adapt refusal
#: is gone.
#:
#: WHAT IS LEFT IS THE WRONG ANSWER, unchanged at 3.50791e+07, AND IT IS NARROWED
#: TO ONE OUTPUT. `h` -- the state the loops carry -- PASSES at 5.96e-08, so the
#: carry itself is computed right. `o` fails: 448 of 512 elements outside
#: tolerance, worst at flat index 118 with got=3.5e+07 against want=-0.011, and
#: that magnitude is what an unwritten scratchpad holds. Those 448 were never
#: stored.
#:
#: RULED OUT by turning each rewrite off in turn -- the others still give the
#: same wrong answer, so none of them is it: `MaterialiseReshapes`,
#: `LowerCrossingSlices`, `ReplicateBroadcasts`. `BuildCopies` and
#: `RewriteTransposes` cannot be turned off (the kernel then refuses), but
#: `RewriteTransposes` is measurably identical either way -- 8 in-bank copies, 0
#: trips, both trees -- and the four stores to `o` are the same four ops with the
#: same strides and the same shape of DRAM index.
#:
#: SOLVED, AND THE WRONG ANSWER WAS TWO BUGS IN
#: THE SPLIT, neither of them about loops. The values told it: they were not
#: garbage but the right vector multiplied by ~17 every trip, so the reduce was
#: accumulating into a tile nobody reset.
#:
#:   `_ReadsWhatItRedefines` asked whether any operand still held the ORIGINAL
#:   buffer name. An earlier split in the same pass renames the version this one
#:   reads (`inst.buffers_[n] = nm`), so a reduce that does read its outs
#:   answered False, no `_CarryContents` ran, and the new allocation started
#:   from nothing. Asking the SLOT instead: 3.50791e+07 -> 285.978.
#:
#:   `_CarryContents` then carried from `globals_[buffer]` -- the name the
#:   allocation was made under -- while the version before it already lived in
#:   another global. Carrying from where it lives now: 285.978 -> PASS at
#:   1.19e-07.
#:
#: Both fixes are NEUTRAL on their own (477/44, stage diff empty): nothing
#: reaches them until the graph can see a loop's carry. LANDED TOGETHER WITH
#: THAT, 477 -> 478 with no stage regression.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

DK, DV = 16, 16                 # key dim, value dim -- the state is DK x DV
Q, NC = 8, 4                    # chunk length, chunk count
T = Q * NC                      # sequence length


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gated_delta_rule_kernel(q_ptr, k_ptr, v_ptr, b_ptr, g_ptr,
                                o_ptr, h_ptr,
                                DK: tl.constexpr, DV: tl.constexpr,
                                Q: tl.constexpr, NC: tl.constexpr):
        rk = tl.arange(0, DK)
        rv = tl.arange(0, DV)

        # THE STATE IS A TILE. [DV, DK] and not [DK, DV]: every contraction here
        # runs over dk, so dk has to be the in-lane axis.
        S = tl.zeros((DV, DK), dtype=tl.float32)

        for ck in tl.static_range(NC):
            for j in range(Q):
                t = ck * Q + j
                kt = tl.load(k_ptr + t * DK + rk)
                qt = tl.load(q_ptr + t * DK + rk)
                # L2 normalise: keeps I - beta k k^T a contraction.
                kt = kt / tl.sqrt(tl.sum(kt * kt, axis=0) + 1e-12)
                qt = qt / tl.sqrt(tl.sum(qt * qt, axis=0) + 1e-12)
                vt = tl.load(v_ptr + t * DV + rv)
                beta = tl.sigmoid(tl.load(b_ptr + t))
                # g in (0, 1): exp of a negative softplus, as the model builds it.
                g = tl.exp(-tl.log(1.0 + tl.exp(tl.load(g_ptr + t))))

                S = S * g                                   # forget
                # THE PREDICTION ERROR: reduction over the in-lane key axis.
                u = beta * (vt - tl.sum(S * kt[None, :], axis=1))    # [DV]
                # THE RANK-1 WRITE: broadcast in both directions at once.
                S = S + u[:, None] * kt[None, :]
                tl.store(o_ptr + t * DV + rv, tl.sum(S * qt[None, :], axis=1))

            # Chunk boundary -- the values a chunked implementation must match.
            tl.store(h_ptr + ck * (DV * DK) + rv[:, None] * DK + rk[None, :], S)

    return gated_delta_rule_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(T * DK, generator=g, dtype=torch.float32),
            "k": torch.randn(T * DK, generator=g, dtype=torch.float32),
            "v": torch.randn(T * DV, generator=g, dtype=torch.float32),
            "b": torch.randn(T, generator=g, dtype=torch.float32),
            "g": torch.randn(T, generator=g, dtype=torch.float32) + 2.0}


def reference(inputs):
    import torch
    q = inputs["q"].reshape(T, DK)
    k = inputs["k"].reshape(T, DK)
    v = inputs["v"].reshape(T, DV)
    beta = torch.sigmoid(inputs["b"])
    gate = torch.exp(-torch.nn.functional.softplus(inputs["g"]))
    S = torch.zeros(DV, DK, dtype=torch.float32)
    o = torch.empty(T, DV, dtype=torch.float32)
    hs = torch.empty(NC, DV, DK, dtype=torch.float32)
    for t in range(T):
        kt = k[t] / (k[t].pow(2).sum() + 1e-12).sqrt()
        qt = q[t] / (q[t].pow(2).sum() + 1e-12).sqrt()
        S = S * gate[t]
        u = beta[t] * (v[t] - (S * kt[None, :]).sum(dim=1))
        S = S + u[:, None] * kt[None, :]
        o[t] = (S * qt[None, :]).sum(dim=1)
        if (t + 1) % Q == 0:
            hs[t // Q] = S
    return {"o": o.reshape(-1).contiguous(), "h": hs.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="gated_delta_rule_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "b_ptr": "*fp32", "g_ptr": "*fp32", "o_ptr": "*fp32",
               "h_ptr": "*fp32", "DK": "constexpr", "DV": "constexpr",
               "Q": "constexpr", "NC": "constexpr"},
    constexprs={"DK": DK, "DV": DV, "Q": Q, "NC": NC},
    args=[Arg("q", IN, "float32", (T * DK,)),
          Arg("k", IN, "float32", (T * DK,)),
          Arg("v", IN, "float32", (T * DV,)),
          Arg("b", IN, "float32", (T,)),
          Arg("g", IN, "float32", (T,)),
          Arg("o", OUT, "float32", (T * DV,)),
          Arg("h", OUT, "float32", (NC * DV * DK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
