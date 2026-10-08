"""Four reductions over one matmul result, two of them fighting the array.

    c = tl.dot(a, b)
    tl.store(..., tl.sum(c, axis=1));  tl.store(..., tl.sum(c, axis=0))
    tl.store(..., tl.max(c, axis=1));  tl.store(..., tl.max(c, axis=0))

mixed_reduce_both_axes has one reduction per axis, which is the smallest
conflict a copy can resolve -- and one the FLIP can halve, since transposing
turns every loser into a winner and back. Two per axis cannot be halved: either
orientation leaves two demands losing, so the answer has to be two copies.

WHAT IT IS ACTUALLY TESTING
The cut used to be chosen by scanning for linalg.reduce and asking whether its
folded axis was claimed. That answers the two-reduction case correctly by luck:
the thing scanned for is the thing that happens to lose. The rule underneath is

    the array's MUST is kept, because a matmul has no other way to be fed
    every other demand on that component moves

which lib_lane_axis.conflicts() states over DEMANDS rather than over op names. Two
cuts fall out of it here, and neither is written down anywhere as "reduce".

It is also the first kernel where the flip is UNDONE. Flipping is tried first
because it is free, but here it converts two losers into two different losers,
so keeping it would buy nothing and cost a round trip; fit_layout_to_lanes measures
the conflict count either way and flips back.

WHAT IT FOUND
Two things that were wrong before this kernel existed, both silent:

  absorb_transpose folded a transpose only when it had exactly ONE consumer.
  `tl.sum(c, 1)` and `tl.max(c, 1)` arrive as two reductions behind one
  transpose (the linalg lowering's cse merged them), which left the transpose with no
  lowering at all.

  the DMA inline asm had no `~{memory}` clobber, so LLVM moved a scratchpad
  fill ABOVE the MVOUT that was still reading that buffer. `d` came back -inf in
  every lane. See lower_to_dma.CONSTRAINTS.

THE TWO CUTS SHARE ONE COPY
Both demands want the same layout of the same buffer, so one relayout feeds both
reductions. That is not a cost model, which is what it looked like at first: the
two copies came out byte for byte identical, source, scratch and stamp alike, so
one tile was making the round trip twice for nothing. fit_layout_to_lanes merges
them AFTER the forest resolves, because that is the first moment "the same
layout" is a recorded fact rather than a guess about what is going to be
decided. Measured: two MVOUT/MVIN pairs became one, one fewer scratchpad buffer,
and every output identical to the digit.

Sharing between demands that want DIFFERENT layouts would be a cost model. No
kernel here asks for that yet.

WHAT PROVES IT FIRED
Four outputs are right, and 04-transformed.mlir holds ONE MVOUT/MVIN pair over the
compiler's scratch argument, going vlane 1 -> vlane 0, read by two reductions.

IT HOLDS TWO, AND BOTH ARE WANTED (rule 8b)
The paragraph above describes one of them. Read off 04-transformed.mlir today:

    MVOUT / MVIN   torchsim.relayout = array<i64: 1, 0>    the merged pair
    MVOUT / MVIN   torchsim.relayout = array<i64: 1, 0>    the second, likewise

Both cross vlane 1 -> vlane 0, so neither arrives at the banking it left with.
This used to read `[2 round trips, 1 POINTLESS]`: the second pair was stamped
`[1, 1]`, and mixed_reduce_both_axes.py carried the identical one. The suite now
has 0 pointless trips anywhere.

Two demands lose to the array at once, so the cut has to be a rule over demands rather than a scan for linalg.reduce.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# K = 2 * BK so the accumulator stays loop-carried.
M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_reduce_four_ways_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m, tl.sum(acc, axis=1))
        tl.store(d_ptr + offs_n, tl.sum(acc, axis=0))
        tl.store(e_ptr + offs_m, tl.max(acc, axis=1))
        tl.store(f_ptr + offs_n, tl.max(acc, axis=0))

    return mixed_reduce_four_ways_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    p = torch.matmul(inputs["a"], inputs["b"])
    return {"c": p.sum(dim=1), "d": p.sum(dim=0),
            "e": p.amax(dim=1), "f": p.amax(dim=0)}


SPEC = KernelSpec(
    name="mixed_reduce_four_ways_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "d_ptr": "*fp32", "e_ptr": "*fp32", "f_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("c", OUT, "float32", (M,)),
        Arg("d", OUT, "float32", (N,)),
        Arg("e", OUT, "float32", (M,)),
        Arg("f", OUT, "float32", (N,)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # Everything here is a reduction over the matmul's own reassociated sum,
    # so even the max inherits its error rather than being exact.
    rtol=1e-3, atol=1e-3,
)
