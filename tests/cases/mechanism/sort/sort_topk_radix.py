"""1-bit MSD radix selection for a row-wise top-k threshold.

This kernel does NOT fully sort each row and does NOT compact the selected values.
Instead, it finds the K-th largest value (the threshold) by following one radix
bucket at a time from the most-significant bit (MSB) to the least-significant bit
(LSB).

High-level idea:
    1. Convert each float32 value into an integer key whose ordering matches the
       original float ordering.
    2. Starting from bit 31, count how many current candidates have that bit set.
    3. Decide whether the K-th largest key lies in the bit-1 or bit-0 half.
    4. Record that decision in `prefix` and continue with the next bit.
    5. After all 32 bits are decided, convert `prefix` back to a float threshold.

A key optimization is that the candidate set is NOT stored as a [B, V] mask.
The current candidates are exactly the keys whose already-decided high bits match
`prefix`, so membership can be recomputed from `key` and `prefix` every round.
Only two per-row states are carried through the loop:

    krem    [B] : rank that still needs to be found inside the current bucket
    prefix  [B] : high bits of the K-th largest key decided so far.

Outputs:
    t [B]   : K-th largest value (threshold) for each row
    o [B,V] : original values >= threshold, zeros elsewhere
    c [B]   : number of values >= threshold; this may exceed K when ties exist.
              Half the rows of the input are rounded so that it actually does --
              see `make_inputs`.

This is therefore a radix-selection threshold kernel, not a complete
`torch.topk` implementation returning compact values and indices.

Radix select: 32 bit-rounds of count-and-narrow over a monotone integer key. A sort algorithm's decision without its permutation — no value ever moves, so nothing crosses a bank.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Problem shape.
# B: number of independent rows processed together.
# V: number of values in each row.
# K: requested top-k rank in each row.
B, V, K = 128, 1024, 5
assert 1 <= K <= V, f"K must be in [1, V]; got K={K}, V={V}"


def kernel():
    """Build and return the Triton JIT kernel used by KernelSpec."""
    import triton
    import triton.language as tl

    @triton.jit
    def topk_radix_kernel(x_ptr, t_ptr, o_ptr, c_ptr,
                          B: tl.constexpr, V: tl.constexpr, K: tl.constexpr):
        # Construct a logical [B, V] index grid.
        # `b` selects the row and `v` selects an element within that row.
        b = tl.arange(0, B)
        v = tl.arange(0, V)
        idx = b[:, None] * V + v[None, :]
        x = tl.load(x_ptr + idx)

        # —————————————————————————————————————
        # Convert float32 values to monotone integer radix keys.
        # —————————————————————————————————————
        # Radix selection inspects bits directly, but raw IEEE-754 float bits do
        # not have the same integer ordering across negative and positive values.
        #
        # First, reinterpret the float bits as int32. For negative values,
        # flipping all non-sign bits reverses the ordering of their magnitude.
        # The final sign-bit flip makes the resulting key convenient for an MSD
        # search where bit=1 always represents the larger half.
        i = x.to(tl.int32, bitcast=True)
        sign = (tl.zeros((B, V), dtype=tl.int32) + 1) << 31
        key = (i ^ ((i >> 31) & 0x7fffffff)) ^ sign

        # —————————————————————————————————————
        # Per-row radix-selection state.
        # —————————————————————————————————————
        # `krem` is the rank that remains to be found inside the current bucket.
        # `prefix` stores the bits of the K-th largest key that have already been
        # determined. No explicit [B, V] candidate mask is carried across rounds.
        krem = tl.zeros((B,), dtype=tl.float32) + K
        prefix = tl.zeros((B,), dtype=tl.int32)

        # —————————————————————————————————————
        # 1-bit MSD radix selection: bit 31 -> bit 0.
        # —————————————————————————————————————
        # Loop invariant before processing bit p:
        #   - bits 31..p+1 of `prefix` are already equal to those of the K-th key;
        #   - the current candidate set consists exactly of keys matching those
        #     decided high bits.
        #
        # Each round asks: among the current candidates, how many have bit p = 1?
        # If that count is at least `krem`, the desired key lies in that larger
        # half. Otherwise, those `cnt` larger values are skipped and the desired
        # rank becomes `krem - cnt` inside the bit-0 half.
        for p in range(31, -1, -1):
            # Keep bits 31..p and require the result to be exactly bit p.
            # This means:
            #   1) all already-decided higher bits match `prefix`, and
            #   2) the current bit p is 1.
            #
            # We intentionally avoid shifting `key` itself. Triton's int32 right
            # shift is arithmetic, which makes a compact `(key >> p)` formulation
            # problematic at p=31.
            himask = -1 << p
            bitp = 1 << p
            hit = ((key ^ prefix[:, None]) & himask) == bitp

            # Count the bit-1 candidates independently for every row.
            # axis=1 reduces the V dimension, producing one count per row [B].
            cnt = tl.sum(tl.where(hit, 1.0, 0.0), axis=1)

            # If there are at least `krem` candidates in the bit-1 bucket, the
            # K-th key remains inside that bucket. Otherwise it lies in bit-0.
            take = cnt >= krem

            # When choosing bit-0, all `cnt` values in bit-1 are known to be larger
            # and can be skipped, so the rank is reduced by `cnt`.
            krem = tl.where(take, krem, krem - cnt)

            # Record the chosen value of bit p. This updated prefix implicitly
            # defines the candidate set for the next round, replacing an explicit
            # candidate-mask update.
            prefix = prefix | (tl.where(take, 1, 0) << p)

        # —————————————————————————————————————
        # Convert the completed radix key back to a float32 threshold.
        # —————————————————————————————————————
        # After 32 rounds, `prefix` is exactly the K-th largest key. Undo the
        # sign-bit flip and the original negative-value transform, then reinterpret
        # the recovered bits as float32.
        pk = prefix ^ ((tl.zeros((B,), dtype=tl.int32) + 1) << 31)
        tb = pk ^ ((pk >> 31) & 0x7fffffff)
        thr = tb.to(tl.float32, bitcast=True)

        # —————————————————————————————————————
        # Threshold-based output.
        # —————————————————————————————————————
        # Values equal to the threshold are all kept, so ties can make c > K.
        # `o` preserves original positions; it is not a compact top-k array.
        ge = x >= thr[:, None]
        tl.store(t_ptr + b, thr)
        tl.store(o_ptr + idx, tl.where(ge, x, 0.0))
        tl.store(c_ptr + b, tl.sum(tl.where(ge, 1.0, 0.0), axis=1).to(tl.int32))

    return topk_radix_kernel


def make_inputs(torch, seed=0):
    """Create test inputs for correctness checking.

    BOTH distributions live in ONE tensor, because the harness only ever calls
    this with seed=0 and a tie path that needs a second seed is a tie path that
    never runs.

    The top half of the rows is broadly distributed float32: every value in a row
    is distinct, so the K-th largest is unique and c == K. The bottom half is
    rounded to integers, which collapses 1024 values into a few dozen levels and
    makes the threshold a repeated value -- measured 40 of those 64 rows have
    c > K, up to c = 13 for K = 5. That is the path where a threshold kernel
    differs from a topk: the rank-K element is not the only element with its
    value, and the count must report all of them.
    """
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B * V, generator=g, dtype=torch.float32) * 4.0).reshape(B, V)
    x[B // 2:] = torch.round(x[B // 2:])
    return {"x": x.reshape(-1).contiguous()}


def reference(inputs):
    """PyTorch reference for the same threshold/mask/count contract."""
    import torch
    x = inputs["x"].reshape(B, V)

    # torch.topk is used only to obtain the exact K-th largest threshold.
    thr = x.topk(K, dim=1).values[:, -1]
    ge = x >= thr[:, None]

    return {"t": thr.contiguous(),
            "o": torch.where(ge, x, torch.zeros_like(x)).reshape(-1).contiguous(),
            "c": ge.sum(1).to(torch.int32).contiguous()}


# KernelSpec describes how the test harness compiles, launches, and validates
# this Triton kernel: pointer types, compile-time constants, tensor shapes, grid,
# input generator, and PyTorch reference implementation.
SPEC = KernelSpec(
    name="topk_radix_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "t_ptr": "*fp32", "o_ptr": "*fp32",
               "c_ptr": "*i32", "B": "constexpr", "V": "constexpr",
               "K": "constexpr"},
    constexprs={"B": B, "V": V, "K": K},
    args=[Arg("x", IN, "float32", (B * V,)),
          Arg("t", OUT, "float32", (B,)),
          Arg("o", OUT, "float32", (B * V,)),
          Arg("c", OUT, "int32", (B,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-6, atol=1e-6,
)
