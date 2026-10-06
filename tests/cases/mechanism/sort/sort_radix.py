"""LSD radix sort of a row -- counting, not comparing.

torch equivalent: x.sort(dim=1).values, on non-negative int32 keys

WHY A THIRD SORT
  Bitonic and radix stress opposite halves of the backend. A bitonic network is
  all COMPARE-EXCHANGE: fixed data movement, no arithmetic that depends on the
  values. Radix is all COUNTING: the movement depends on a prefix sum of the
  data, so the destination of every element is computed at run time. If one
  works and the other does not, the difference says whether the gap is in moving
  data or in deriving where to move it.

  It is also the sort that scales. A comparison sort is log-squared stages;
  radix is one pass per digit, and for the small integer keys a router or a
  bucketing kernel actually sorts, that is fewer.

HOW ONE PASS WORKS, WITHOUT A SCATTER
  The stable partition per bit is: everything with the bit clear keeps its
  relative order and moves to the front, everything set follows.

      rank(i) = bit(i) ? (zeros_total + prefix_ones(i)) : prefix_zeros(i)

  Both prefixes are `tl.cumsum` along the row, which this backend already has
  (ops_cumsum.py). What it does NOT have is a scatter into a register tile --
  `out[rank[i]] = x[i]` needs dynamic indexing. So the permutation is applied as
  a COMPARE instead: element j of the output is the one whose rank equals j,
  selected by summing over a rank-equals-position mask.

      out[j] = sum_i x[i] * (rank(i) == j)

  That turns the scatter into a matmul-shaped reduction over an N x N mask,
  which is quadratic and deliberately so -- it is the formulation that exists in
  ops this hardware has. A real radix sort would scatter; this witnesses whether
  the counting half lowers at all, and the mask is how it gets to be expressible.

KEYS ARE NON-NEGATIVE int32 so the bit order is the value order. Signed keys
need the sign bit flipped first, which is a different kernel's problem.

REFUSED at stage 3, on the mask rather than on the counting: vectorize reports `operand [128, 16, 1] broadcasts along dim 0 into a tile [128, 16, 16] ... every lane needs the whole row, which takes a replicating transfer this operand did not get`. So the prefix sums lower and it is the rank-3 broadcast standing in for the scatter that does not. A real scatter would not need the broadcast at all, which is the other half of what this kernel is asking for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, BITS = 128, 16, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sort_radix_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                          BITS: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        cur = tl.load(x_ptr + rm[:, None] * N + rn[None, :])

        for b in tl.static_range(BITS):
            bit = (cur >> b) & 1
            # Exclusive prefix sums along the row, both directions of the bit.
            ones = tl.cumsum(bit, axis=1) - bit
            zeros = tl.cumsum(1 - bit, axis=1) - (1 - bit)
            total_zeros = tl.sum(1 - bit, axis=1)[:, None]
            rank = tl.where(bit == 1, total_zeros + ones, zeros)
            # Apply the permutation without a scatter: out[j] is the element
            # whose rank is j. Exactly one term of each sum is non-zero.
            sel = (rank[:, :, None] == rn[None, None, :])
            cur = tl.sum(tl.where(sel, cur[:, :, None], 0), axis=1)

        tl.store(o_ptr + rm[:, None] * N + rn[None, :], cur)

    return sort_radix_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(0, 1 << BITS, (M, N), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    return {"o": inputs["x"].sort(dim=1).values}


SPEC = KernelSpec(
    name="sort_radix_kernel", kernel=kernel,
    signature={"x_ptr": "*i32", "o_ptr": "*i32", "M": "constexpr",
               "N": "constexpr", "BITS": "constexpr"},
    constexprs={"M": M, "N": N, "BITS": BITS},
    args=[Arg("x", IN, "int32", (M, N)), Arg("o", OUT, "int32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
