"""Grouped GEMM, CONTIGUOUS RAGGED mode -- G matmuls of different M, one launch.

    for g in range(G):                       # off is DATA, not a constant
        lo, hi = off[g], off[g+1]
        o[lo:hi] = a[lo:hi] @ w[g]

One A matrix of M_TOT rows, G weight matrices, and an offset array that says
where each group's rows start and stop. Group sizes are unequal, one group is
EMPTY, and none of that is known when the kernel is compiled.

WHY THIS IS THE MoE GEMM AND NOT A BATCHED ONE. A batched GEMM has a uniform
group size, so `m_of_group(g)` is `g * M` and every address folds. Here the
number of tokens routed to expert g is whatever the router decided this step,
the sum is fixed by the batch and the split is not, so the only thing the kernel
can be given is the prefix-sum -- and it has to be READ. That is the shape
cuBLAS grouped GEMM, CUTLASS `GroupedProblemVisitor` and DeepGEMM's
`m_grouped_gemm_contiguous` all take, and it is the one an MoE forward pass
produces after `moe_align_block_size.py` and `moe_permute.py` have run.

WHAT THE BACKEND IS BEING ASKED. A per-ROW selection of which WEIGHT MATRIX
applies, resolved from data, with the rows still in one tile. The kernel masks
the A operand rather than the output:

    a_g = where(row in group g, a, 0)  ;  acc += dot(a_g, w[g])

which is exact -- a row outside group g contributes `dot(0, w) == 0`, not a
rounded zero -- and it turns a per-row branch into an accumulation. That is
worth more than the branch it replaces: the accumulator is live across all G
iterations, so triton's own CombineDotAddFPattern (see `ops_dot.py`) folds each
`acc + tl.dot(...)` into the dot's C operand and stage 2 sees G chained
`tt.dot`s whose C is the previous one's result. A grouped GEMM's entire cost
model lives in whether that chain stays in the array.

  The mask comes from two SCALAR loads per group, `off[g]` and `off[g+1]`, so the
  comparison is a runtime value against an affine row index -- the offsets cannot
  be constant-folded into the loop bound, which is exactly the property a real
  grouped GEMM has and a batched one does not.

THE EMPTY GROUP IS DELIBERATE. `off[1] == off[2]`, so group 1's mask is
identically false and its dot contributes nothing. An expert receiving zero
tokens is the common case at low batch size, and a kernel that assumed `hi > lo`
would either read a negative extent or write somebody else's rows.

CONTRAST WITH `grouped_gemm_masked.py` -- same G matmuls, opposite control
shape. There the per-expert extent is a fixed capacity with a runtime mask; here
it is a runtime offset over a fixed TOTAL. Ragged mode is what prefill wants,
because the total token count is known and no capacity is wasted.

WHAT WAS PINNED. M_TOT, the total row count, is a constant 32. That is the extent
a real contiguous grouped GEMM also fixes -- the packed activation buffer is
allocated once at `num_tokens * top_k` -- while every per-group extent stays in
`off`. So nothing that varies at runtime has been made static: the question
"can a group boundary that arrives as data drive which weight tile a row sees"
is asked in full.

G chained tt.dot with a live accumulator and a data-driven row mask on the A operand. The offsets are unequal and one group is empty, so a backend that folded the split would misplace whole rows, not drift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, G = 32, 16, 16, 4      # total rows, contraction, out width, groups


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def grouped_gemm_ragged_kernel(a_ptr, w_ptr, off_ptr, o_ptr,
                                   M: tl.constexpr, K: tl.constexpr,
                                   N: tl.constexpr, G: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])      # [M, K], once
        acc = tl.zeros((M, N), dtype=tl.float32)

        for g in tl.static_range(G):
            # The group's extent is two SCALAR LOADS. Neither can be folded.
            lo = tl.load(off_ptr + g)
            hi = tl.load(off_ptr + g + 1)
            live = (rm >= lo) & (rm < hi)                       # [M]

            # Mask the A OPERAND, not the result: a row outside the group
            # contributes dot(0, w) == 0 exactly, so the accumulate is exact and
            # an empty group is simply an all-false mask.
            ag = tl.where(live[:, None], a, 0.0)
            wg = tl.load(w_ptr + g * (K * N) + rk[:, None] * N + rn[None, :])
            acc += tl.dot(ag, wg)

        tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)

    return grouped_gemm_ragged_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Unequal spans summing to M, with group 1 EMPTY.
    off = torch.tensor([0, 9, 9, 20, M], dtype=torch.int32)
    return {"a": torch.randn(M * K, generator=g, dtype=torch.float32),
            "w": torch.randn(G * K * N, generator=g, dtype=torch.float32),
            "off": off}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(M, K)
    w = inputs["w"].reshape(G, K, N)
    off = inputs["off"].tolist()
    o = torch.zeros(M, N, dtype=torch.float32)
    for g in range(G):
        lo, hi = off[g], off[g + 1]
        if hi > lo:
            o[lo:hi] = a[lo:hi] @ w[g]
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="grouped_gemm_ragged_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "w_ptr": "*fp32", "off_ptr": "*i32",
               "o_ptr": "*fp32", "M": "constexpr", "K": "constexpr",
               "N": "constexpr", "G": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "G": G},
    args=[Arg("a", IN, "float32", (M * K,)),
          Arg("w", IN, "float32", (G * K * N,)),
          Arg("off", IN, "int32", (G + 1,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
