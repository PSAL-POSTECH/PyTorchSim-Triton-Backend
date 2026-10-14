"""Grouped GEMM, MASKED mode -- fixed capacity per expert, the count is data.

    for g in range(G):                       # cnt is DATA; C is a constant
        rows = arange(C) < cnt[g]
        o[g] = where(rows, x[g], 0) @ w[g]

Every expert owns the same C-row slot of the activation buffer whether it needs
it or not. How many of those rows are real arrives as `cnt`, a vector read at
runtime; the rows past it hold whatever the last step left there.

WHY A RUNTIME SHAPE GETS REPLACED BY A RUNTIME MASK. Ragged mode
(`grouped_gemm_ragged.py`) needs the group offsets BEFORE the launch, because
the offsets are what size the launch. In expert-parallel decoding they are not
available in time: each rank has to send its tokens to the ranks that own the
chosen experts, and how many arrive is only known once the all-to-all lands. If
the GEMM's shape depended on that, the CPU would have to wait for the transfer,
read a count back, and only then launch -- a device-to-host sync in the middle
of every decode step, which is precisely what CUDA-graph capture forbids. So
DeepGEMM's `m_grouped_gemm_masked` fixes the M extent at a capacity and takes
the counts as a TENSOR. The shape is now constant, the graph replays, and the
count moved from the launch configuration into the data.

  The price is arithmetic on rows that do not exist. That is accepted because
  decode's per-expert token count is small and irregular, and a wasted tile is
  cheaper than a sync.

CONTRAST WITH `grouped_gemm_ragged.py` -- SAME G matmuls, same weights, same
contraction; only the control shape differs.

    ragged   total M fixed, per-group M from an offset array, groups packed
             back to back, no wasted rows, needs the counts before launch
    masked   per-group M fixed at C, occupancy from a count vector, G*C rows
             allocated regardless, needs nothing before launch

  And the masks differ in kind, not just in source. Ragged's mask separates two
  groups that share one tile, so it selects BETWEEN weights. Masked's mask
  separates live rows from garbage inside ONE group, so it selects between data
  and nothing. Nothing accumulates across groups here -- each expert writes its
  own disjoint output slice, which is why there is no live accumulator and no
  chained `tt.dot`.

THE MASK IS ON THE OPERAND, AND THAT IS WHAT THE TEST CHECKS. `make_inputs` fills
the rows past `cnt[g]` with 1e30, i.e. real garbage, and the kernel zeroes them
BEFORE the dot rather than zeroing the result after. Both give the same answer
for the live rows -- a GEMM's rows are independent -- so the only way to tell the
two apart is the dead rows, and here they come out EXACTLY zero because the
operand was zero. Drop the `where` and they come out 1e30-scale instead of zero,
which the comparison catches. Masking the result would have hidden that.

  `cnt[1] == 0` (an expert nobody routed to) and `cnt[2] == C` (an expert at
  full capacity) are both present: the two ends where an off-by-one in the row
  compare stops being visible.

WHAT WAS PINNED. Nothing that a real masked GEMM does not also pin. C is the
capacity, a constant by construction -- that IS the mode. G is the expert count,
static in any deployment. The counts, the only genuinely runtime quantity, are
still a tensor.

The dead rows are exactly zero only if the mask reached the OPERAND; the garbage is 1e30, so a mask applied to the result instead shows up as an enormous error and not a drift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

G, C, K, N = 4, 16, 16, 16      # experts, per-expert capacity, contraction, width


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def grouped_gemm_masked_kernel(x_ptr, w_ptr, cnt_ptr, o_ptr,
                                   G: tl.constexpr, C: tl.constexpr,
                                   K: tl.constexpr, N: tl.constexpr):
        rc = tl.arange(0, C)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        for g in tl.static_range(G):
            n = tl.load(cnt_ptr + g)              # how many of C rows are real
            live = rc < n                         # [C], runtime, not a bound

            a = tl.load(x_ptr + g * (C * K) + rc[:, None] * K + rk[None, :])
            # Zero the OPERAND. The garbage past cnt then cannot reach the
            # array, and the dead output rows are exactly 0 rather than nearly.
            a = tl.where(live[:, None], a, 0.0)

            wg = tl.load(w_ptr + g * (K * N) + rk[:, None] * N + rn[None, :])
            # Disjoint output slice per expert: no accumulator crosses groups.
            tl.store(o_ptr + g * (C * N) + rc[:, None] * N + rn[None, :],
                     tl.dot(a, wg))

    return grouped_gemm_masked_kernel


def make_inputs(torch, seed=0):
    gen = torch.Generator().manual_seed(seed)
    cnt = torch.tensor([11, 0, C, 5], dtype=torch.int32)   # empty and full both
    x = torch.randn(G, C, K, generator=gen, dtype=torch.float32)
    # Past the count the buffer holds whatever the previous step left. Make it
    # loud, so a missing operand mask cannot be mistaken for rounding.
    row = torch.arange(C)[None, :, None]
    x = torch.where(row < cnt.long()[:, None, None], x,
                    torch.full_like(x, 1e30))
    return {"x": x.reshape(-1).contiguous(),
            "w": torch.randn(G * K * N, generator=gen, dtype=torch.float32),
            "cnt": cnt}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(G, C, K)
    w = inputs["w"].reshape(G, K, N)
    cnt = inputs["cnt"].long()
    row = torch.arange(C)[None, :, None]
    xm = torch.where(row < cnt[:, None, None], x, torch.zeros_like(x))
    return {"o": torch.bmm(xm, w).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="grouped_gemm_masked_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "cnt_ptr": "*i32",
               "o_ptr": "*fp32", "G": "constexpr", "C": "constexpr",
               "K": "constexpr", "N": "constexpr"},
    constexprs={"G": G, "C": C, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (G * C * K,)),
          Arg("w", IN, "float32", (G * K * N,)),
          Arg("cnt", IN, "int32", (G,)),
          Arg("o", OUT, "float32", (G * C * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
