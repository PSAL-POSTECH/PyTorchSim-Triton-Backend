"""Twenty absorbable transposes in one function, and a pass that stopped at sixteen.

WRONG NUMBERS ON THE GRAPH ROUTE (2026-09-22), max_abs_err 3.93 on one output
and 5.72e-06 on the other. NOT A REGRESSION AND NOT NEW: before the mint counter
was made one per pass this file died earlier, at "redefinition of symbol named
'graphtrip90082_dram'", so the wrong answer was there and unreachable. Twenty
transposes come out as twenty rewrites -- some an in-bank `linalg.generic`, some
a scattering round trip -- and which one is wrong has not been bisected.

WHAT IT PINS. absorb_layout_ops's outer loop. Absorbing one layout op can expose the
next -- a transpose feeding a transpose collapses only once the inner one is
gone -- so the pass sweeps the function repeatedly, and each sweep takes the
FIRST op it can absorb and starts over. The number of sweeps has to be the
number of ops, and it was the constant 16.

A `tl.static_range` unrolls into as many transposes as it has iterations. This
file has 20, every one of them a plain fold: two readers, an MVOUT taking the
permutation into its `dram_stride` and a `linalg.reduce` taking it into its
`dimensions`, neither of which moves an element. Sixteen went, four stayed, and
vectorize is where that was discovered:

    'linalg.transpose' still touches a scratchpad buffer after banking. Nothing
    lowers it from here except convert-linalg-to-loops, which reads a
    lane-banked buffer linearly

WHY IT IS THE PASS'S BUG AND NOT A BUDGET. Rule 8: a layout op below
absorb_layout_ops is a demand the hardware cannot answer, and finding one there is a
bug rather than a fallback. A cap on the loop that does that job turns "I could
not" into "I stopped", and nothing said which -- the four leftovers looked
exactly like the ones this pass genuinely refuses. Found on dl/ssd_chunk, whose
20 come out of a static_range over four chunks and which was hitting the cap
exactly: 20 in, 4 out, 16 absorbed.

WHAT WOULD BREAK IT. Put any constant back on that loop below 20. The count
here is deliberately just over it, and a file that pins a loop bound has to sit
on the far side of the bound to pin anything.

torch equivalent: x.transpose(1, 2), stored, and summed along its last axis.

How many layout ops a function has is not a constant, so how many sweeps it takes to absorb them is not one either.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, NI = 8, 8, 20


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_trans_many_unrolled_kernel(x_ptr, s_ptr, o_ptr, M: tl.constexpr,
                                       N: tl.constexpr, NI: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        acc = tl.zeros((N,), dtype=tl.float32)
        for i in tl.static_range(NI):
            t = tl.load(x_ptr + i * M * N + rm[:, None] * N + rn[None, :])
            tt = tl.trans(t, 1, 0)                 # one per unrolled iteration
            tl.store(s_ptr + i * M * N + rn[:, None] * M + rm[None, :], tt)
            acc += tl.sum(tt, axis=1)
        tl.store(o_ptr + rn, acc)

    return ops_trans_many_unrolled_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NI * M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(NI, M, N)
    tt = x.transpose(1, 2).contiguous()
    return {"s": tt.reshape(-1), "o": tt.sum(dim=(0, 2))}


SPEC = KernelSpec(
    name="ops_trans_many_unrolled_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "NI": "constexpr"},
    constexprs={"M": M, "N": N, "NI": NI},
    args=[Arg("x", IN, "float32", (NI * M * N,)),
          Arg("s", OUT, "float32", (NI * M * N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
