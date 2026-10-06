"""An UNBALANCED clash: three reductions fold axis 1, one folds axis 0.

    c = tl.dot(a, b)
    tl.store(..., tl.sum(c, axis=1))     three demands the array's axis refuses
    tl.store(..., tl.max(c, axis=1))
    tl.store(..., tl.min(c, axis=1))
    tl.store(..., tl.sum(c, axis=0))     one it does not

WHAT THIS PINS, AND WHY THE TWO KERNELS BESIDE IT DO NOT
mixed_reduce_both_axes folds 1 and 0; mixed_reduce_four_ways folds 1, 0, 1, 0.
Both are BALANCED, and a whole-group turn on a balanced clash swaps which side
is unhappy and leaves the count exactly where it was. fit_layout_to_lanes used to
read that as a property of clashes in general -- `_turn_wanted` collected two
reductions folding opposite axes into `clash` and refused the group outright,
with the reason written down as "a whole-group turn swaps which one is unhappy".

That is true of a balance and false of everything else. Here the turn moves
three demands off the contested axis and one onto it, so the group goes from
three losers to one: a strict improvement that the a-priori refusal could not
see, because the thing that distinguishes the two shapes is the count AFTER the
turn rather than the fact that a clash exists.

WHAT BREAKS IT
Restoring the `clash` set in `_turn_wanted`, or any gate that decides a turn
before making one. With the refusal in place this kernel does not turn; it takes
the DRAM cut instead and pays three round trips where one turn plus one cut is
enough. It still ANSWERS correctly -- the cut is always available -- so the
signal is the round-trip count in `regress.py`, not the verdict:

    with the measured turn      this many trips or fewer
    with `clash` refusing       three, one per reduction that lost

STATUS. `expect="pass"`: the numbers are right either way, and what this kernel
is really holding is the cost. Read it together with the trip count on its
`regress.py` line, which is the number rule 8b says is the score.

Three reductions lose to the array and one does not, so a turn is a strict improvement rather than a relabelling -- the shape an a-priori `clash` refusal cannot tell from a balanced one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# K = 2 * BK so the accumulator stays loop-carried, as in the two kernels beside
# this one -- lower_to_vcix needs the loop to mark as accumulation_loop.
M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_reduce_three_to_one_kernel(
                    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr,
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

        # THREE on axis 1 against ONE on axis 0. That imbalance is the whole
        # kernel: it is what makes turning the group a strict improvement
        # instead of a relabelling.
        tl.store(c_ptr + offs_m, tl.sum(acc, axis=1))
        tl.store(d_ptr + offs_m, tl.max(acc, axis=1))
        tl.store(e_ptr + offs_m, tl.min(acc, axis=1))
        tl.store(f_ptr + offs_n, tl.sum(acc, axis=0))

    return mixed_reduce_three_to_one_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    p = torch.matmul(inputs["a"], inputs["b"])
    return {"c": p.sum(dim=1), "d": p.amax(dim=1), "e": p.amin(dim=1),
            "f": p.sum(dim=0)}


SPEC = KernelSpec(
    name="mixed_reduce_three_to_one_kernel",
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
        Arg("d", OUT, "float32", (M,)),
        Arg("e", OUT, "float32", (M,)),
        Arg("f", OUT, "float32", (N,)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # Every output is a reduction over the matmul's own reassociated sum, so
    # even max and min inherit its error rather than being exact.
    rtol=1e-3, atol=1e-3,
)
