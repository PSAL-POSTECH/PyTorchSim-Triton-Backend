"""Cross-entropy loss -- a reduction and a gather over THE SAME tile.

torch equivalent: F.cross_entropy(logits, target, reduction='none')

    lse  = logsumexp(logits, dim=1)
    loss = lse - logits[row, target[row]]

WHY BOTH AT ONCE.  Every gather kernel in the suite gathers and stores; every
reduction kernel reduces and stores. This one does both TO ONE BUFFER, and the
two want incompatible things from it:

  the reduction   needs `logits` as a rank-2 tile banked with rows on the lanes
                  and the reduced axis inside a lane
  the gather      needs ONE element per row, at a column that is only known at
                  run time -- `t_ptr` is loaded, so the offset
                  `rows * N + target` is unstructured in-lane and affine across
                  lanes, i.e. the OPPOSITE of embedding.py

so a single loaded buffer reaches both a `tts.load` of a structured tile and a
`tts.gather`, and the lane axis chosen for one is the lane axis of the other.
This is the smallest kernel where a gather and a reduction share a component.

THE THREE-STEP LOGSUMEXP IS THE POINT OF THE REDUCTIONS. `m = max(x)`,
`sum(exp(x - m))`, `log(...) + m` is two folds on the same axis with an
elementwise chain between them and a scalar-per-row result used twice -- once
inside the second fold's operand, once in the final subtraction. The second use
is what stops the first reduction's result from being dead after the fusion.

THE TARGETS ARE THE INPUT THAT MATTERS: unsorted, repeating, and spread over the
full column range, so nothing about the gather is contiguous.

Complex-op tier: a structured reduction and an in-lane-unstructured gather reading the same buffer, in one component.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 128          # rows == TORCHSIM_COMPILE_VECTORLANE_SIZE, one program
N = 64           # classes, the reduced axis, in-lane


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def cross_entropy_kernel(x_ptr, t_ptr, o_ptr, N: tl.constexpr,
                             M: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, N)

        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        m = tl.max(x, axis=1)
        lse = m + tl.log(tl.sum(tl.exp(x - m[:, None]), axis=1))

        t = tl.load(t_ptr + rows)
        picked = tl.load(x_ptr + rows * N + t)
        tl.store(o_ptr + rows, lse - picked)

    return cross_entropy_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(M * N, generator=g, dtype=torch.float32),
        "t": torch.randint(0, N, (M,), generator=g, dtype=torch.int32),
    }


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, N)
    return {"o": torch.nn.functional.cross_entropy(
        x, inputs["t"].long(), reduction="none")}


SPEC = KernelSpec(
    name="cross_entropy_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "t_ptr": "*i32", "o_ptr": "*fp32",
               "N": "constexpr", "M": "constexpr"},
    constexprs={"N": N, "M": M},
    args=[
        Arg("x", IN, "float32", (M * N,)),
        Arg("t", IN, "int32", (M,)),
        Arg("o", OUT, "float32", (M,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
)
