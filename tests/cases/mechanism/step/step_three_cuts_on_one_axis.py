"""GRID: THREE splits of one axis. `64*(p//64) + 8*(p//8) + 2*(p//2)`.

A cut point is a suffix product of a factorisation, and two factorisations meet
at the coarsest one both refine to -- `[4, 64]` against `[128, 2]` meets at
`[4, 32, 2]`, which is what the two-cut sibling shows. THREE is not a new rule:
addState runs twice, so the third factorisation meets the refinement of the
first two, and the answer only exists if it is still a single divisibility
chain. 64, 8 and 2 nest, so it is.

WHY THIS FILE EXISTS SEPARATELY. The refinement was built and measured for two
sides -- "4 accesses see two divisors and all four are {2, 64}" -- and nothing
said whether a third meets them or whether it is where the iteration stops.
Measured here: `sizes: [4, 8, 4, 2]`, one tile, rung 1.

torch equivalent: o = x[64*(p//64) + 8*(p//8) + 2*(p//2)].

Three cuts of one axis meet at their common refinement. The numbers are the same whichever rung it lands on, so expect_ir is what says the refinement iterated rather than stopping after two.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K1, K2, K3 = 256, 64, 8, 2
NX = (K1 * ((N - 1) // K1) + K2 * ((N - 1) // K2)
      + K3 * ((N - 1) // K3) + 1)                            # 695


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_three_cuts_kernel(x_ptr, o_ptr, N: tl.constexpr, K1: tl.constexpr,
                               K2: tl.constexpr, K3: tl.constexpr):
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (K1 * (p // K1) + K2 * (p // K2) + K3 * (p // K3)))
        tl.store(o_ptr + p, v)

    return step_three_cuts_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    return {"o": inputs["x"][K1 * (p // K1) + K2 * (p // K2) + K3 * (p // K3)]}


SPEC = KernelSpec(
    name="step_three_cuts_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K1": "constexpr", "K2": "constexpr", "K3": "constexpr"},
    constexprs={"N": N, "K1": K1, "K2": K2, "K3": K3},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE TILE, FOUR STATE AXES. [4, 8, 4, 2] is the coarsest factorisation all
    # three cut points refine to; a second tptr would mean the iteration stopped.
    expect_ir={"03-linalg.mlir": {"tts.gather": 0, "tts.make_tptr": 2}},
)
