"""A lane order that comes from MEMORY -- the pattern nothing can compute.

WHAT THIS PINS. `torchsim.crosslane = "permute"` is a lane-to-lane mapping, and
every one the suite mints today is AFFINE: `rounds_of_pairs` derives `mul` and
`add` from the correspondence and `_xlu_permute` rebuilds the pattern with two
arithmetic ops on the lane number. That is a property of the COMPILER, not of the
unit -- `rpu`'s arbitrary mode reads one lane number per lane as a plain integer
and never asks where it came from.

Here the order is a value: `idx` is loaded, so no arithmetic on the lane number
can produce it. The tile it reorders is computed in the scratchpad (`tl.exp`),
so it is already banked across the lanes when the reorder is asked for.

    today    the value is stored and gathered back -- a DRAM round trip per tile,
             and the numbers are right, so the verdict is `pass` and what records
             the gap is `expect_ir` below: NO crossbar move at all
    wanted   one pass of the unit with the loaded vector as the pattern, which is
             the day this kernel fails and whoever did it raises the count

WHY IT IS SPELLED AS A STORE AND A GATHER. Triton has no way to index a TILE by a
tensor -- `y[idx]` is not in the language -- so a lane permutation can only be
written as a memory gather. Recognising that the gathered buffer is one this
kernel just wrote, and that the DMA is therefore carrying what the unit could,
is the work this kernel asks for.

torch equivalent: exp(x)[idx]

A lane order loaded from memory: pins the permute whose pattern is a VALUE, which no arithmetic on the lane number can make.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, M = 16, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_permute_by_index_kernel(x_ptr, p_ptr, t_ptr, o_ptr,
                                        N: tl.constexpr, M: tl.constexpr):
        i = tl.arange(0, N)
        j = tl.arange(0, M)
        flat = i[:, None] * M + j[None, :]
        y = tl.exp(tl.load(x_ptr + flat))
        tl.store(t_ptr + flat, y)
        idx = tl.load(p_ptr + i)
        tl.store(o_ptr + flat, tl.load(t_ptr + idx[:, None] * M + j[None, :]))

    return ops_xlu_permute_by_index_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N * M, generator=g, dtype=torch.float32),
            "p": torch.randperm(N, generator=g).to(torch.int32),
            "t": torch.zeros(N * M, dtype=torch.float32)}


def reference(inputs):
    y = inputs["x"].reshape(N, M).exp()
    return {"t": y.reshape(-1), "o": y[inputs["p"].long()]}


SPEC = KernelSpec(
    name="ops_xlu_permute_by_index_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "p_ptr": "*i32", "t_ptr": "*fp32",
               "o_ptr": "*fp32", "N": "constexpr", "M": "constexpr"},
    constexprs={"N": N, "M": M},
    args=[Arg("x", IN, "float32", (N * M,)), Arg("p", IN, "int32", (N,)),
          Arg("t", OUT, "float32", (N * M,)),
          Arg("o", OUT, "float32", (N, M))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane_move": 0}},
)
