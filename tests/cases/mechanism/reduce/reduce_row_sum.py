"""`tl.sum(x, axis=1)` -- reduction over the axis that is IN each lane's vector.

WHY THIS IS THE ONLY REDUCTION SHAPE THAT LOWERS
A tile is banked with dim 0 across the lanes: row r lives in lane r's private
bank. Folding dim 0 would have to move data between lanes and there is no
instruction for that. The innermost axis is what sits inside one lane's vector
register, so folding it is a single `vfredosum.vs` -- one instruction, executed
by every lane at once over its own bank, producing one output element per lane.
The output tile [128] is banked the same way, so it lands where the MVOUT wants it.

HOW IT GETS INTO THAT SHAPE
the linalg lowering normalises every reduction to reduce over dim 0, so this kernel
arrives as

    linalg.transpose ins(128x16) permutation = [1, 0]   -> 16x128
    linalg.reduce    ins(16x128) dimensions = [0]       -> 128

which is the WRONG one: reduced axis on the lanes. absorb_transpose folds the
transpose into the reduce (`reduce(transpose(x, p), d) == reduce(x, p[d])`), the
dimensions become [1], and the 16x128 buffer disappears. No data moves -- the
transpose is pure relabelling once its only consumer is a reduction.

WHAT PROVES IT FIRED
  04-transformed.mlir has no linalg.transpose and no 16x128 global, and
      linalg.reduce ins(%1 : memref<128x16xf32, 1>) ... dimensions = [1]
  06-kernel.s has `vfredosum.vs` and no scalar f-register in the loop.

Before absorb_transpose the transpose ran through convert-linalg-to-loops, which
addressed the banked scratchpad linearly: err 9.36.

In-lane reduction. Covers absorb_transpose (transpose folded into the reduce) and vectorize._bank_reduce. Error is fp32 accumulation order, not a lowering fault.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_row_sum_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        tl.store(o_ptr + rm, tl.sum(x, axis=1))

    return reduce_row_sum_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_row_sum_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
