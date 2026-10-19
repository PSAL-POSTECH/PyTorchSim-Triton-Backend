"""A softmax of 8 rows of 384 in a 512 block, in Inductor's mask form: the max and the sum reduce
`where(c < 384, x, identity)`. fit-to-numel cuts the columns to 384, where the mask is always true, so
each select is its value and its generic only copies its input: the reduces read that input (the
loaded tile, the exp) and no buffer holds a copy of it. Breaks if a select fit-to-numel folded is
walked again as a copy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, B = 8, 384, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_masked_select_folded_reads_its_input_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
        c = tl.arange(0, B)[None, :]
        mask = c < C
        at = tl.arange(0, R)[:, None] * C + c
        x = tl.load(x_ptr + at, mask, other=0.0)
        m = tl.max(tl.where(mask, x, float("-inf")), axis=1)[:, None]
        s = tl.sum(tl.where(mask, tl.exp(x - m), 0.0), axis=1)[:, None]
        y = tl.load(x_ptr + at, mask, other=0.0)
        tl.store(o_ptr + at, tl.exp(y - m) / s, mask)

    return reduce_masked_select_folded_reads_its_input_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": torch.softmax(x, 1)}


SPEC = KernelSpec(
    name="reduce_masked_select_folded_reads_its_input_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr", "B": "constexpr"},
    constexprs={"R": R, "C": C, "B": B},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    # THE READ IS THE CLAIM: six stores; the two copies of the tile and the exp were two more.
    expect_ir={"08-transformed.mlir": {"memref.global": 7, "affine.vector_store": 6}},
)
