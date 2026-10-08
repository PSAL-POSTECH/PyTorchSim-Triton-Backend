"""`tl.max(x)` over a 2-D tile -- ONE `linalg.reduce` over BOTH dimensions.

WHAT IT PINS. vectorize can bank one axis, and it says so rather than
guessing when handed two:

    reduce over [0, 1]: only one axis at a time is expressible

That refusal is live code with a live message, and until this file nothing in
the suite reached it. `tl.sum` does not: Triton lowers the SUM of a 2-D tile as
two folds (inner axis, then outer), so the linalg lowering hands down two one-
dimensional `linalg.reduce`s -- which is why reduce_captured_scalar.py folds an
8x16 tile to rank 0 and never goes near this. `tl.max` arrives as one reduce
over `dimensions = [0, 1]`, and this kernel is where that shape is measured.

HOW IT WAS FOUND, AND WHAT THE HONEST STATUS IS. reduce_scalar_exp.py was
written 2-D first and died here, so this went in as `expect="blocked"`. It is
NOT blocked on the tree as it stands: the reduce reaches vectorize already
split, and the kernel compiles and returns exact values. The failure was
reproducible only under an abandoned rewrite that changed what fuse
saw -- which is the point of keeping the file rather than deleting it. Whether
this reduce arrives as one op or two is decided several passes above by fusion,
nothing states it, and the pass below refuses one of the two answers. This is
the kernel that notices when that changes.

WHAT WOULD BREAK IT. Any change to fusion or absorption that leaves the
two-dimensional reduce standing at vectorize. The fix then is to SPLIT it
-- a reduce over two dimensions is the composition of two one-dimensional folds,
and splitting is a rewrite on `dimensions`, read off the op rather than pattern
matched (rule 5) -- and it belongs above the pass that has to choose one axis to
bank.

A whole-tile max, which is one two-dimensional reduce rather than the two one-dimensional ones tl.sum arrives as. Reaches vectorize's two-axis refusal without tripping it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_rank0_two_axis_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                     N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        idx = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + idx)
        m = tl.max(x)                  # ONE reduce over dimensions [0, 1]
        tl.store(o_ptr + idx, x - m)

    return reduce_rank0_two_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x - x.max()}


SPEC = KernelSpec(
    name="reduce_rank0_two_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
