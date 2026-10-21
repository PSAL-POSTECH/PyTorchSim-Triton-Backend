"""`x // K` and `x % K` off `pid*BLOCK + arange` gather ROWS, not elements.

WHAT IT IS. Inductor's 1-D pointwise kernel over a [T, 2K] buffer read in two
halves -- `silu(gu[:, :K]) * gu[:, K:]` -- addresses `x % K + 2K*(x // K)` and
`K + x % K + 2K*(x // K)` with `x = pid*BLOCK + arange(BLOCK)`. The block
offset is runtime and not a multiple of K, so nobody knows where the runs of K
start and `visitOperandDiv` refused the division ("the base offset is not known
here"): both loads became one index per element.

WHAT IS KNOWN INSTEAD. `pid*BLOCK` is a multiple of BLOCK, so a multiple of
g = gcd(BLOCK, K). A run of g values from a multiple of g never crosses a
multiple of K, so on it `x // K` is constant and `x % K` steps by one. The axis
is cut into [BLOCK/g, g]: the inner half affine, the outer half one gather
index per row -- the row starts, built in IR. The mask bound
`min(N - pid*BLOCK, BLOCK)` is a multiple of g as well, so the box survives the
cut.

THE SHAPE IS CHOSEN SO NOTHING ELSE CAN EXPLAIN IT: BLOCK = 1024 is not a
multiple of K = 768, K is not a power of two but has the factor 256, and the
grid has 6 programs, so the block offset really is runtime. g = 256 covers the
lanes, so a row is one index for every lane (ptr_div_into_rows_on_lanes.py is
the shape where it does not).

With the old analysis both loads are 1024-wide gathers with no collapse; with
the row cut each is a [4, 256] gather folded back by one collapse_shape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, K, BLOCK = 8, 768, 1024
N = T * K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_into_rows_kernel(gu_ptr, o_ptr, N: tl.constexpr,
                                 K: tl.constexpr, BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = x < N
        col = x % K
        row = x // K
        a = tl.load(gu_ptr + (col + 2 * K * row), m)
        b = tl.load(gu_ptr + (K + col + 2 * K * row), m)
        tl.store(o_ptr + x, a / (1.0 + tl.exp(-a)) * b, m)

    return ptr_div_into_rows_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"gu": torch.randn(T * 2 * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    gu = inputs["gu"].view(T, 2 * K)
    return {"o": (torch.nn.functional.silu(gu[:, :K]) * gu[:, K:]).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_div_into_rows_kernel", kernel=kernel,
    signature={"gu_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr", "BLOCK": "constexpr"},
    constexprs={"N": N, "K": K, "BLOCK": BLOCK},
    args=[Arg("gu", IN, "float32", (T * 2 * K,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(N // BLOCK,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
    # Two loads, each a gather over BLOCK/g = 4 rows folded back by one
    # collapse_shape; before the row cut this was 0 collapses.
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 2,
                                       "tensor.collapse_shape": 2}},
)
