"""One 64-element row sorted on a [128, 64] tile, the shape Inductor gives a sort whose parallel extent
is 1 (a MoE router's expert ranking): each lane holds its own copy and the network stays inside it.
Breaks if the sort's compare-exchanges cross lanes again (a torchsim.crosslane_move in 08)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64
X = 128        # the lane count, which is what Inductor picks when xnumel is 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sort_single_row_wide_tile_kernel(x_ptr, o_ptr, X: tl.constexpr,
                                         N: tl.constexpr):
        r = tl.arange(0, N)[None, :]
        # ONE row, held X times. This is the shape Inductor emits for xnumel 1:
        # the mask is all-true, the tile is [XBLOCK, R0_BLOCK], and every lane
        # ends up with a copy of the same row.
        v = tl.broadcast_to(tl.load(x_ptr + r), [X, N])
        s = tl.sort(v, dim=1)
        tl.store(o_ptr + tl.broadcast_to(r, [X, N]), s)

    return sort_single_row_wide_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(1, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sort(dim=1).values}


SPEC = KernelSpec(
    name="sort_single_row_wide_tile_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "X": "constexpr", "N": "constexpr"},
    constexprs={"X": X, "N": N},
    args=[Arg("x", IN, "float32", (1, N)), Arg("o", OUT, "float32", (1, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane_move": 0}},
)
