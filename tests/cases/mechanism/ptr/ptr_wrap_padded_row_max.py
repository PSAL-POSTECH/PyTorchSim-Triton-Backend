"""A row max over 384 columns read at (c + 100) % 384 in a block of 512: the cut load's -inf fill
is the max's identity, so drop-unread-padding cuts the in-lane axis to 384 as for a plain load (2 pieces).
Breaks if drop-unread-padding skips a cut transfer again: the tile stays 512 wide and a fill piece is
added."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, C, CB, O = 128, 384, 512, 100


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_padded_row_max_kernel(x_ptr, y_ptr, M: tl.constexpr, C: tl.constexpr,
                                       CB: tl.constexpr, O: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, CB)[None, :]
        x = tl.load(x_ptr + rows * C + (cols + O) % C, cols < C, other=float("-inf"))
        tl.store(y_ptr + rows, tl.max(x, 1)[:, None])

    return ptr_wrap_padded_row_max_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"].reshape(M, C).amax(dim=1)}


SPEC = KernelSpec(
    name="ptr_wrap_padded_row_max_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "C": "constexpr", "CB": "constexpr", "O": "constexpr"},
    constexprs={"M": M, "C": C, "CB": CB, "O": O},
    args=[Arg("x", IN, "float32", (M * C,)), Arg("y", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
    expect_ir={"08-transformed.mlir": {"memref<128x512xf32": 0, "memref<128x384xf32": 4},
               "09-custom.mlir": {"pytorchsim.transfer": 3}},
)
