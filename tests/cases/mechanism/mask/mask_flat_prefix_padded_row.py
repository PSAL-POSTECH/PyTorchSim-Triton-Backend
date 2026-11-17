"""A mask that is a prefix of a tile's own flat index is a box per digit, not an element mask.

`pos = reg * 128 + lane` over a [128, 4] tile is a contiguous 512-element run and `pos < E` keeps its
first E elements (E runtime): full registers below E // 128 plus one partial register. The padded row
(stride 640) sits at the end of x/o so an unmasked read runs past the tensor and an unmasked write
into the pad shows as a changed sentinel. Breaks if the prefix goes back to an element mask: the store
is refused as MaskedStoreNotABox, or the read leaves the tensor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

L, D, STRIDE, E, ROWS = 128, 4, 640, 300, 2
N = (ROWS - 1) * STRIDE + E


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_flat_prefix_padded_row_kernel(x_ptr, o_ptr, E, STRIDE: tl.constexpr,
                                           L: tl.constexpr, D: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, L)[:, None]
        reg = tl.arange(0, D)[None, :]
        pos = reg * L + lane
        live = pos < E
        v = tl.load(x_ptr + row * STRIDE + pos, mask=live, other=0.0)
        tl.store(o_ptr + row * STRIDE + pos, v * 2.0, mask=live)

    return mask_flat_prefix_padded_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "o": torch.full((N,), 7.0, dtype=torch.float32)}


def reference(inputs):
    import torch
    o = inputs["o"].clone()
    for r in range(ROWS):
        o[r * STRIDE:r * STRIDE + E] = inputs["x"][r * STRIDE:r * STRIDE + E] * 2.0
    return {"o": o}


SPEC = KernelSpec(
    name="mask_flat_prefix_padded_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "E": "i32",
               "STRIDE": "constexpr", "L": "constexpr", "D": "constexpr"},
    constexprs={"STRIDE": STRIDE, "L": L, "D": D},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", INOUT, "float32", (N,))],
    grid=(ROWS,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("E", "int32_t")], "scalar_values": {"E": E}},
    expect_ir={"07-regularized.mlir": {"tts.scatter": 0, "tts.gather": 0}},
)
