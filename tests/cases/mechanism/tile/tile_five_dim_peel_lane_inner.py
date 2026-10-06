"""Five effective tile dims, lane axis in the inner four: first kernel through
lower_to_dma's >4-D peel, max_abs_err 0 where MLIR rejected the peel's
subview. Breaks it: typing that subview off anything but the source's strides.

THE FIRST KERNEL TO LOWER THROUGH THE PEEL. The MVIN stages memref<2x2x2x2x256xf32, 1> banked on axis 1: five effective dims, one more than a DMA descriptor has, so lower_to_dma peels axis 0 into an affine.for and describes the inner four. Reducing over axis 0 is what keeps fit_layout_to_lanes off axis 0, which the peel refuses ('vlane split axis 0 peeled into the outer loop nest'); N=256 keeps the lane axis worth banking. WHAT IT FAILED WITH BEFORE: the peel typed its memref.subview with the tile's BANKED strides instead of the source memref's, and MLIR rejected the op in stage 5 -- "expected result type to be 'memref<1x2x2x2x256xf32, strided<[2048, 1024, 512, 256, 1]>, 1>' or a rank-reduced version (mismatch of result layout), but got 'memref<2x2x2x256xf32, strided<[2048, 512, 256, 1]>, 1>'". 2048 is the lane axis's banked step written where the source steps 1024, so every peel whose lane axis was not dim 0 died there, and dim 0 is refused: the path had no reachable case at all. Now max_abs_err = 0. WHAT BREAKS IT: typing that subview off anything but the source's own strides, or handing _emit_asm a fixed number of sram indices.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT  # noqa: E402

D, N = 2, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def five_dim_peel_lane_inner_kernel(x_ptr, o_ptr, D: tl.constexpr,
                                        N: tl.constexpr):
        i0 = tl.arange(0, D)
        i1 = tl.arange(0, D)
        i2 = tl.arange(0, D)
        i3 = tl.arange(0, D)
        n = tl.arange(0, N)
        off = (i0[:, None, None, None, None] * (D * D * D * N)
               + i1[None, :, None, None, None] * (D * D * N)
               + i2[None, None, :, None, None] * (D * N)
               + i3[None, None, None, :, None] * N
               + n[None, None, None, None, :])
        x = tl.load(x_ptr + off)
        s = tl.sum(x, axis=0)
        off_o = (i1[:, None, None, None] * (D * D * N)
                 + i2[None, :, None, None] * (D * N)
                 + i3[None, None, :, None] * N
                 + n[None, None, None, :])
        tl.store(o_ptr + off_o, s)

    return five_dim_peel_lane_inner_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(D * D * D * D * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(D, D, D, D, N).sum(dim=0).reshape(-1)}


SPEC = KernelSpec(
    name="five_dim_peel_lane_inner_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "D": "constexpr", "N": "constexpr"},
    constexprs={"D": D, "N": N},
    args=[Arg("x", IN, "float32", (D * D * D * D * N,)),
          Arg("o", OUT, "float32", (D * D * D * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
