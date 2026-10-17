"""Inductor's conv template on a 3-channel channels-last image: the row mask of
the X load reaches the access through layout conversions, and the gather it
rides has its index in the layout before them.

WHAT IT PINS. Two places where a mask meets something in another layout:

  MaskAnalysis   the masks of a state parsed from a value are in that value's
                 layout -- parseCmp collapses in its own, parseAnd rebuilds in
                 the and's. `parse` looked through ttg.convert_layout without
                 moving them, so `(idx_n < BATCH) & (idx_x_h >= 0) ...` built an
                 arith.andi of a slice of blocked order [1, 0] with a slice of
                 order [0, 1]: "'arith.andi' op requires the same encoding for
                 all operands and results" in TritonToStructured.
  PtrAnalysis    a conversion of a gather pointer passes the same op through,
                 so the gather index stays in the layout it was computed in,
                 while the row mask is in the access's. rebuildGatherWithMask
                 paired the two as they were, and the select that parks masked
                 rows mixed memref<1x64xi8> with memref<64x1xi32>:
                 NotVectorisable in torchsim-compile:transform.

WHY THE TEMPLATE AND THESE SIZES. The conversions are left in the mask chain
since remove-layout-conversions stopped recomputing across encodings of one
linear layout (5ad49a1), and the row index is a gather only when OUT_W does
not divide BLOCK_M (`nhw % OUT_W` does not fold): OUT_W = 12 against 64. The
three input channels (stride_xw = 3) keep K padded past its mask, as in the
first conv of resnet / vit / clip / mobilenet, where it was found.

torch equivalent: conv2d of an NHWC image with OHWI weights, stride 2,
padding 1, output written NHWC (see reference).

A row mask behind layout conversions on a row gather: pins masks moving with a conversion and joining the gather index's layout.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C_IN, HW, C_OUT, K, OHW = 3, 24, 32, 3, 12


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def inductor_conv_row_mask_behind_convert_kernel(arg_X, arg_W, out_ptr1):
        KERNEL_H : tl.constexpr = 3
        KERNEL_W : tl.constexpr = 3
        STRIDE_H : tl.constexpr = 2
        STRIDE_W : tl.constexpr = 2
        PADDING_H : tl.constexpr = 1
        PADDING_W : tl.constexpr = 1
        GROUPS : tl.constexpr = 1
        UNROLL : tl.constexpr = False
        ALLOW_TF32 : tl.constexpr = False
        BLOCK_M : tl.constexpr = 64
        BLOCK_N : tl.constexpr = 32
        BLOCK_K : tl.constexpr = 16
        INDEX_DTYPE : tl.constexpr = tl.int32
        X = arg_X
        W = arg_W

        # Tensor dimensions
        BATCH = 1
        IN_C = 3
        IN_H = 24
        IN_W = 24
        OUT_C = 32
        OUT_H = 12
        OUT_W = 12

        # Strides:
        stride_xn = 1728
        stride_xc = 1
        stride_xh = 72
        stride_xw = 3
        stride_wc_out = 27
        stride_wc_in = 1
        stride_wh = 9
        stride_ww = 3

        nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
        idx_y_w = nhw % OUT_W
        nh = nhw // OUT_W
        idx_y_h = nh % OUT_H
        idx_n = nh // OUT_H
        idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


        group = 0
        GROUP_IN_C = IN_C
        GROUP_OUT_C = OUT_C


        x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
        w_base = (
            W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


        # Could be simplified, but slightly slower:
        # for i in range(KERNEL_H):
        #     for j in range(KERNEL_W):
        #         for k in range(0, GROUP_IN_C, BLOCK_K):
        BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
        for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
            k = (ijk % BLOCK_K_COUNT) * BLOCK_K
            ij = ijk // BLOCK_K_COUNT
            i = ij // KERNEL_W
            j = ij % KERNEL_W

            idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
            idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
            idx_x_c = tl.arange(0, BLOCK_K) + k

            x_ptrs = x_base + (
                (idx_x_h * stride_xh)[:, None]
                + (idx_x_w * stride_xw)[:, None]
                + (idx_x_c * stride_xc)[None, :]
            )
            mask_x = (
                (idx_n < BATCH)[:, None]
                & (idx_x_h >= 0)[:, None]
                & (idx_x_h < IN_H)[:, None]
                & (idx_x_w >= 0)[:, None]
                & (idx_x_w < IN_W)[:, None]
                & (idx_x_c < GROUP_IN_C)[None, :]
            )
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

            w_ptrs = w_base + (
                (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
            )
            mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
            matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
            acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



        mask = (
            (idx_n < BATCH)[:, None]
            & (idx_y_h < OUT_H)[:, None]
            & (idx_y_w < OUT_W)[:, None]
            & (idx_y_c < GROUP_OUT_C)[None, :]
        )
        idx_n = idx_n[:, None]
        idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
        idx_h = idx_y_h[:, None]
        idx_w = idx_y_w[:, None]

        # inductor generates a suffix
        xindex = idx_w + 12*idx_h + 144*idx_c + 4608*idx_n
        x5 = (xindex % 144)
        tl.store(out_ptr1 + (idx_c + 32*x5), acc, mask)

    return inductor_conv_row_mask_behind_convert_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"arg_X": torch.randn(HW * HW * C_IN, generator=g).to(torch.float32),
            "arg_W": torch.randn(C_OUT * K * K * C_IN, generator=g).to(torch.float32)}


def reference(inputs):
    import torch
    x = inputs["arg_X"].reshape(1, HW, HW, C_IN).permute(0, 3, 1, 2)
    w = inputs["arg_W"].reshape(C_OUT, K, K, C_IN).permute(0, 3, 1, 2)
    y = torch.nn.functional.conv2d(x, w, stride=2, padding=1)
    return {"out_ptr1": y.permute(0, 2, 3, 1).reshape(-1)}


SPEC = KernelSpec(
    name="inductor_conv_row_mask_behind_convert_kernel", kernel=kernel,
    signature={"arg_X": "*fp32", "arg_W": "*fp32", "out_ptr1": "*fp32"},
    constexprs={},
    args=[Arg("arg_X", IN, "float32", (HW * HW * C_IN,)),
          Arg("arg_W", IN, "float32", (C_OUT * K * K * C_IN,)),
          Arg("out_ptr1", OUT, "float32", (OHW * OHW * C_OUT,))],
    grid=(3, 1, 1), parallel_axes=("z", "y", "x"),
    reference=reference, make_inputs=make_inputs,
)
