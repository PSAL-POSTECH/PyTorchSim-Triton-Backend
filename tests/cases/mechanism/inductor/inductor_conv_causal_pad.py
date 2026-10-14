"""Inductor's conv template on a causal conv1d: the taps before the start are
masked by `idx_x_h >= 0`, and that lower bound used to vanish.

WHAT IT PINS. `MaskAnalysis::parseCmp` turns a comparison into a box end on the
axis -- `value >= c` is `lane >= c - start` -- and it could only do that when
both `c` and the range's start were constants. For a runtime start it folded
`>= 0` away as ALWAYS TRUE, reasoning that such a start is a program id and
cannot be negative. The reasoning was written in the arm it lived in. This
template breaks it: the start is `i - PADDING_H + idx_y_h`, where `idx_y_h` comes
through a program id and a `% OUT_H`, and it really does go negative.

    before   max_abs_err 3.10739 -- the first PADDING_H outputs of every channel
             read BEFORE the channel, silently, with no refusal
    after    max_abs_err 0

HOW IT IS CARRIED NOW. `tts.load`/`tts.store` grew a dynamic `mask_lows` beside
the static one, the way `mask_dims` has always had both, so the first live lane
may be a VALUE; `parseCmp` computes it as `clamp(bound - start, 0, dim)` with no
constant requirement. torchsim-compile's descriptor already had two ends per masked axis --
what it lacked was a runtime one, and `dim_high` is then one add from `dim_low`.

WHY THE TEMPLATE AND NOT SOMETHING SMALLER. Three hand-written shapes did not
reproduce it, all for the same reason: a `tl.arange` start, a constexpr `range`
loop and a program id alone all FOLD, and a folded start takes the constant path
that was always right. What does not fold is the template's `nhw % OUT_W` /
`nh % OUT_H` chain, so the smallest thing that reaches the arm is the template
itself, trimmed to 16 channels.

FOUND BY Qwen3-Next's short convolution: the per-kernel cross-check named `buf4`
(aten.convolution.default) with 23428 of 155648 elements over tol -- exactly the
3 of 19 output positions whose taps reach before the start. The upper end
(`idx_x_h < IN_H`) was always right, because a high bound IS a prefix length,
which is all a single `dim` can say.

torch equivalent: a per-channel conv1d of x with w, zero-padded by K-1 on the
left (see reference).

A causal conv's left padding: pins the mask's lower bound being carried as a value instead of folded away over a runtime start.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, H, K, OH = 16, 16, 4, 19


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def inductor_conv_causal_pad_kernel(arg_X, arg_W, out_ptr0):
        KERNEL_H : tl.constexpr = 4
        KERNEL_W : tl.constexpr = 1
        STRIDE_H : tl.constexpr = 1
        STRIDE_W : tl.constexpr = 1
        PADDING_H : tl.constexpr = 3
        PADDING_W : tl.constexpr = 0
        GROUPS : tl.constexpr = 16
        UNROLL : tl.constexpr = False
        ALLOW_TF32 : tl.constexpr = True
        BLOCK_M : tl.constexpr = 16
        BLOCK_N : tl.constexpr = 16
        BLOCK_K : tl.constexpr = 16
        INDEX_DTYPE : tl.constexpr = tl.int32
        X = arg_X
        W = arg_W

        # Tensor dimensions
        BATCH = 1
        IN_C = 16
        IN_H = 16
        IN_W = 1
        OUT_C = 16
        OUT_H = 19
        OUT_W = 1

        # Strides:
        stride_xn = 256
        stride_xc = 16
        stride_xh = 1
        stride_xw = 1
        stride_wc_out = 4
        stride_wc_in = 4
        stride_wh = 1
        stride_ww = 1

        nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
        idx_y_w = nhw % OUT_W
        nh = nhw // OUT_W
        idx_y_h = nh % OUT_H
        idx_n = nh // OUT_H
        idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


        group = tl.program_id(2).to(INDEX_DTYPE)
        GROUP_IN_C = IN_C // GROUPS
        GROUP_OUT_C = OUT_C // GROUPS


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
        xindex = idx_h + idx_w + 19*idx_c + 155648*idx_n
        tl.store(out_ptr0 + (tl.broadcast_to(idx_h + 19*idx_c, [BLOCK_M, BLOCK_N])), acc, mask)

    return inductor_conv_causal_pad_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"arg_X": torch.randn(C * H, generator=g).to(torch.float32),
            "arg_W": torch.randn(C * K, generator=g).to(torch.float32)}


def reference(inputs):
    import torch
    x = inputs["arg_X"].reshape(C, H)
    w = inputs["arg_W"].reshape(C, K)
    out = torch.zeros(C, OH)
    for h in range(OH):
        for i in range(K):
            j = i - (K - 1) + h
            if 0 <= j < H:
                out[:, h] += x[:, j] * w[:, i]
    return {"out_ptr0": out.reshape(-1)}


SPEC = KernelSpec(
    name="inductor_conv_causal_pad_kernel", kernel=kernel,
    signature={"arg_X": "*fp32", "arg_W": "*fp32", "out_ptr0": "*fp32"},
    constexprs={},
    args=[Arg("arg_X", IN, "float32", (C * H,)),
          Arg("arg_W", IN, "float32", (C * K,)),
          Arg("out_ptr0", OUT, "float32", (C * OH,))],
    grid=(2, 1, C), parallel_axes=("z", "y", "x"),
    reference=reference, make_inputs=make_inputs,
)
