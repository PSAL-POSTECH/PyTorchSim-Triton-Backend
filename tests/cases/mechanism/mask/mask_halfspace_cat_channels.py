"""A cat whose channel is `3 * x + r // 256` reads each half as a box per row, never before its tensor.

The tile is [32, 1024] with r = 256 * d + e, d < 3 cut by `r < 768`; channel c = 3x + d couples two axes,
so `c >= 64` is not a box but a suffix of the flat (x, d) index: rows x >= 22 plus x = 21 with d >= 1.
The second half reads b at (c - 64) * 256 + e, 16384 elements before b where the mask is off. Breaks if the
suffix goes back to an element mask: MaskedBoxBeforeTensor, or a read before b.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R, K = 32, 1024, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_halfspace_cat_channels_kernel(a_ptr, b_ptr, o_ptr, xnumel,
                                           X: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
        x = tl.program_id(0) * X + tl.arange(0, X)[:, None]
        r = tl.arange(0, R)[None, :]
        d = r // K
        e = r % K
        c = 3 * x + d
        live = (r < 3 * K) & (x < xnumel)
        va = tl.load(a_ptr + c * K + e, mask=live & (c < 64), other=0.0)
        vb = tl.load(b_ptr + (c - 64) * K + e, mask=live & (c >= 64), other=0.0)
        tl.store(o_ptr + x * 3 * K + r, tl.where(c < 64, va, vb), mask=live)

    return mask_halfspace_cat_channels_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(64 * K, generator=g, dtype=torch.float32),
            "b": torch.randn(32 * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.cat([inputs["a"], inputs["b"]])}


SPEC = KernelSpec(
    name="mask_halfspace_cat_channels_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32", "xnumel": "i32",
               "X": "constexpr", "R": "constexpr", "K": "constexpr"},
    constexprs={"X": X, "R": R, "K": K},
    args=[Arg("a", IN, "float32", (64 * K,)), Arg("b", IN, "float32", (32 * K,)),
          Arg("o", OUT, "float32", (96 * K,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("xnumel", "int32_t")], "scalar_values": {"xnumel": X}},
    expect_ir={"07-regularized.mlir": {"tts.scatter": 0, "tts.gather": 0}},
)
