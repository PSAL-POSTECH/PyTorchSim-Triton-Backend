"""A rank-2 masked LOAD whose mask is LOADED from memory stays a structured box access.

The address is a plain row-major 8x16 tile; the mask `(flag[:, None] > 0) & (m[None, :] > 0)` is two
data-dependent i1s on both axes, which MaskAnalysis cannot box (more than one unstructured axis).
The mask must ride as a full-rank element mask: tts.load stays, no gather or scatter.
Breaks if rewriteLoadOp gives up on the whole access when the mask is not a box.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BN = 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_2d_loaded_mask_load_kernel(x_ptr, f_ptr, m_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        keep = (tl.load(f_ptr + rm) > 0)[:, None] & (tl.load(m_ptr + rn) > 0)[None, :]
        off = rm[:, None] * BN + rn[None, :]
        v = tl.load(x_ptr + off, mask=keep, other=0.0)
        tl.store(o_ptr + off, v)

    return mask_2d_loaded_mask_load_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(BM, dtype=torch.int32); f[::2] = 1
    m = torch.zeros(BN, dtype=torch.int32); m[:11] = 1
    return {"x": torch.randn(BM * BN, generator=g, dtype=torch.float32), "f": f, "m": m}


def reference(inputs):
    import torch
    keep = (inputs["f"] > 0)[:, None] & (inputs["m"] > 0)[None, :]
    return {"o": (inputs["x"].reshape(BM, BN) * keep).reshape(-1)}


SPEC = KernelSpec(
    name="mask_2d_loaded_mask_load_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "m_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (BM * BN,)), Arg("f", IN, "int32", (BM,)),
          Arg("m", IN, "int32", (BN,)), Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.scatter": 0}},
)
