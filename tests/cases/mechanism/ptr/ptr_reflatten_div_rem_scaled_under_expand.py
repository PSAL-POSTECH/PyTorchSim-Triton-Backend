"""`(x%K*S)[:, None] + (x//K*(K*S))[:, None]`: the scaled pair meets under a shape op, as a conv template's row does.

The scale is built at the 1D index's type and the expand_dims is cloned over the product, not at the 2D sum's type.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, S, BM, BC = 4, 16, 64, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_reflatten_div_rem_scaled_under_expand_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                                         S: tl.constexpr, BM: tl.constexpr,
                                                         BC: tl.constexpr):
        m = tl.arange(0, BM)
        c = tl.arange(0, BC)
        row = ((m % K) * S)[:, None] + ((m // K) * (K * S))[:, None]     # == (m*S)[:, None]
        v = tl.load(x_ptr + row + c[None, :])
        tl.store(o_ptr + m[:, None] * BC + c[None, :], v * 2.0)

    return ptr_reflatten_div_rem_scaled_under_expand_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn((BM - 1) * S + BC, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    idx = torch.arange(BM)[:, None] * S + torch.arange(BC)[None, :]
    return {"o": (inputs["x"][idx] * 2.0).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_reflatten_div_rem_scaled_under_expand_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr", "S": "constexpr",
               "BM": "constexpr", "BC": "constexpr"},
    constexprs={"K": K, "S": S, "BM": BM, "BC": BC},
    args=[Arg("x", IN, "float32", ((BM - 1) * S + BC,)), Arg("o", OUT, "float32", (BM * BC,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.load": 1}},
)
