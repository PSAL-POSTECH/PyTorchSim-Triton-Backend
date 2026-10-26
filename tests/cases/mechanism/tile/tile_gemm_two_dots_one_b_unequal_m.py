"""Pins two bf16 dots on one B, nothing writing it between, of 128 and 64 rows: their streams differ,
so they are not one stream and the second is stamped pytorchsim.kept_apart with why. The values break
if either dot reads the other's rows."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 192, 128, 256
BLOCK_M0, BLOCK_M1 = 128, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_two_dots_one_b_unequal_m(a_ptr, b_ptr, c_ptr, N: tl.constexpr, K: tl.constexpr,
                                           BLOCK_M0: tl.constexpr, BLOCK_M1: tl.constexpr):
        k = tl.arange(0, K)
        n = tl.arange(0, N)
        m0 = tl.arange(0, BLOCK_M0)
        m1 = tl.arange(0, BLOCK_M1)
        a_0 = tl.load(a_ptr + m0[:, None] * K + k[None, :])
        a_1 = tl.load(a_ptr + (BLOCK_M0 + m1)[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        acc = tl.dot(a_0, b, out_dtype=tl.float32)
        tl.store(c_ptr + m0[:, None] * N + n[None, :], acc.to(tl.bfloat16))
        acc = tl.dot(a_1, b, out_dtype=tl.float32)
        tl.store(c_ptr + (BLOCK_M0 + m1)[:, None] * N + n[None, :], acc.to(tl.bfloat16))

    return tile_gemm_two_dots_one_b_unequal_m


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="tile_gemm_two_dots_one_b_unequal_m", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "N": "constexpr",
               "K": "constexpr", "BLOCK_M0": "constexpr", "BLOCK_M1": "constexpr"},
    constexprs={"N": N, "K": K, "BLOCK_M0": BLOCK_M0, "BLOCK_M1": BLOCK_M1},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
