"""Pins a bf16 512^3 dot accumulated in f32, streamed as tile_gemm_streamed_subtiles is: each push
carries rs1 3 so the array reads bfloat16, and llc keeps the bf16 loads as vle16 (+zvfbfmin). Breaks
if a push says 0 (the bits read as halves) or the loads scalarize (binary: vector->scalar extract)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_streamed_bf16(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, b))

    return tile_gemm_streamed_bf16


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    return {"c": inputs["a"].float() @ inputs["b"].float()}


SPEC = KernelSpec(
    name="tile_gemm_streamed_bf16", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-3, atol=1e-3,
    expect_ir={"09-custom.mlir": {"vcix.v.i": 48, "torchsim.wait": 2}},
)
