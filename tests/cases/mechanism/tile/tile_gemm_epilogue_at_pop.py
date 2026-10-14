"""Pins a 512^3 bf16 dot whose f32 result is scaled by a loaded tile, relu'd and cast to bf16: the
epilogue runs on the pops of each N pair's last K pair, inside the stream, not as a loop after it.
Breaks if it runs on a partial sum (values), stays a loop or runs at every K pair (expect_ir)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_epilogue_at_pop(a_ptr, b_ptr, s_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                                  K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        s = tl.load(s_ptr + m[:, None] * N + n[None, :])
        acc = tl.dot(a, b, out_dtype=tl.float32)
        out = tl.maximum(acc * s, 0.0)
        tl.store(c_ptr + m[:, None] * N + n[None, :], out.to(tl.bfloat16))

    return tile_gemm_epilogue_at_pop


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16),
            "s": torch.randn(M, N, generator=g)}


def reference(inputs):
    import torch
    acc = inputs["a"].float() @ inputs["b"].float()
    return {"c": torch.clamp(acc * inputs["s"], min=0.0).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="tile_gemm_epilogue_at_pop", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "s_ptr": "*fp32", "c_ptr": "*bf16",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("s", IN, "float32", (M, N)), Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"09-passes/01-tritonpytorchsim-lower-matmul-tpuv6e.mlir": {"arith.truncf": 12},
               "09-custom.mlir": {"affine.vector_load": 0}},
)
