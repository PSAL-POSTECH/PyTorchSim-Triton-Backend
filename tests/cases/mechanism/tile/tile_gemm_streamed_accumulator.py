"""Pins a loaded accumulator (tl.dot(a, b, acc)) in a 512^3 f32 dot on the matrix unit: C is read
and added into under every pop of the one stream, none written as if C were zero. Breaks if a pop
skips C (arith.addf below 18, one per vcix.v.i), a wait enters the stream, or values drift."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_streamed_accumulator(a_ptr, b_ptr, c_ptr, y_ptr, M: tl.constexpr,
                                            N: tl.constexpr, K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        acc = tl.load(c_ptr + m[:, None] * N + n[None, :])
        tl.store(y_ptr + m[:, None] * N + n[None, :], tl.dot(a, b, acc))

    return tile_gemm_streamed_accumulator


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g), "b": torch.randn(K, N, generator=g),
            "c": torch.randn(M, N, generator=g)}


def reference(inputs):
    return {"y": inputs["c"] + inputs["a"] @ inputs["b"]}


SPEC = KernelSpec(
    name="tile_gemm_streamed_accumulator", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", IN, "float32", (M, N)), Arg("y", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-3, atol=1e-3,
    expect_ir={"09-custom.mlir": {"vcix.v.i": 26, "arith.addf": 26, "pytorchsim.wait": 5}},
    expect_ir_on={"tpuv3": {"09-custom.mlir": {"vcix.v.i": 61, "arith.addf": 61}}},
)
