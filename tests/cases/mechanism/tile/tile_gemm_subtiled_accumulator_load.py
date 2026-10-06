"""Pins a loaded accumulator (tl.dot(a, b, acc)) twice the 256 array on N and M: C arrives as
one async DMA per (n, m) subtile, waited for right before the pop that first adds into it.
Breaks if C stays whole-tile, or its wait lands after the matmul that reads it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_subtiled_accumulator_load(a_ptr, b_ptr, c_ptr, y_ptr, M: tl.constexpr,
                                            N: tl.constexpr, K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        acc = tl.load(c_ptr + m[:, None] * N + n[None, :])
        tl.store(y_ptr + m[:, None] * N + n[None, :], tl.dot(a, b, acc))

    return tile_gemm_subtiled_accumulator_load


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g), "b": torch.randn(K, N, generator=g),
            "c": torch.randn(M, N, generator=g)}


def reference(inputs):
    return {"y": inputs["c"] + inputs["a"] @ inputs["b"]}


SPEC = KernelSpec(
    name="tile_gemm_subtiled_accumulator_load", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", IN, "float32", (M, N)), Arg("y", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-3, atol=1e-3,
    expect_ir={"05-custom.mlir": {"subtile_size": 6}},
)
