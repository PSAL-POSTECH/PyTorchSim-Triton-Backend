"""Pins a dot whose block is wider than its matrix (N = 768 in a block of 1024): the stream walks
the 3x2x2 subtiles the mask leaves live, their count taken from the load's clamp at run time, not
the block's 4x2x2. Breaks if the stream goes back to a constant trip count (no scf.for left), or
the values drift."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 768, 512
BN = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_streamed_masked_tail(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                                       K: tl.constexpr, BN: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, BN)
        k = tl.arange(0, K)
        live = n[None, :] < N
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :], mask=live, other=0.0)
        tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, b), mask=live)

    return tile_gemm_streamed_masked_tail


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g), "b": torch.randn(K, N, generator=g)}


def reference(inputs):
    return {"c": inputs["a"] @ inputs["b"]}


SPEC = KernelSpec(
    name="tile_gemm_streamed_masked_tail", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr", "BN": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BN": BN},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-3, atol=1e-3,
    expect_ir={"09-custom.mlir": {"scf.for": 1, "vcix.v.i": 48}},
)
