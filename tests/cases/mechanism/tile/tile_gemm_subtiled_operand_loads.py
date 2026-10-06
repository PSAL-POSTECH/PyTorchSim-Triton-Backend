"""Pins a matmul tile twice the 256 array on every dim loading A and B as one async DMA per
array subtile (2x2 each, 8 in all), each waited for right before its rows are pushed.
Breaks if the loads stay whole-tile, or a subtile's slot or offset drifts from its push."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_subtiled_operand_loads(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, b))

    return tile_gemm_subtiled_operand_loads


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g), "b": torch.randn(K, N, generator=g)}


def reference(inputs):
    return {"c": inputs["a"] @ inputs["b"]}


SPEC = KernelSpec(
    name="tile_gemm_subtiled_operand_loads", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-3, atol=1e-3,
    expect_ir={"05-custom.mlir": {"subtile_size": 4}},
)
