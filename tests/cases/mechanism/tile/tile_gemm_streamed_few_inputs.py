"""Pins a bf16 dot with fewer inputs a panel than prep/push pairs: M = 32 is two inputs, K = 2048
eight panels of 16 pairs. Each panel is pushed once, one pair after each input of the panel before
and the rest before its own first input. Breaks if a panel is pushed twice or the values change."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 32, 256, 2048


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_streamed_few_inputs(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                                      K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, b))

    return tile_gemm_streamed_few_inputs


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    return {"c": inputs["a"].float() @ inputs["b"].float()}


SPEC = KernelSpec(
    name="tile_gemm_streamed_few_inputs", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-3, atol=1e-3,
)
