"""Pins a dot of K 384 (one lane-tile pair and a lone tile) and N 256 (one N pair): on two arrays the pair's panel
goes to array 0 and the lone 128-deep tile's to array 1, popped as one sum, as the v6e gives K 384's 128-deep
panel to its other MXU; on one array both panels run in turn. The values say whether either array's part of
K dropped or doubled."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 256, 256, 384


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_k_pair_and_lone_tile(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BK: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, BK)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :], mask=k[None, :] < K, other=0.0)
        b = tl.load(b_ptr + k[:, None] * N + n[None, :], mask=k[:, None] < K, other=0.0)
        tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, b, out_dtype=tl.float32).to(tl.bfloat16))

    return tile_gemm_k_pair_and_lone_tile


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="tile_gemm_k_pair_and_lone_tile", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "M": "constexpr", "N": "constexpr",
               "K": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BK": 512},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)), Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
