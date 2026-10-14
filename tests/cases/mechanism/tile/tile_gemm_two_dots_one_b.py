"""Pins two bf16 dots in a block that read one B, nothing writing it between: N = 256 walked in
windows of 128, each window's B (K = 256, one K pair) taken by two dots of 128 rows. The second
dot's panel is the one the first left staged, so it is not staged again; the values break if the
second dot latched anything but that panel, or if a panel of another window were taken as held."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 256, 256, 256
BLOCK_M, BLOCK_N = 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_two_dots_one_b(a_ptr, b_ptr, c_ptr, N: tl.constexpr, K: tl.constexpr,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        k = tl.arange(0, K)
        m = tl.arange(0, BLOCK_M)
        a_0 = tl.load(a_ptr + m[:, None] * K + k[None, :])
        a_1 = tl.load(a_ptr + (BLOCK_M + m)[:, None] * K + k[None, :])
        for n0 in range(0, N, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            b = tl.load(b_ptr + k[:, None] * N + n[None, :])
            acc = tl.dot(a_0, b, out_dtype=tl.float32)
            tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(tl.bfloat16))
            acc = tl.dot(a_1, b, out_dtype=tl.float32)
            tl.store(c_ptr + (BLOCK_M + m)[:, None] * N + n[None, :], acc.to(tl.bfloat16))

    return tile_gemm_two_dots_one_b


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="tile_gemm_two_dots_one_b", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "N": "constexpr",
               "K": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
    constexprs={"N": N, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
