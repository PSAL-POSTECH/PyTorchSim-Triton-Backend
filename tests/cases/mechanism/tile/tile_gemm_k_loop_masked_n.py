"""Pins a bf16 dot as Inductor's mm template writes it: K walked in blocks (acc += dot), N = 384 in a
block of 512 that the loads mask, the f32 sum cast to bf16 after the loop and stored through the
same mask. On the matrix unit the K loop's first trip takes a C its fill proves zero, its last runs
the cast at its pops, and the N tiles past 384 are neither fed nor popped; the values say whether
any of it dropped or doubled a column or a K block."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 64, 384, 384
BLOCK_N, BLOCK_K = 512, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_k_loop_masked_n(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                                  K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, BLOCK_N)
        acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + m[:, None] * K + k[None, :], mask=k[None, :] < K, other=0.0)
            b = tl.load(b_ptr + k[:, None] * N + n[None, :],
                        mask=(k[:, None] < K) & (n[None, :] < N), other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32)
        tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=n[None, :] < N)

    return tile_gemm_k_loop_masked_n


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch_bf16())}


def torch_bf16():
    import torch
    return torch.bfloat16


SPEC = KernelSpec(
    name="tile_gemm_k_loop_masked_n", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "M": "constexpr",
               "N": "constexpr", "K": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
