"""Pins an int8 dot of K 320 (one lane-tile pair and a lone tile of 64 rows) and N 256: on two arrays the pair's panel
goes to array 0 and the short tile's to array 1, its rows past 64 zero, both int32 sums taken by one vpop.add;
on one array both panels run in turn. The values say whether either array's part of K dropped, doubled or
read past the short tile."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 256, 256, 320


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_k_pair_and_short_tile_int8(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BK: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, BK)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :], mask=k[None, :] < K, other=0)
        b = tl.load(b_ptr + k[:, None] * N + n[None, :], mask=k[:, None] < K, other=0)
        tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, b, out_dtype=tl.int32))

    return tile_gemm_k_pair_and_short_tile_int8


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randint(-128, 128, (M, K), generator=g, dtype=torch.int8),
            "b": torch.randint(-128, 128, (K, N), generator=g, dtype=torch.int8)}


def reference(inputs):
    import torch
    return {"c": inputs["a"].int() @ inputs["b"].int()}


SPEC = KernelSpec(
    name="tile_gemm_k_pair_and_short_tile_int8", kernel=kernel,
    signature={"a_ptr": "*i8", "b_ptr": "*i8", "c_ptr": "*i32", "M": "constexpr", "N": "constexpr",
               "K": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BK": 512},
    args=[Arg("a", IN, "int8", (M, K)), Arg("b", IN, "int8", (K, N)), Arg("c", OUT, "int32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
)
