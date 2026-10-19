"""Pins a 128^3 bf16 dot whose f32 rows are cast to i32 and folded by xor, one i32 a row: the walk
over C's rows runs at the pops, its seed the xor's identity. Breaks if the fold is lost or a row's
seed is folded more than once (values)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_row_xor_at_pop(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                                 K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        acc = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(c_ptr + m, tl.xor_sum(acc.to(tl.int32), 1))

    return tile_gemm_row_xor_at_pop


def make_inputs(torch, seed=0):
    # Small integers: every sum is exact in f32, so the rows' i32 are torch's.
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randint(-4, 5, (M, K), generator=g).to(torch.bfloat16),
            "b": torch.randint(-4, 5, (K, N), generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = (inputs["a"].float() @ inputs["b"].float()).to(torch.int32)
    out = torch.zeros(M, dtype=torch.int32)
    for j in range(N):
        out ^= x[:, j]
    return {"c": out}


SPEC = KernelSpec(
    name="tile_gemm_row_xor_at_pop", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*i32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "int32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
)
