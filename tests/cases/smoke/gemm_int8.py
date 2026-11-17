"""GEMM of int8 operands summed in int32, tl.dot(a, b, out_dtype=tl.int32) as Inductor's _int_mm template emits it:
format 1 on the matrix unit (four int8 rows a word), its pops int32 (format 1) added into C as int32. K = 2 * BK so
the loop is not unrolled away; the operands take the extremes -128 and 127, and the result is exact."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 256, 256, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_int8_kernel(a_ptr, b_ptr, c_ptr,
                       M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b, out_dtype=tl.int32)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return gemm_int8_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-128, 128, (M, K), generator=g).to(torch.int8)
    b = torch.randint(-128, 128, (K, N), generator=g).to(torch.int8)
    a[0, :2], b[:2, 0] = torch.tensor([-128, 127], dtype=torch.int8), torch.tensor([-128, 127], dtype=torch.int8)
    return {"a": a,
            "b": b}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].to(torch.int64) @ inputs["b"].to(torch.int64)).to(torch.int32)}


SPEC = KernelSpec(
    name="gemm_int8_kernel",
    kernel=kernel,
    signature={"a_ptr": "*i8", "b_ptr": "*i8", "c_ptr": "*i32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "int8", (M, K)), Arg("b", IN, "int8", (K, N)), Arg("c", OUT, "int32", (M, N))],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
