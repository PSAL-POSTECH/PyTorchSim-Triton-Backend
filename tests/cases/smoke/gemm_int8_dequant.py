"""W8A8: gemm_int8.py's int32 sum with a dequant epilogue -- to f32, times the per-row and per-column scales,
plus a bias -- as torchao's int8 dynamic quantization leaves it after the matmul."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 256, 256, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_int8_dequant_kernel(a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, c_ptr,
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
        sa = tl.load(sa_ptr + offs_m)
        sb = tl.load(sb_ptr + offs_n)
        bias = tl.load(bias_ptr + offs_n)
        out = acc.to(tl.float32) * sa[:, None] * sb[None, :] + bias[None, :]
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], out)

    return gemm_int8_dequant_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-128, 128, (M, K), generator=g).to(torch.int8)
    b = torch.randint(-128, 128, (K, N), generator=g).to(torch.int8)
    a[0, :2], b[:2, 0] = torch.tensor([-128, 127], dtype=torch.int8), torch.tensor([-128, 127], dtype=torch.int8)
    return {"a": a,
            "b": b,
            "sa": torch.rand(M, generator=g) / 128, "sb": torch.rand(N, generator=g) / 128,
            "bias": torch.randn(N, generator=g)}


def reference(inputs):
    import torch
    acc = (inputs["a"].to(torch.int64) @ inputs["b"].to(torch.int64)).to(torch.int32)
    return {"c": acc.to(torch.float32) * inputs["sa"][:, None] * inputs["sb"][None, :] + inputs["bias"][None, :]}


SPEC = KernelSpec(
    name="gemm_int8_dequant_kernel",
    kernel=kernel,
    signature={"a_ptr": "*i8", "b_ptr": "*i8", "sa_ptr": "*fp32", "sb_ptr": "*fp32", "bias_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "int8", (M, K)), Arg("b", IN, "int8", (K, N)), Arg("sa", IN, "float32", (M,)),
          Arg("sb", IN, "float32", (N,)), Arg("bias", IN, "float32", (N,)), Arg("c", OUT, "float32", (M, N))],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
)
