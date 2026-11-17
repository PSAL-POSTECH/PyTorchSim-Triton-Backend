"""E4M3 dot on the matrix unit: A and B narrowed in the kernel, pushed as words of four E4M3 rows (format 2),
f32 popped. Breaks if lower_matmul's _get_fp8_format loses E4M3 (the dot refuses), if the model's format 2
stops being E4M3, or if packed_view reads the bytes in another order than the model unpacks them."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_fp8_e4m3_dot_kernel(a_ptr, b_ptr, c_ptr,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a.to(tl.float8e4nv), b.to(tl.float8e4nv),
                          out_dtype=tl.float32, max_num_imprecise_acc=0)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return ops_fp8_e4m3_dot_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    d = torch.float8_e4m3fn
    return {"c": torch.matmul(inputs["a"].to(d).float(), inputs["b"].to(d).float())}


SPEC = KernelSpec(
    name="ops_fp8_e4m3_dot_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="pass",
)
