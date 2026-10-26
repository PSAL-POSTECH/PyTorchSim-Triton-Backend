"""GEMM with fp16 operands and an fp32 accumulator: the only matmul in the suite whose linalg op carries two
element types, so it pins lower_matmul's elt/acc_elt split."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Dim 0 of every tile divides by PYTORCHSIM_COMPILE_VECTORLANE_SIZE (128) -- the DMA splits it
# across lanes. N and K are whole lane tiles; M is two tiles and K is three,
# so the grid is 2 x 1 and the reduction loop runs three trips.
M, N, K = 256, 128, 384
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_fp16_kernel(a_ptr, b_ptr, c_ptr,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        # f32 accumulator on f16 loads -- this is what makes the linalg.matmul
        # mixed. `tl.dot` takes its output type from `acc`, and the operands
        # keep theirs: ins(f16, f16) outs(f32).
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return gemm_fp16_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.float16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.float16),
    }


def reference(inputs):
    import torch
    # Widened to f32 BEFORE the matmul, deliberately: that is what the hardware
    # does -- the array multiplies in the input format and accumulates in float
    # (the v6 matrix unit pops f32, Coprocessor examples/tpuv6e). A torch
    # matmul left in fp16 would accumulate in half and be the wrong reference to
    # compare an f32 accumulator against.
    return {"c": torch.matmul(inputs["a"].to(torch.float32),
                              inputs["b"].to(torch.float32))}


SPEC = KernelSpec(
    name="gemm_fp16_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float16", (M, K)),
        Arg("b", IN, "float16", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # The inputs are exact in f32 and the accumulator is f32, so what is left is
    # summation ORDER -- the array's K walk against torch's. Measured
    # 5.72205e-05 absolute, 0.0202492 relative: the relative figure is large
    # because some entries of C are near zero, which is why the pass is decided
    # by atol at this scale and not by rtol.
    rtol=1e-3, atol=1e-3,
)
