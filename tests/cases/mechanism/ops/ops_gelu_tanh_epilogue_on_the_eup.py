"""gemma's MLP gate, gelu_tanh(g) * (A @ B) in bf16, run as the GEMM's epilogue at its pops: the gelu's tanh pushed
to the EUP where its sf.vc.v.iv was. Breaks if an epilogue's tanh goes back to its sf.vc.v.iv (the counts), or the
push's field is lost or the pops of the EUP and the MXU cross (the values)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128
EUP_COUNTS = {'"llvm.riscv.sf.vc.vv.se"': 1, '"llvm.riscv.sf.vc.v.x.se"': 1, '"vcix.v.iv"': 2}


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def ops_gelu_tanh_epilogue_on_the_eup_kernel(a_ptr, b_ptr, g_ptr, c_ptr,
                                                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        up = acc.to(tl.bfloat16).to(tl.float32)
        idx = offs_m[:, None] * N + offs_n[None, :]
        g = tl.load(g_ptr + idx).to(tl.float32)
        gelu = g * 0.5 * (1.0 + libdevice.tanh((g + 0.044715 * g * g * g) * 0.7978845608028654))
        tl.store(c_ptr + idx, (gelu * up).to(tl.bfloat16))

    return ops_gelu_tanh_epilogue_on_the_eup_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": (torch.randn(M, K, generator=g) * 0.25).to(torch.bfloat16),
            "b": (torch.randn(K, N, generator=g) * 0.25).to(torch.bfloat16),
            "g": (torch.randn(M, N, generator=g) * 2).to(torch.bfloat16)}


def reference(inputs):
    import torch
    up = (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16).float()
    gelu = torch.nn.functional.gelu(inputs["g"].float(), approximate="tanh")
    return {"c": (gelu * up).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_gelu_tanh_epilogue_on_the_eup_kernel", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "g_ptr": "*bf16", "c_ptr": "*bf16",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)), Arg("g", IN, "bfloat16", (M, N)),
          Arg("c", OUT, "bfloat16", (M, N))],
    grid=(M // BM, N // BN), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"09-custom.mlir": EUP_COUNTS},
)
