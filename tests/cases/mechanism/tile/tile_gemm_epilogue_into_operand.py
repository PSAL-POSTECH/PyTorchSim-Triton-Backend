"""Pins x = dot(x, w).to(bf16) twice, in a scf.for: the cast's store lands in x's own buffer, A of
the dot it follows. Breaks if that cast runs at the pops -- they write x while the stream still
reads it as A (values) -- rather than staying a loop marked torchsim.kept_apart."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = K = 512
STEPS = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_epilogue_into_operand(x_ptr, w_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr,
            STEPS: tl.constexpr):
        m = tl.arange(0, M)
        k = tl.arange(0, K)
        x = tl.load(x_ptr + m[:, None] * K + k[None, :])
        w = tl.load(w_ptr + k[:, None] * K + k[None, :])
        for i in range(STEPS):
            x = tl.dot(x, w, out_dtype=tl.float32).to(tl.bfloat16)
        tl.store(o_ptr + m[:, None] * K + k[None, :], x)

    return tile_gemm_epilogue_into_operand


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": (torch.randn(M, K, generator=g) / 8).to(torch.bfloat16),
            "w": (torch.randn(K, K, generator=g) / 8).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"]
    for _ in range(STEPS):
        x = (x.float() @ inputs["w"].float()).to(torch.bfloat16)
    return {"o": x}


SPEC = KernelSpec(
    name="tile_gemm_epilogue_into_operand", kernel=kernel,
    signature={"x_ptr": "*bf16", "w_ptr": "*bf16", "o_ptr": "*bf16",
               "M": "constexpr", "K": "constexpr", "STEPS": "constexpr"},
    constexprs={"M": M, "K": K, "STEPS": STEPS},
    args=[Arg("x", IN, "bfloat16", (M, K)), Arg("w", IN, "bfloat16", (K, K)),
          Arg("o", OUT, "bfloat16", (M, K))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=3e-2, atol=3e-2,
    expect="pass",
)
