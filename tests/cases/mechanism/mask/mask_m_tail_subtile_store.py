"""A 64-row output stored from a 128-row tile, its 1024 columns cut into subtiles: the TOG sizes its masked
M-tail rows at run time (64), as the subtile's clamp moves them. Breaks under --tog if togcheck and the
emitter disagree on that store again (the TOG's [256, 64] against an unclamped [256, 128])."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 64, 256, 768
T, NP = 128, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_m_tail_subtile_store_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                                         T: tl.constexpr, NP: tl.constexpr):
        rk = tl.arange(0, K)
        rm = tl.arange(0, T)
        rn = tl.arange(0, NP)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0.0)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
        c = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], c.to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    return mask_m_tail_subtile_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="mask_m_tail_subtile_store_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "M": "constexpr", "K": "constexpr",
               "N": "constexpr", "T": "constexpr", "NP": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "T": T, "NP": NP},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)), Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1, 1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-1,
)
