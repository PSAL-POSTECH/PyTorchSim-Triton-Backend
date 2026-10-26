"""C = A @ B with K and N of 384 on blocks of 512: the masked loads fill the padding with 0, so
pytorchsim-drop-unread-padding cuts the dot's buffers to 384 along N (what the store keeps) and K (where the
product is 0); Spike checks the values (tests/passes/drop_unread_padding_cases.py checks the cut)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, P = 128, 384, 384, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_dot_cut_to_numel_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                                    P: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, P)
        rn = tl.arange(0, P)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rk[None, :] < K, other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        c = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], c.to(tl.bfloat16), mask=rn[None, :] < N)

    return ops_dot_cut_to_numel_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_dot_cut_to_numel_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "M": "constexpr", "K": "constexpr",
               "N": "constexpr", "P": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "P": P},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)), Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1, 1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-1,
)
