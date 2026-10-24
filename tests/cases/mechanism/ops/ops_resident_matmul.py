"""C = A @ B with B loaded whole once at the top and A streamed by 128-row tiles in a loop, K whole: the shape
of the resident mm. The compiler makes B's load a copy node and the kernel's TOG reads B held
(tests/runtime/copy_node_cases.py); here Spike checks the values."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, T = 384, 256, 192, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_resident_matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                                   NP: tl.constexpr, T: tl.constexpr):
        rk = tl.arange(0, K)
        rn = tl.arange(0, NP)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0.0)
        for i in range(0, M // T):
            rm = i * T + tl.arange(0, T)
            a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
            c = tl.dot(a, b, out_dtype=tl.float32)
            tl.store(c_ptr + rm[:, None] * N + rn[None, :], c.to(tl.bfloat16), mask=rn[None, :] < N)

    return ops_resident_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_resident_matmul_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "M": "constexpr", "K": "constexpr",
               "N": "constexpr", "NP": "constexpr", "T": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "NP": 256, "T": T},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)), Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1, 1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-1,
)
