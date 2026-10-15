"""Pins that a bf16 tl.dot of M=2 rows is refused: two bf16 rows pack into one word, narrower than the two a bank
moves at the least. A GQA decode group of Hq/Hkv = 2 query heads (qwen3-0.6b, 16/8) is this dot; validation/attention
pads it to 4 rows. Breaks if M=2 starts lowering, which then has to show it computes both rows."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 2, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_dot_bf16_m2_refused(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + n[None, :])
        tl.store(c_ptr + m[:, None] * N + n[None, :], tl.dot(a, b, out_dtype=tl.float32))

    return ops_dot_bf16_m2_refused


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    return {"c": inputs["a"].float() @ inputs["b"].float()}


SPEC = KernelSpec(
    name="ops_dot_bf16_m2_refused", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect="refuse",
    expect_why="bf16 M=2 is one word of inputs, narrower than the two a bank moves at the least "
               "(lower_matmul._refuse)",
)
