"""Pins that a 3-D tl.dot is refused: it reaches p08_lower_matmul as a batched torchsim.dot, and
that lowering has no batch loop. Breaks if it starts compiling without one, which would compute
the first batch only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, M, N, K = 2, 16, 16, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_dot_3d_refused(a_ptr, b_ptr, c_ptr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        bb = tl.arange(0, B)[:, None, None]
        m = tl.arange(0, M)[None, :, None]
        n = tl.arange(0, N)[None, None, :]
        k = tl.arange(0, K)
        a = tl.load(a_ptr + bb * M * K + m * K + k[None, None, :])
        b = tl.load(b_ptr + bb * K * N + k[None, :, None] * N + n)
        tl.store(c_ptr + bb * M * N + m * N + n, tl.dot(a, b))

    return ops_dot_3d_refused


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(B, M, K, generator=g), "b": torch.randn(B, K, N, generator=g)}


def reference(inputs):
    return {"c": inputs["a"] @ inputs["b"]}


SPEC = KernelSpec(
    name="ops_dot_3d_refused", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "B": "constexpr", "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"B": B, "M": M, "N": N, "K": K},
    args=[Arg("a", IN, "float32", (B, M, K)), Arg("b", IN, "float32", (B, K, N)),
          Arg("c", OUT, "float32", (B, M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-3, atol=1e-3,
    expect="refuse",
    expect_why="a 3-D tl.dot reaches p08 as a batched torchsim.dot, which has no batch "
               "loop to lower to",
)
