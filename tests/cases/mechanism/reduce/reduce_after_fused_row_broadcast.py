"""A row sum of an elementwise chain one of whose inputs is a per-row value: fusion
takes that broadcast in, the reduction's input map loses the folded dim, and the
generic has to be made a `linalg.reduce` again before vectorize sees it.

convnextv2's addmm+gelu+vector_norm in Inductor's looped form: a per-row bias inside the squared sum, a one-trip loop and a keepdim sum -- the reshape fusion expands. Without the loop and the keepdim the fused reduction never loses its dim and this passes either way.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 64, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_after_fused_row_broadcast_kernel(x_ptr, b_ptr, y_ptr,
                                                M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        b = tl.load(b_ptr + rows)
        acc = tl.full([M, N], 0, tl.float32)
        for r0 in range(0, N, N):
            x = tl.load(x_ptr + rows * N + (r0 + cols))
            h = x + b
            acc = acc + h * h
        tl.store(y_ptr + rows, tl.sum(acc, 1)[:, None])

    return reduce_after_fused_row_broadcast_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32),
            "b": torch.randn(M, generator=g, dtype=torch.float32)}


def reference(inputs):
    h = inputs["x"].reshape(M, N) + inputs["b"][:, None]
    return {"y": (h * h).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_after_fused_row_broadcast_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*fp32", "y_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("b", IN, "float32", (M,)),
          Arg("y", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
