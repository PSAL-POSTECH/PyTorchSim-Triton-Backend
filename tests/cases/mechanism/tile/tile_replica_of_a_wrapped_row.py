"""A wrapped (modulo) row read as a replica and broadcast over the lanes: a03 carries the transfer's cut onto the replica's axes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 256, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_replica_of_a_wrapped_row_kernel(x_ptr, w_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        r = tl.arange(0, M)
        c = tl.arange(0, N)
        x = tl.load(x_ptr + r[:, None] * N + c[None, :])
        w = tl.load(w_ptr + (c + 3) % N)
        tl.store(o_ptr + r[:, None] * N + c[None, :], x + w[None, :])

    return tile_replica_of_a_wrapped_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g), "w": torch.randn(N, generator=g)}


def reference(inputs):
    import torch
    x = inputs["x"].view(M, N)
    c = torch.arange(N)
    return {"o": (x + inputs["w"][(c + 3) % N][None, :]).reshape(-1)}


SPEC = KernelSpec(
    name="tile_replica_of_a_wrapped_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("w", IN, "float32", (N,)), Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
