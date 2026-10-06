"""GRID: `((r*N + c) % T) // K`, a step under a SUM wrap. The wrap is over the flattened index.

`flat_shape` wraps the two axes taken together, so its runs do not line up with
either axis's own -- and a step is a per-axis fact. `(x mod W)//K == (x//K) mod
(W/K)` needs `K | W`, which a sum wrap gives no axis, so this one demotes.

A division under a sum wrap: no axis owns the wrap, so no axis can carry the step and the address is per-element.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, T, K = 4, 8, 13, 3     # 3 divides neither 13 nor the 32 flat positions
NX = (T - 1) // K + 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_under_a_sum_wrap_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                     N: tl.constexpr, T: tl.constexpr,
                                     K: tl.constexpr):
        r = tl.arange(0, M)[:, None]
        c = tl.arange(0, N)[None, :]
        v = tl.load(x_ptr + ((r * N + c) % T) // K)
        tl.store(o_ptr + r * N + c, v)

    return step_under_a_sum_wrap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(M)[:, None]
    c = torch.arange(N)[None, :]
    return {"o": inputs["x"][((r * N + c) % T) // K].reshape(-1)}


SPEC = KernelSpec(
    name="step_under_a_sum_wrap_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr", "T": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "T": T, "K": K},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
