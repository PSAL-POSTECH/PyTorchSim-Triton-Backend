"""`(p // K + o) % W` with K not dividing the extent: one axis that steps and wraps, digits
[(rest, 1, W, o), (K, 0)], read as six runs of rows 1 2 3 0 1 2 -- six pieces, not a gather.
Breaks if a02 stops composing a wrap with a step on one axis: the load turns per-element."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K, W, O = 32, 6, 4, 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_of_an_uneven_step_kernel(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
                                          W: tl.constexpr, O: tl.constexpr):
        p = tl.arange(0, N)
        tl.store(o_ptr + p, tl.load(x_ptr + (p // K + O) % W))

    return ptr_wrap_of_an_uneven_step_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"][(torch.arange(N) // K + O) % W]}


SPEC = KernelSpec(
    name="ptr_wrap_of_an_uneven_step_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "K": "constexpr",
               "W": "constexpr", "O": "constexpr"},
    constexprs={"N": N, "K": K, "W": W, "O": O},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"09-custom.mlir": {"pytorchsim.transfer": 7}},
)
