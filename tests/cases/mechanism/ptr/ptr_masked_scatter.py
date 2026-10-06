"""A masked store through a select of two bases writes both: a tile of pointers' scatter is bound by all
of memory, not by one argument, and parks its masked-off writes below address 0.
Breaks, with `a` or `b` wrong, if the bound narrows to one buffer again while the kept writes span two."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

N = 128
LIVE = 100


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_masked_scatter_kernel(x_ptr, a_ptr, b_ptr, LIVE: tl.constexpr,
                                  BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = tl.where((r % 2) == 0, a_ptr + r, b_ptr + r)
        tl.store(p, tl.load(x_ptr + r), mask=r < LIVE)

    return ptr_masked_scatter_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "a": torch.arange(1, N + 1, dtype=torch.float32),
            "b": -torch.arange(1, N + 1, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(N)
    live = r < LIVE
    a, b = inputs["a"].clone(), inputs["b"].clone()
    a[live & (r % 2 == 0)] = inputs["x"][live & (r % 2 == 0)]
    b[live & (r % 2 == 1)] = inputs["x"][live & (r % 2 == 1)]
    return {"a": a, "b": b}


SPEC = KernelSpec(
    name="ptr_masked_scatter_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "LIVE": "constexpr", "BLOCK": "constexpr"},
    constexprs={"LIVE": LIVE, "BLOCK": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("a", INOUT, "float32", (N,)),
          Arg("b", INOUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"04-transformed.mlir": {"dram_arg = -2": 1}},
)
