"""A runtime mask on the wrapped axis stays a per-element gather: a piece's box is a static [lo, hi),
and meeting it with a bound known only at run time needs a box sized at run time per piece, which
nothing states yet. Pins that p01 keeps it a gather (and p11 refuses one, tests/passes)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, W, O, M, GRID = 32, 24, 4, 26, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_clamped_at_run_time_kernel(x_ptr, o_ptr, N: tl.constexpr, W: tl.constexpr,
                                            O: tl.constexpr, M: tl.constexpr):
        pid = tl.program_id(0)
        p = tl.arange(0, N)
        v = tl.load(x_ptr + (p + O) % W, mask=p < M + pid, other=-1.0)
        tl.store(o_ptr + pid * N + p, v)

    return ptr_wrap_clamped_at_run_time_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = torch.arange(N)
    rows = [torch.where(p < M + pid, inputs["x"][(p + O) % W], torch.tensor(-1.0))
            for pid in range(GRID)]
    return {"o": torch.cat(rows)}


SPEC = KernelSpec(
    name="ptr_wrap_clamped_at_run_time_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "W": "constexpr",
               "O": "constexpr", "M": "constexpr"},
    constexprs={"N": N, "W": W, "O": O, "M": M},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (GRID * N,))],
    grid=(GRID,), reference=reference, make_inputs=make_inputs,
    expect_ir={"09-custom.mlir": {"pytorchsim.transfer": 2}},
)
