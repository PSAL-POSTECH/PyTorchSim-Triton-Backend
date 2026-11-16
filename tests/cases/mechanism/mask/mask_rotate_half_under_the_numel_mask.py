"""rotate_half over a whole tensor in one program, halves ANDed with `x < N` as Inductor writes them.

The `col >= HALF` load reads D/2 before its tensor where masked off; its box must stay a box through
the `& (x < N)` conjunct. Breaks if it falls to an element mask: MaskedBoxBeforeTensor, or a read before x.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, D = 8, 128
HALF = D // 2
N = R * D


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_rotate_half_whole_grid_xmask_kernel(x_ptr, o_ptr, D: tl.constexpr, HALF: tl.constexpr,
                                                 N: tl.constexpr):
        x = tl.program_id(0) * N + tl.arange(0, N)
        m = x < N
        col = x % D
        row = x // D
        lo = col < HALF
        a = tl.load(x_ptr + (HALF + D * row + col), lo & m, other=0.0)
        b = tl.load(x_ptr + (D * row + (col - HALF)), (col >= HALF) & m, other=0.0)
        tl.store(o_ptr + x, tl.where(lo, -a, b), m)

    return mask_rotate_half_whole_grid_xmask_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, D)
    return {"o": torch.cat([-x[:, HALF:], x[:, :HALF]], dim=-1).reshape(-1).clone()}


SPEC = KernelSpec(
    name="mask_rotate_half_whole_grid_xmask_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "D": "constexpr", "HALF": "constexpr",
               "N": "constexpr"},
    constexprs={"D": D, "HALF": HALF, "N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}},
)
