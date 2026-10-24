"""RoPE's rotate_half on a flat index, a select on `(x % D) < D/2`, with D the lane count: the flat index
spreads over the lanes as `slot * lanes + lane`, and the program's base and the slot term are multiples of
D, so the remainder is the lane number's alone."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, D = 64, 128
HALF = D // 2
BLOCK = R * D // 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_select_on_lane_remainder_kernel(x_ptr, o_ptr, D: tl.constexpr, HALF: tl.constexpr,
                                              BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = x % D
        row = x // D
        lo = col < HALF
        a = tl.load(x_ptr + (HALF + D * row + col), lo, other=0.0)
        b = tl.load(x_ptr + (D * row + (col - HALF)), col >= HALF, other=0.0)
        tl.store(o_ptr + x, tl.where(lo, -a, b))

    return ops_select_on_lane_remainder_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, D)
    return {"o": torch.cat([-x[:, HALF:], x[:, :HALF]], dim=-1).reshape(-1).clone()}


SPEC = KernelSpec(
    name="ops_select_on_lane_remainder_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "D": "constexpr", "HALF": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"D": D, "HALF": HALF, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (R * D,)), Arg("o", OUT, "float32", (R * D,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"vector.step": 0}},
)
