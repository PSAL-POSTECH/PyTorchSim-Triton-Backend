"""`(x % D) < D/2` and `(x % D) >= D/2` on a flat index: RoPE's rotate_half, `cat([-x[..., D/2:],
x[..., :D/2]], -1)`, as Inductor writes it -- two masked loads shifted by D/2 and a select.

The mask is a box on every run of D lanes, not on the flat axis, so MaskAnalysis refused it and
both loads were gathers. A remainder compared with a constant is now read as a period D and a box
[lo, hi) inside it, when the range starts on a lap (pid*BLOCK, a multiple of D) and runs whole
laps; the load's pointer, already cut at D by the `// D` it carries, takes the box on its inner
axis, and the wrap that box never crosses is dropped.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, D = 8, 128
HALF = D // 2
BLOCK = R * D // 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_periodic_remainder_halves_kernel(x_ptr, o_ptr, D: tl.constexpr, HALF: tl.constexpr,
                                              BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = x % D
        row = x // D
        lo = col < HALF
        a = tl.load(x_ptr + (HALF + D * row + col), lo, other=0.0)
        b = tl.load(x_ptr + (D * row + (col - HALF)), col >= HALF, other=0.0)
        tl.store(o_ptr + x, tl.where(lo, -a, b))

    return mask_periodic_remainder_halves_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, D)
    return {"o": torch.cat([-x[:, HALF:], x[:, :HALF]], dim=-1).reshape(-1).clone()}


SPEC = KernelSpec(
    name="mask_periodic_remainder_halves_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "D": "constexpr", "HALF": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"D": D, "HALF": HALF, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (R * D,)), Arg("o", OUT, "float32", (R * D,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.load": 2}},
)
