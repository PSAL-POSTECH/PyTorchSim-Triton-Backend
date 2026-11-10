"""`offs = pid * 64 + r` masked `offs >= 40`: program 0's clamp low is 40, program 1's negative (held at 0),
built at run time; togcheck judges the footprint. expect_ir counts tog.cpp's three `+`, one the start
dram_idx + max(low, 0): breaks if the start is the unmasked dram_idx again."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, H, G = 64, 40, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_runtime_low_from_the_program_id(x_ptr, y_ptr, B: tl.constexpr, H: tl.constexpr):
        offs = tl.program_id(0) * B + tl.arange(0, B)
        tl.store(y_ptr + offs, tl.load(x_ptr + offs - H, mask=offs >= H, other=0.0))

    return mask_tog_runtime_low_from_the_program_id


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(B * G - H, generator=g)}


def reference(inputs):
    import torch
    return {"y": torch.cat([torch.zeros(H), inputs["x"]])}


SPEC = KernelSpec(
    name="mask_tog_runtime_low_from_the_program_id", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "B": "constexpr", "H": "constexpr"},
    constexprs={"B": B, "H": H},
    args=[Arg("x", IN, "float32", (B * G - H,)), Arg("y", OUT, "float32", (B * G,))],
    grid=(G,), reference=reference, make_inputs=make_inputs,
    expect_ir={"tog.cpp": {"+": 3}}, expect_stage="togcheck",
)
