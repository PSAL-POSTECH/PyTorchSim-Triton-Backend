"""Rows masked `i >= lo`, lo = 20 a run-time scalar on a 16-row tile: the live extent is <= 0 in the only
work item, so the TOG runs no DMA (under if) and the store reads the masked fill. Breaks if togcheck does
not take that as written and the transfer as never run ("reads buffer 0, which nothing ... writes")."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, LO = 16, 32, 20


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_dma_empty_in_every_work_item(x_ptr, y_ptr, lo, R: tl.constexpr, C: tl.constexpr):
        i = tl.arange(0, R)[:, None]
        j = tl.arange(0, C)[None, :]
        v = tl.load(x_ptr + (i - lo) * C + j, mask=i >= lo, other=0.0)
        tl.store(y_ptr + i * C + j, v)

    return mask_tog_dma_empty_in_every_work_item


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(1, C, generator=g)}


def reference(inputs):
    import torch
    return {"y": torch.zeros(R, C)}


SPEC = KernelSpec(
    name="mask_tog_dma_empty_in_every_work_item", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "lo": "i32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (1, C)), Arg("y", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"lo": LO}},
)
