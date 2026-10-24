"""y = x - max(x, axis 0) on a 128x16 block holding 8 columns: the column max is broadcast back over the rows
by a pytorchsim.copy, so pytorchsim-fit-to-numel cuts every buffer to the 8 columns the store keeps through that
copy; Spike checks the values (tests/passes/fit_to_numel_cases.py checks the cut)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, P = 128, 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_broadcast_copy_cut_to_numel_kernel(x_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr, P: tl.constexpr):
        r = tl.arange(0, R)
        c = tl.arange(0, P)
        x = tl.load(x_ptr + r[:, None] * C + c[None, :], mask=c[None, :] < C, other=0.0).to(tl.float32)
        m = tl.max(x, axis=0)
        tl.store(y_ptr + r[:, None] * C + c[None, :], (x - m[None, :]).to(tl.bfloat16), mask=c[None, :] < C)

    return ops_broadcast_copy_cut_to_numel_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R, C, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].float()
    return {"y": (x - x.max(dim=0, keepdim=True).values).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ops_broadcast_copy_cut_to_numel_kernel",
    kernel=kernel,
    signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "R": "constexpr", "C": "constexpr", "P": "constexpr"},
    constexprs={"R": R, "C": C, "P": P},
    args=[Arg("x", IN, "bfloat16", (R, C)), Arg("y", OUT, "bfloat16", (R, C))],
    grid=(1, 1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=5e-2,
)
