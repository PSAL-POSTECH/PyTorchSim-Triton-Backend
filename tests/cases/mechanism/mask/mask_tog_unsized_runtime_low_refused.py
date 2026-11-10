"""The peeled 5-d load of mask_tog_peeled_load_sized_across_its_loop with its clamp `e >= el` a run-time
scalar: a load waited in another block is not sized, so its DMA would start at column 0, el columns before
the live ones. Refused at tog; breaks if the TOG moves the unmasked tile again (it passed on 11930f50)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

A, B, C, D, E, EL = 2, 2, 2, 4, 32, 8
SA, SB, SC, SD = 1000, 300, 100, 40
NX = SA * (A - 1) + SB * (B - 1) + SC * (C - 1) + SD * (D - 1) + E - EL


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_unsized_runtime_low_refused(x_ptr, y_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                                             D: tl.constexpr, E: tl.constexpr, el, SA: tl.constexpr, SB: tl.constexpr,
                                             SC: tl.constexpr, SD: tl.constexpr):
        a = tl.arange(0, A)[:, None, None, None, None]
        b = tl.arange(0, B)[None, :, None, None, None]
        c = tl.arange(0, C)[None, None, :, None, None]
        d = tl.arange(0, D)[None, None, None, :, None]
        e = tl.arange(0, E)[None, None, None, None, :]
        v = tl.load(x_ptr + a * SA + b * SB + c * SC + d * SD + (e - el), mask=e >= el, other=0.0)
        tl.store(y_ptr + (((a * B + b) * C + c) * D + d) * E + e, v)

    return mask_tog_unsized_runtime_low_refused


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g)}


def reference(inputs):
    import torch
    x = inputs["x"]
    y = torch.zeros(A, B, C, D, E)
    for a in range(A):
        for b in range(B):
            for c in range(C):
                for d in range(D):
                    o = a * SA + b * SB + c * SC + d * SD
                    y[a, b, c, d, EL:] = x[o:o + E - EL]
    return {"y": y.reshape(-1)}


SPEC = KernelSpec(
    name="mask_tog_unsized_runtime_low_refused", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "A": "constexpr", "B": "constexpr", "C": "constexpr",
               "D": "constexpr", "E": "constexpr", "el": "i32", "SA": "constexpr", "SB": "constexpr",
               "SC": "constexpr", "SD": "constexpr"},
    constexprs={"A": A, "B": B, "C": C, "D": D, "E": E, "SA": SA, "SB": SB, "SC": SC, "SD": SD},
    args=[Arg("x", IN, "float32", (NX,)), Arg("y", OUT, "float32", (A * B * C * D * E,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("el", "int32_t")], "scalar_values": {"el": EL}},
    expect="refuse", expect_stop=Stop("tog", Reason.TOG_MASK_START),
    expect_why="a load waited in another block is not sized at run time, so its DMA would start at the "
               "unmasked column 0 while the clamp's low is a run-time scalar",
)
