"""A 5-d gather legalize-transfers peels into a loop (wait after it), masked off its gather axis by a runtime
`e >= lo`: the DMA stays unsized, so it would start lo columns before the live ones. Refused at tog; breaks if
the TOG moves the unmasked tile again (it passed on 11930f50 as gather_tog_off_axis_mask_left_unsized)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

V, A, B, C, D, E, EL = 6, 2, 2, 2, 4, 32, 8
SA, SB, SC, SD = 1000, 300, 100, 40
NX = SA * (V - 1) + SB * (B - 1) + SC * (C - 1) + SD * (D - 1) + E - EL


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_tog_unsized_runtime_low_refused(x_ptr, idx_ptr, y_ptr, lo, A: tl.constexpr, B: tl.constexpr,
                                              C: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                                              SA: tl.constexpr, SB: tl.constexpr,
                                              SC: tl.constexpr, SD: tl.constexpr):
        r = tl.load(idx_ptr + tl.arange(0, A))[:, None, None, None, None]
        a = tl.arange(0, A)[:, None, None, None, None]
        b = tl.arange(0, B)[None, :, None, None, None]
        c = tl.arange(0, C)[None, None, :, None, None]
        d = tl.arange(0, D)[None, None, None, :, None]
        e = tl.arange(0, E)[None, None, None, None, :]
        v = tl.load(x_ptr + r * SA + b * SB + c * SC + d * SD + (e - lo), mask=e >= lo, other=0.0)
        tl.store(y_ptr + (((a * B + b) * C + c) * D + d) * E + e, v)

    return gather_tog_unsized_runtime_low_refused


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g), "idx": torch.tensor([5, 2], dtype=torch.int32)}


def reference(inputs):
    import torch
    x, idx = inputs["x"], inputs["idx"].long()
    y = torch.zeros(A, B, C, D, E)
    for a in range(A):
        for b in range(B):
            for c in range(C):
                for d in range(D):
                    base = idx[a] * SA + b * SB + c * SC + d * SD - EL
                    y[a, b, c, d, EL:] = x[base + EL:base + E]
    return {"y": y.reshape(-1)}


SPEC = KernelSpec(
    name="gather_tog_unsized_runtime_low_refused", kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "y_ptr": "*fp32", "lo": "i32", "A": "constexpr", "B": "constexpr",
               "C": "constexpr", "D": "constexpr", "E": "constexpr", "SA": "constexpr", "SB": "constexpr",
               "SC": "constexpr", "SD": "constexpr"},
    constexprs={"A": A, "B": B, "C": C, "D": D, "E": E, "SA": SA, "SB": SB, "SC": SC, "SD": SD},
    args=[Arg("x", IN, "float32", (NX,)), Arg("idx", IN, "int32", (A,)), Arg("y", OUT, "float32", (A * B * C * D * E,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"lo": EL}},
    expect="refuse", expect_stop=Stop("tog", Reason.TOG_MASK_START),
    expect_why="a gather waited in another block is not sized at run time, so its DMA would start at the "
               "unmasked column 0 while the clamp's low is a run-time scalar",
)
