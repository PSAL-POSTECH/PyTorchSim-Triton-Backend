"""An in-bank gather of i8 by an i8 index over a 512 row: past one register group, the windows are compared
at the wider of the two, i8, and 256 folds to 0 -- window [256, 512) also claims every index under 128.
Refused rather than wrong; breaks if the windows are compared at i8 again or the refusal goes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Stop

L, D = 128, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_in_bank_narrow_index_windows_refused_kernel(x_ptr, i_ptr, y_ptr, L: tl.constexpr,
                                                           D: tl.constexpr):
        at = tl.arange(0, L)[:, None] * D + tl.arange(0, D)[None, :]
        x = (tl.load(x_ptr + at) * 3).to(tl.int8)
        idx = tl.load(i_ptr + at)
        tl.store(y_ptr + at, tl.gather(x, idx, 1))

    return gather_in_bank_narrow_index_windows_refused_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(0, 41, (L, D), generator=g, dtype=torch.int32),
            "i": torch.randint(0, 128, (L, D), generator=g, dtype=torch.int8)}


def reference(inputs):
    import torch
    return {"y": (inputs["x"] * 3).to(torch.int8).gather(1, inputs["i"].long())}


SPEC = KernelSpec(
    name="gather_in_bank_narrow_index_windows_refused_kernel", kernel=kernel,
    signature={"x_ptr": "*i32", "i_ptr": "*i8", "y_ptr": "*i8", "L": "constexpr", "D": "constexpr"},
    constexprs={"L": L, "D": D},
    args=[Arg("x", IN, "int32", (L, D)), Arg("i", IN, "int8", (L, D)),
          Arg("y", OUT, "int8", (L, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse", expect_stop=Stop("lower", "builtins.NotImplementedError"),
    expect_why="an i8 index windowed at i8 over a 512 row cannot name the windows past 127; refused rather than wrong",
)
