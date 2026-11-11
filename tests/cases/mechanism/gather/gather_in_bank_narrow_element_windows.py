"""An in-bank gather of i8 by an i32 index past one register group: [128, 256] is one row of 256 a lane.
_gather_by_groups compares the index against its windows (64..256) at the i32, not the element's i8,
where 128 reads as -128 and 256 as 0 and the wrong chunk is picked -- max abs diff 120 if it goes back."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

L, D = 128, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_in_bank_narrow_element_windows_kernel(x_ptr, i_ptr, y_ptr, L: tl.constexpr,
                                                     D: tl.constexpr):
        at = tl.arange(0, L)[:, None] * D + tl.arange(0, D)[None, :]
        x = (tl.load(x_ptr + at) * 3).to(tl.int8)
        idx = tl.load(i_ptr + at)
        tl.store(y_ptr + at, tl.gather(x, idx, 1))

    return gather_in_bank_narrow_element_windows_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(0, 41, (L, D), generator=g, dtype=torch.int32),
            "i": torch.randint(0, D, (L, D), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    return {"y": (inputs["x"] * 3).to(torch.int8).gather(1, inputs["i"].long())}


SPEC = KernelSpec(
    name="gather_in_bank_narrow_element_windows_kernel", kernel=kernel,
    signature={"x_ptr": "*i32", "i_ptr": "*i32", "y_ptr": "*i8", "L": "constexpr", "D": "constexpr"},
    constexprs={"L": L, "D": D},
    args=[Arg("x", IN, "int32", (L, D)), Arg("i", IN, "int32", (L, D)),
          Arg("y", OUT, "int8", (L, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
