"""Pins a float read turned into a DMA address as a refusal by name: the row each program loads is
int(f[pid]), and a runtime attribute holds int64 integers alone. Breaks if the address is recorded,
read as 0, or the kernel stops anywhere else."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

S, D, NROW = 4, 64, 16
ROWS = [3.0, 0.0, 9.0, 12.0]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_float_offset_tog(x_ptr, f_ptr, o_ptr, D: tl.constexpr):
        pid = tl.program_id(0)
        rd = tl.arange(0, D)
        row = tl.load(f_ptr + pid).to(tl.int32)
        tl.store(o_ptr + pid * D + rd, tl.load(x_ptr + row * D + rd))

    return rtattr_float_offset_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NROW, D, generator=g), "f": torch.tensor(ROWS, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"][[int(r) for r in ROWS]]}


SPEC = KernelSpec(
    name="rtattr_float_offset_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*fp32", "o_ptr": "*fp32", "D": "constexpr"},
    constexprs={"D": D},
    args=[Arg("x", IN, "float32", (NROW, D)), Arg("f", IN, "float32", (S,)), Arg("o", OUT, "float32", (S, D))],
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect="refuse", expect_stop=Stop("lower", Reason.RUNTIME_ATTRIBUTE),
    expect_why="the row a program loads is a float read turned into an integer; a runtime attribute "
               "holds integers alone",
)
