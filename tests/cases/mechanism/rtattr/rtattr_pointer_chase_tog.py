"""Pins pointer chasing as a refusal by name: a linked-list walk loads the row at `cur`, then cur = next[cur],
so where trip t+1 reads rests on what trip t read and no record taken before the loop can state it. Breaks if
the address is recorded as an array or read as 0, or the kernel stops anywhere else."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

S, L, D, NROW = 4, 4, 64, 16
HEAD = [3, 0, 9, 12]
NEXT = [(5 * r + 1) % NROW for r in range(NROW)]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_pointer_chase_tog(x_ptr, head_ptr, next_ptr, o_ptr, L: tl.constexpr, D: tl.constexpr):
        pid = tl.program_id(0)
        rd = tl.arange(0, D)
        cur = tl.load(head_ptr + pid)
        for t in range(0, L):
            tl.store(o_ptr + (pid * L + t) * D + rd, tl.load(x_ptr + cur * D + rd))
            cur = tl.load(next_ptr + cur)

    return rtattr_pointer_chase_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NROW, D, generator=g), "head": torch.tensor(HEAD, dtype=torch.int32),
            "next": torch.tensor(NEXT, dtype=torch.int32)}


def reference(inputs):
    import torch
    rows = []
    for s in range(S):
        cur = HEAD[s]
        for _ in range(L):
            rows.append(inputs["x"][cur])
            cur = NEXT[cur]
    return {"o": torch.stack(rows)}


SPEC = KernelSpec(
    name="rtattr_pointer_chase_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "head_ptr": "*i32", "next_ptr": "*i32", "o_ptr": "*fp32",
               "L": "constexpr", "D": "constexpr"},
    constexprs={"L": L, "D": D},
    args=[Arg("x", IN, "float32", (NROW, D)), Arg("head", IN, "int32", (S,)),
          Arg("next", IN, "int32", (NROW,)), Arg("o", OUT, "float32", (S * L, D))],
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect="refuse", expect_stop=Stop("lower", Reason.RUNTIME_ATTRIBUTE),
    expect_why="the address of next[cur] rests on what the previous trip read (pointer chasing): no "
               "runtime attribute recorded before the loop states it",
)
