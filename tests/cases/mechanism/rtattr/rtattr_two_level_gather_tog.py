"""Pins two-level indirection in one trip as a refusal by name: page = remap[bt[pid, j]], so the address of
the second read rests on the first, read in the same trip, and one array taken before the loop cannot state
it. Breaks if it is recorded, read as 0, or the kernel stops anywhere else."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

S, NB, P, D, NPAGE = 4, 4, 16, 64, 32
TABLE = [[(7 * s + 3 * j) % NPAGE for j in range(NB)] for s in range(S)]
REMAP = [(11 * p + 5) % NPAGE for p in range(NPAGE)]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_two_level_gather_tog(k_ptr, bt_ptr, rm_ptr, o_ptr, NB: tl.constexpr, P: tl.constexpr,
                                    D: tl.constexpr):
        pid = tl.program_id(0)
        rp = tl.arange(0, P)
        rd = tl.arange(0, D)
        acc = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, NB):
            blk = tl.load(bt_ptr + pid * NB + j)
            page = tl.load(rm_ptr + blk)
            k = tl.load(k_ptr + page * (P * D) + rp[:, None] * D + rd[None, :])
            acc += tl.sum(k, 0)
        tl.store(o_ptr + pid * D + rd, acc)

    return rtattr_two_level_gather_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"k": torch.randn(NPAGE, P, D, generator=g), "bt": torch.tensor(TABLE, dtype=torch.int32),
            "rm": torch.tensor(REMAP, dtype=torch.int32)}


def reference(inputs):
    import torch
    k = inputs["k"]
    return {"o": torch.stack([sum(k[REMAP[TABLE[s][j]]].sum(0) for j in range(NB)) for s in range(S)])}


SPEC = KernelSpec(
    name="rtattr_two_level_gather_tog", kernel=kernel,
    signature={"k_ptr": "*fp32", "bt_ptr": "*i32", "rm_ptr": "*i32", "o_ptr": "*fp32",
               "NB": "constexpr", "P": "constexpr", "D": "constexpr"},
    constexprs={"NB": NB, "P": P, "D": D},
    args=[Arg("k", IN, "float32", (NPAGE, P, D)), Arg("bt", IN, "int32", (S, NB)),
          Arg("rm", IN, "int32", (NPAGE,)), Arg("o", OUT, "float32", (S, D))],
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect="refuse", expect_stop=Stop("lower", Reason.RUNTIME_ATTRIBUTE),
    expect_why="remap[bt[j]]: the second read's address rests on the first, read in the same trip "
               "(two-level indirection), which no array record states",
)
