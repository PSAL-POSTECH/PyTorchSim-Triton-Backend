"""Pins an in-loop read as an array runtime attribute: a paged-attention walk runs cdiv(seq_lens[pid], P)
trips, each loading the K page block_table[pid, j] names. The run records the table row once before the
loop, its extent the trip count; the TOG reads page j with togsim::element. Breaks if the loop is refused."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, MAXB, P, D, NPAGE = 4, 8, 16, 64, 32
LENS = [16, 70, 128, 1]
TABLE = [[(7 * s + 3 * j) % NPAGE for j in range(MAXB)] for s in range(S)]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_paged_block_table_tog(k_ptr, bt_ptr, sl_ptr, o_ptr, MAXB: tl.constexpr, P: tl.constexpr,
                                D: tl.constexpr):
        pid = tl.program_id(0)
        n = tl.load(sl_ptr + pid)
        rp = tl.arange(0, P)
        rd = tl.arange(0, D)
        acc = tl.zeros((D,), dtype=tl.float32)
        for j in range(0, tl.cdiv(n, P)):
            blk = tl.load(bt_ptr + pid * MAXB + j)
            k = tl.load(k_ptr + blk * (P * D) + rp[:, None] * D + rd[None, :])
            acc += tl.sum(k, 0)
        tl.store(o_ptr + pid * D + rd, acc)

    return rtattr_paged_block_table_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"k": torch.randn(NPAGE, P, D, generator=g),
            "bt": torch.tensor(TABLE, dtype=torch.int32), "sl": torch.tensor(LENS, dtype=torch.int32)}


def reference(inputs):
    import torch
    k, out = inputs["k"], []
    for s in range(S):
        acc = torch.zeros(D)
        for j in range(-(-LENS[s] // P)):
            acc += k[TABLE[s][j]].sum(0)
        out.append(acc)
    return {"o": torch.stack(out)}


SPEC = KernelSpec(
    name="rtattr_paged_block_table_tog", kernel=kernel,
    signature={"k_ptr": "*fp32", "bt_ptr": "*i32", "sl_ptr": "*i32", "o_ptr": "*fp32",
               "MAXB": "constexpr", "P": "constexpr", "D": "constexpr"},
    constexprs={"MAXB": MAXB, "P": P, "D": D},
    args=[Arg("k", IN, "float32", (NPAGE, P, D)), Arg("bt", IN, "int32", (S, MAXB)),
          Arg("sl", IN, "int32", (S,)), Arg("o", OUT, "float32", (S, D))],
    expect_ir={"09-custom.mlir": {"torchsim.runtime_attribute": 3}},
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
