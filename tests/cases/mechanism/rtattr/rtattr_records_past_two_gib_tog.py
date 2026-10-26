"""Pins a launch whose runtime attribute records the old hidden output could not hold: 512 programs of a
paged walk each record their block-table row, an array whose capacity is the 2^20-entry table its data
trip count may reach -- 4 GiB of rows preallocated, past what pk's heap gives a calloc. Breaks if the
wrapper allocates every program's capacity up front instead of streaming each record as it is made."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, MAXB, P, D, NPAGE = 512, 2048, 16, 64, 32
LENS = [(37 * s) % 49 for s in range(S)]


def _table(torch):
    s = torch.arange(S, dtype=torch.int32)[:, None]
    j = torch.arange(MAXB, dtype=torch.int32)[None, :]
    return (7 * s + 3 * j) % NPAGE


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_records_past_two_gib_tog(k_ptr, bt_ptr, sl_ptr, o_ptr, MAXB: tl.constexpr, P: tl.constexpr,
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

    return rtattr_records_past_two_gib_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"k": torch.randn(NPAGE, P, D, generator=g), "bt": _table(torch),
            "sl": torch.tensor(LENS, dtype=torch.int32)}


def reference(inputs):
    import torch
    k, bt = inputs["k"], inputs["bt"]
    pages = [[int(b) for b in bt[s, :-(-LENS[s] // P)]] for s in range(S)]
    return {"o": torch.stack([k[p].sum((0, 1)) if p else torch.zeros(D) for p in pages])}


SPEC = KernelSpec(
    name="rtattr_records_past_two_gib_tog", kernel=kernel,
    signature={"k_ptr": "*fp32", "bt_ptr": "*i32", "sl_ptr": "*i32", "o_ptr": "*fp32",
               "MAXB": "constexpr", "P": "constexpr", "D": "constexpr"},
    constexprs={"MAXB": MAXB, "P": P, "D": D},
    args=[Arg("k", IN, "float32", (NPAGE, P, D)), Arg("bt", IN, "int32", (S, MAXB)),
          Arg("sl", IN, "int32", (S,)), Arg("o", OUT, "float32", (S, D))],
    expect_ir={"09-custom.mlir": {"pytorchsim.runtime_attribute": 3}},
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
)
