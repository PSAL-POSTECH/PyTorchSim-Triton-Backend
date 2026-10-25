"""A [4] row statistic every lane needs for a [4, 1024] score tile: a gather whose DESTINATION is
eight lane-fulls deep, and every lane-full past the first holds the first again.

A FOLD NOW. The row statistics fold across the lanes, and every lane keeps the fold's result in its
own bank, so each lane copies the rows it needs from there: the pin is two folds and no replicate,
and no all-gather is left here.

WHAT IT PINS. The row max of decode attention's score (4 query rows x 1024 keys, the keys on the
lanes) is made in a [4] row and read where the [1024, 4] tile lives, 8 rows per lane. All-gather
passes carry one value per lane into every lane; `gather_rounds` used to run them once per
lane-full of the destination (`laps * rows` passes reading the same source depths `i % rows`),
so every pass past the first lap re-sent what the first had delivered. Now the passes stop at the
first lap and each lane copies its own first lap in-lane, a register group at a time
(`lower_xlu_moves._copy_first_lap`). The source depths a pass reads and the offsets it writes are
static and inside the bank, so those accesses carry no bounds check.

    before   the pass runs in an 8-trip loop (`affine.for 0 to 8`, one push and one pop under it);
             09-passes lower-xlu-moves holds 32 vector.transfer_write, 118 vector.transfer_read
             and no in_bounds beyond the 110 the earlier passes wrote
    after    one pass, no trip loop, then one read of the first lap and 7 writes of it deeper:
             39 vector.transfer_write, 119 vector.transfer_read; every row move's statically
             in-bank access is marked in_bounds (158)
    since    the panel staging (7e9dc4a, 7773ae1) reads the next panel among the inputs of the one
             in use: 155 vector.transfer_read and 194 in_bounds, the writes unchanged

torch equivalent: softmax(q @ k.T) @ v, one decode step of one KV head
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

G, D, S = 4, 128, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_all_gather_first_lap_kernel(q_ptr, k_ptr, v_ptr, o_ptr, G: tl.constexpr,
                                            D: tl.constexpr, S: tl.constexpr):
        rg = tl.arange(0, G)
        rd = tl.arange(0, D)
        t = tl.arange(0, S)
        q = tl.load(q_ptr + rg[:, None] * D + rd[None, :])
        kt = tl.load(k_ptr + t[None, :] * D + rd[:, None])
        v = tl.load(v_ptr + t[:, None] * D + rd[None, :])
        s = tl.dot(q, kt, out_dtype=tl.float32)
        p = tl.exp(s - tl.max(s, axis=1)[:, None])
        o = tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        tl.store(o_ptr + rg[:, None] * D + rd[None, :], o / tl.sum(p, axis=1)[:, None])

    return ops_xlu_all_gather_first_lap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(G * D, generator=g).to(torch.bfloat16),
            "k": torch.randn(S * D, generator=g).to(torch.bfloat16),
            "v": torch.randn(S * D, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    s = inputs["q"].float().reshape(G, D) @ inputs["k"].float().reshape(S, D).T
    p = (s - s.max(-1, keepdim=True).values).exp()
    o = p.to(torch.bfloat16).float() @ inputs["v"].float().reshape(S, D)
    return {"o": o / p.sum(-1, keepdim=True)}


SPEC = KernelSpec(
    name="ops_xlu_all_gather_first_lap_kernel", kernel=kernel,
    signature={"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "o_ptr": "*fp32",
               "G": "constexpr",
               "D": "constexpr", "S": "constexpr"},
    constexprs={"G": G, "D": D, "S": S},
    args=[Arg("q", IN, "bfloat16", (G * D,)), Arg("k", IN, "bfloat16", (S * D,)),
          Arg("v", IN, "bfloat16", (S * D,)), Arg("o", OUT, "float32", (G, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    # THE LAPS ARE THE CLAIM: the copy's read and its laps - 1 writes, and the dropped bounds checks.
    # V1 (p05_vectorize) leaves these three counts alone: they are the same with it reverted.
    expect_ir={"08-transformed.mlir": {"pytorchsim.distributed_reduction": 2,
                                       "kind = \"replicate\"": 0}},
)
