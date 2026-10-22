"""ptr_gather_rows_by_two_axes with the reads masked over the top-k axis, as Inductor writes them
(`r < K`, `other=0`): the loads that build the row index are then plain tile loads, read at their full
width, and the index is computed on the flat (t*H + h, r) tile. The index of the y gather must still be
rebuilt at one position of h -- one row per (t, r) -- so no lane carries a copy of it and no index row
is permuted out of a full tile.



The tile is a flat token x hidden axis (x = t*H + h) and the top-k axis r. Each (t, r) names one row of y
through two lookups (ids -> a per-expert table), and the row is read contiguously along h:

    y[h + H * (start[ids[t, r]] + pos[t, r])]

so the index is one value per (t, r) and the gather reads TOK*K rows of H elements. The table lookups are
one value per (t, r) too, read the same at every h.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

TOK, H, K, E = 4, 256, 4, 8
ROWS = TOK * K
XBLOCK = TOK * H


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_rows_by_two_axes_masked(ids_ptr, pos_ptr, start_ptr, y_ptr, w_ptr, out_ptr,
                                    H: tl.constexpr, K: tl.constexpr, XBLOCK: tl.constexpr):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        r = tl.arange(0, K)[None, :]
        t = xindex // H
        h = xindex % H
        rm = r < K
        e = tl.load(ids_ptr + (r + K * t), rm, other=0)
        row = tl.load(start_ptr + e, rm) + tl.load(pos_ptr + (r + K * t), rm, other=0)
        v = tl.load(y_ptr + (h + H * row), rm).to(tl.float32)
        w = tl.load(w_ptr + (r + K * t), rm, other=0.0).to(tl.float32)
        tl.store(out_ptr + xindex, tl.sum(v * w, 1)[:, None].to(tl.bfloat16))

    return ptr_gather_rows_by_two_axes_masked


def _routing():
    import torch
    ids = (torch.arange(TOK)[:, None] * K + torch.arange(K)[None, :]) % E
    sizes = torch.bincount(ids.reshape(-1), minlength=E)
    start = torch.cumsum(sizes, 0) - sizes
    flat = ids.reshape(-1)
    oh = (flat[:, None] == torch.arange(E)[None, :]).long()
    pos = ((torch.cumsum(oh, 0) * oh).sum(1) - 1).reshape(TOK, K)
    return ids.int(), pos.int(), start.int()


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids, pos, start = _routing()
    return {"ids": ids.reshape(-1), "pos": pos.reshape(-1), "start": start,
            "y": (torch.randn(ROWS, H, generator=g) * 0.1).to(torch.bfloat16),
            "w": (torch.rand(TOK * K, generator=g)).to(torch.bfloat16)}


def reference(inputs):
    import torch
    ids, pos, start = inputs["ids"].long(), inputs["pos"].long(), inputs["start"].long()
    rows = start[ids] + pos
    v = inputs["y"].float()[rows].view(TOK, K, H)
    w = inputs["w"].float().view(TOK, K, 1)
    return {"out": (v * w).sum(1).reshape(-1).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ptr_gather_rows_by_two_axes_masked", kernel=kernel,
    signature={"ids_ptr": "*i32", "pos_ptr": "*i32", "start_ptr": "*i32", "y_ptr": "*bf16", "w_ptr": "*bf16",
               "out_ptr": "*bf16", "H": "constexpr", "K": "constexpr", "XBLOCK": "constexpr"},
    constexprs={"H": H, "K": K, "XBLOCK": XBLOCK},
    args=[Arg("ids", IN, "int32", (TOK * K,)), Arg("pos", IN, "int32", (TOK * K,)),
          Arg("start", IN, "int32", (E,)), Arg("y", IN, "bfloat16", (ROWS, H)),
          Arg("w", IN, "bfloat16", (TOK * K,)), Arg("out", OUT, "bfloat16", (TOK * H,))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    # Two gathers over (t, r), not one per element, and the index of the y gather made at one
    # position of h: no copy between lanes.
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 2, "tts.gather": 0},
               "08-transformed.mlir": {"torchsim.crosslane_move": 0}},
)
