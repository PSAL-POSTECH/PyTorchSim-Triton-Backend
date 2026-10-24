"""MoE combine as Inductor writes it: 8192 rows (64 a lane), each a weighted sum of K=8 gathered rows.
The short reduced axis is held outermost in a lane's registers, so the sum adds whole vectors of rows.
Breaks if each row is reduced on its own again (a vector.multi_reduction per row).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

TOK, H, K, E = 32, 256, 8, 16
ROWS = TOK * K
XBLOCK = TOK * H


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_short_axis_across_rows_kernel(ids_ptr, start_ptr, pos_ptr, y_ptr, w_ptr, out_ptr,
                                             H: tl.constexpr, K: tl.constexpr, XBLOCK: tl.constexpr):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        r = tl.arange(0, K)[None, :]
        t = xindex // H
        h = xindex % H
        e = tl.load(ids_ptr + (r + K * t))
        row = tl.load(start_ptr + e) + tl.load(pos_ptr + (r + K * t))
        v = tl.load(y_ptr + (h + H * row)).to(tl.float32)
        w = tl.load(w_ptr + (r + K * t)).to(tl.float32)
        tl.store(out_ptr + xindex, tl.sum(v * w, 1)[:, None].to(tl.bfloat16))

    return reduce_short_axis_across_rows_kernel


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
    return {"ids": ids.reshape(-1), "start": start, "pos": pos.reshape(-1),
            "y": (torch.randn(ROWS, H, generator=g) * 0.1).to(torch.bfloat16),
            "w": torch.rand(TOK * K, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    ids, pos, start = inputs["ids"].long(), inputs["pos"].long(), inputs["start"].long()
    v = inputs["y"].float()[start[ids] + pos].view(TOK, K, H)
    w = inputs["w"].float().view(TOK, K, 1)
    return {"out": (v * w).sum(1).reshape(-1).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="reduce_short_axis_across_rows_kernel", kernel=kernel,
    signature={"ids_ptr": "*i32", "start_ptr": "*i32", "pos_ptr": "*i32", "y_ptr": "*bf16", "w_ptr": "*bf16",
               "out_ptr": "*bf16", "H": "constexpr", "K": "constexpr", "XBLOCK": "constexpr"},
    constexprs={"H": H, "K": K, "XBLOCK": XBLOCK},
    args=[Arg("ids", IN, "int32", (TOK * K,)), Arg("start", IN, "int32", (E,)),
          Arg("pos", IN, "int32", (TOK * K,)),
          Arg("y", IN, "bfloat16", (ROWS, H)), Arg("w", IN, "bfloat16", (TOK * K,)),
          Arg("out", OUT, "bfloat16", (TOK * H,))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"08-transformed.mlir": {"vector.multi_reduction": 0}},
)
