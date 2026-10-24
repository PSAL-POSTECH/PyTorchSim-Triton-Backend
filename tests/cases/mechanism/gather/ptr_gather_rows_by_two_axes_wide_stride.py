"""MoE combine's row gather by (t, r) with rows 70000 elements apart, a row stride past the descriptor's
uint16 offset_stride field. Breaks if a stride that does not fit stops the compile instead of reading
per element.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

TOK, H, K, E = 4, 256, 4, 8
RS = 70000
ROWS = TOK * K
XBLOCK = TOK * H


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_rows_by_two_axes_wide_stride(ids_ptr, pos_ptr, start_ptr, y_ptr, w_ptr, out_ptr,
                                    H: tl.constexpr, K: tl.constexpr, XBLOCK: tl.constexpr, RS: tl.constexpr):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        r = tl.arange(0, K)[None, :]
        t = xindex // H
        h = xindex % H
        e = tl.load(ids_ptr + (r + K * t))
        row = tl.load(start_ptr + e) + tl.load(pos_ptr + (r + K * t))
        v = tl.load(y_ptr + (h + RS * row)).to(tl.float32)
        w = tl.load(w_ptr + (r + K * t)).to(tl.float32)
        tl.store(out_ptr + xindex, tl.sum(v * w, 1)[:, None].to(tl.bfloat16))

    return ptr_gather_rows_by_two_axes_wide_stride


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
            "y": (torch.randn(ROWS * RS, generator=g) * 0.1).to(torch.bfloat16),
            "w": (torch.rand(TOK * K, generator=g)).to(torch.bfloat16)}


def reference(inputs):
    import torch
    ids, pos, start = inputs["ids"].long(), inputs["pos"].long(), inputs["start"].long()
    rows = start[ids] + pos
    v = inputs["y"].float().view(ROWS, RS)[rows, :H].reshape(TOK, K, H)
    w = inputs["w"].float().view(TOK, K, 1)
    return {"out": (v * w).sum(1).reshape(-1).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="ptr_gather_rows_by_two_axes_wide_stride", kernel=kernel,
    signature={"ids_ptr": "*i32", "pos_ptr": "*i32", "start_ptr": "*i32", "y_ptr": "*bf16", "w_ptr": "*bf16",
               "out_ptr": "*bf16", "H": "constexpr", "K": "constexpr", "XBLOCK": "constexpr",
               "RS": "constexpr"},
    constexprs={"H": H, "K": K, "XBLOCK": XBLOCK, "RS": RS},
    args=[Arg("ids", IN, "int32", (TOK * K,)), Arg("pos", IN, "int32", (TOK * K,)),
          Arg("start", IN, "int32", (E,)), Arg("y", IN, "bfloat16", (ROWS * RS,)),
          Arg("w", IN, "bfloat16", (TOK * K,)), Arg("out", OUT, "bfloat16", (TOK * H,))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
