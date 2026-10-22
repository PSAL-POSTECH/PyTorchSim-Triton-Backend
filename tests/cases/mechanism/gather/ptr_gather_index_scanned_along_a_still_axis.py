"""A gather whose index is a cumsum along an axis its input holds still on: the scan's result varies on
that axis, so it is gathered there. An op whose operands and result share a shape is not elementwise for
that alone (tt.scan, tt.atomic_rmw and tt.atomic_cas carry that trait too).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, S, D, V = 4, 4, 128, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_index_scanned_along_a_still_axis(ids_ptr, table_ptr, out_ptr, S: tl.constexpr,
                                                    B: tl.constexpr, D: tl.constexpr):
        rb = tl.arange(0, B)[:, None]
        rs = tl.arange(0, S)[None, :]
        rd = tl.arange(0, D)[None, None, :]
        a = tl.load(ids_ptr + rb + rs * 0)
        ids = tl.cumsum(a, axis=1)
        v = tl.load(table_ptr + ids[:, :, None] * D + rd)
        tl.store(out_ptr + (rb * S + rs)[:, :, None] * D + rd, v)

    return ptr_gather_index_scanned_along_a_still_axis


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.randint(0, V // S, (B,), generator=g).int(),
            "table": torch.randn(V, D, generator=g)}


def reference(inputs):
    import torch
    rows = inputs["ids"].long()[:, None] * torch.arange(1, S + 1)[None, :]
    return {"out": inputs["table"][rows.reshape(-1)]}


SPEC = KernelSpec(
    name="ptr_gather_index_scanned_along_a_still_axis", kernel=kernel,
    signature={"ids_ptr": "*i32", "table_ptr": "*fp32", "out_ptr": "*fp32", "S": "constexpr",
               "B": "constexpr", "D": "constexpr"},
    constexprs={"S": S, "B": B, "D": D},
    args=[Arg("ids", IN, "int32", (B,)), Arg("table", IN, "float32", (V, D)),
          Arg("out", OUT, "float32", (B * S, D))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs,
)
