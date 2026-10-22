"""An embedding lookup whose index is a 2-D tile: table[ids[b, s], :].

The index varies on both tile axes b and s, the hidden axis d is read contiguously:

    table[d + D * ids[b, s]]

so the gather has two gathered axes and one row per (b, s). The tile lays b over the lanes, so b is
the descriptor's indirect axis (a piece per position of a lane axis would sit in other lanes) and
there is one row gather per s, each reading B rows of D contiguous elements. Before multi-axis
gathers this was an element-wise gather of B*S*D addresses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, S, D, V = 4, 8, 64, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_rows_by_index_tile(ids_ptr, table_ptr, out_ptr, S: tl.constexpr,
                                      D: tl.constexpr, B: tl.constexpr):
        rb = tl.arange(0, B)[:, None]
        rs = tl.arange(0, S)[None, :]
        ids = tl.load(ids_ptr + rb * S + rs)
        rd = tl.arange(0, D)[None, None, :]
        v = tl.load(table_ptr + ids[:, :, None] * D + rd)
        tl.store(out_ptr + (rb * S + rs)[:, :, None] * D + rd, v)

    return ptr_gather_rows_by_index_tile


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.randint(0, V, (B * S,), generator=g, dtype=torch.int32),
            "table": torch.randn(V * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    table = inputs["table"].view(V, D)
    return {"out": table[inputs["ids"].long()].reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_rows_by_index_tile", kernel=kernel,
    signature={"ids_ptr": "*i32", "table_ptr": "*fp32", "out_ptr": "*fp32",
               "S": "constexpr", "D": "constexpr", "B": "constexpr"},
    constexprs={"S": S, "D": D, "B": B},
    args=[Arg("ids", IN, "int32", (B * S,)), Arg("table", IN, "float32", (V * D,)),
          Arg("out", OUT, "float32", (B * S * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 1, "tts.gather": 0}},
)
