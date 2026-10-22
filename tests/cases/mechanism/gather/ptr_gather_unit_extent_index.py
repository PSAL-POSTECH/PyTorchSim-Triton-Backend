"""A gather whose index tile has extent 1 on every axis, the BLOCK_M = 1 shape of a decode embedding
lookup: the one row it names must be read, not row 0. A [1] or [1, 1] index has no axis it varies on, and
it must keep its index rather than be read as a structured pointer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, V, D = 8, 16, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_unit_extent_index(ids_ptr, table_ptr, out_ptr, D: tl.constexpr):
        r = tl.arange(0, 1)
        ids = tl.load(ids_ptr + 3 + r)[:, None]
        rd = tl.arange(0, D)[None, :]
        v = tl.load(table_ptr + ids * D + rd)
        tl.store(out_ptr + r[:, None] * D + rd, v)

    return ptr_gather_unit_extent_index


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.randperm(V, generator=g)[:N].int(),
            "table": torch.randn(V, D, generator=g)}


def reference(inputs):
    return {"out": inputs["table"][inputs["ids"][3].long()]}


SPEC = KernelSpec(
    name="ptr_gather_unit_extent_index", kernel=kernel,
    signature={"ids_ptr": "*i32", "table_ptr": "*fp32", "out_ptr": "*fp32", "D": "constexpr"},
    constexprs={"D": D},
    args=[Arg("ids", IN, "int32", (N,)), Arg("table", IN, "float32", (V, D)),
          Arg("out", OUT, "float32", (D,))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs,
)
