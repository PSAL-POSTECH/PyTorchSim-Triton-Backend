"""A two-axis gather whose index is `ids + ids`: the sum of one index with itself at one scale is twice
that index, not the index once.
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
    def ptr_gather_index_added_to_itself(ids_ptr, table_ptr, out_ptr, S: tl.constexpr, B: tl.constexpr,
                                         D: tl.constexpr):
        rb = tl.arange(0, B)[:, None]
        rs = tl.arange(0, S)[None, :]
        rd = tl.arange(0, D)[None, None, :]
        ids = tl.load(ids_ptr + rb * S + rs)
        v = tl.load(table_ptr + (ids + ids)[:, :, None] * D + rd)
        tl.store(out_ptr + (rb * S + rs)[:, :, None] * D + rd, v)

    return ptr_gather_index_added_to_itself


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.randint(0, V // 2, (B * S,), generator=g).int(),
            "table": torch.randn(V, D, generator=g)}


def reference(inputs):
    return {"out": inputs["table"][2 * inputs["ids"].long()]}


SPEC = KernelSpec(
    name="ptr_gather_index_added_to_itself", kernel=kernel,
    signature={"ids_ptr": "*i32", "table_ptr": "*fp32", "out_ptr": "*fp32", "S": "constexpr",
               "B": "constexpr", "D": "constexpr"},
    constexprs={"S": S, "B": B, "D": D},
    args=[Arg("ids", IN, "int32", (B * S,)), Arg("table", IN, "float32", (V, D)),
          Arg("out", OUT, "float32", (B * S, D))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs,
)
