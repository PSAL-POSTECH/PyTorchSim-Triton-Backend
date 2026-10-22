"""A two-axis gather of one-element rows, masked on the gathered axis that is not the descriptor's
indirect one: the masked positions read `other`, the rest the element their index names.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, S, N, V = 8, 128, 100, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_one_element_rows_masked_off_the_indirect_axis(ids_ptr, table_ptr, out_ptr,
                                                                  S: tl.constexpr, B: tl.constexpr,
                                                                  N: tl.constexpr):
        rb = tl.arange(0, B)[:, None]
        rs = tl.arange(0, S)[None, :]
        ids = tl.load(ids_ptr + rb * S + rs)
        v = tl.load(table_ptr + ids * 3, mask=(rs < N), other=0.0)
        tl.store(out_ptr + rb * S + rs, v)

    return ptr_gather_one_element_rows_masked_off_the_indirect_axis


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.randint(0, V // 3, (B * S,), generator=g).int(),
            "table": torch.randn(V, generator=g)}


def reference(inputs):
    import torch
    v = inputs["table"][3 * inputs["ids"].long()].view(B, S)
    v[:, N:] = 0.0
    return {"out": v.reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_one_element_rows_masked_off_the_indirect_axis", kernel=kernel,
    signature={"ids_ptr": "*i32", "table_ptr": "*fp32", "out_ptr": "*fp32", "S": "constexpr",
               "B": "constexpr", "N": "constexpr"},
    constexprs={"S": S, "B": B, "N": N},
    args=[Arg("ids", IN, "int32", (B * S,)), Arg("table", IN, "float32", (V,)),
          Arg("out", OUT, "float32", (B * S,))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs,
)
