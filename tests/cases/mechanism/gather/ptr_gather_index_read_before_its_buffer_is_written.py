"""A gather whose index was loaded before the kernel overwrote the index buffer: the gather reads the rows
the loaded values name, not what the buffer holds after the store. Reading the index again at one position
of the axes it holds still on must read it where it was first read.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT, INOUT

B, S, D, V = 4, 4, 128, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_index_read_before_its_buffer_is_written(ids_ptr, table_ptr, out_ptr, S: tl.constexpr,
                                                           B: tl.constexpr, D: tl.constexpr):
        rb = tl.arange(0, B)[:, None, None]
        rs = tl.arange(0, S)[None, :, None]
        rd = tl.arange(0, D)[None, None, :]
        a = tl.load(ids_ptr + rb * S + rs + rd * 0)
        tl.store(ids_ptr + tl.arange(0, B * S), tl.full([B * S], 0, tl.int32))
        v = tl.load(table_ptr + a * D + rd)
        tl.store(out_ptr + (rb * S + rs) * D + rd, v)

    return ptr_gather_index_read_before_its_buffer_is_written


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.randint(1, V, (B * S,), generator=g).int(),
            "table": torch.randn(V, D, generator=g)}


def reference(inputs):
    import torch
    return {"ids": torch.zeros(B * S, dtype=torch.int32),
            "out": inputs["table"][inputs["ids"].long()]}


SPEC = KernelSpec(
    name="ptr_gather_index_read_before_its_buffer_is_written", kernel=kernel,
    signature={"ids_ptr": "*i32", "table_ptr": "*fp32", "out_ptr": "*fp32", "S": "constexpr",
               "B": "constexpr", "D": "constexpr"},
    constexprs={"S": S, "B": B, "D": D},
    args=[Arg("ids", INOUT, "int32", (B * S,)), Arg("table", IN, "float32", (V, D)),
          Arg("out", OUT, "float32", (B * S, D))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs,
)
