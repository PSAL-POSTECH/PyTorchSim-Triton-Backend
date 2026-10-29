"""Inductor's flattened MoE dispatch: x1 = x // C reads one index a row, fixed up (t < 0: t + V), and gathers
row t of the table: the index is read once a row and the gather is one row index per row ([rows, C], dim 0).
Breaks if the index chain is made per element and gathered one element at a time ([rows * C], 1-D)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, V, ROWS = 128, 16, 4
N = ROWS * C


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_row_index_rebuilt_on_rows_kernel(idx_ptr, tab_ptr, out_ptr, XBLOCK: tl.constexpr):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        x1 = xindex // 128
        x0 = xindex % 128
        t = tl.load(idx_ptr + x1)
        t = tl.where(t < 0, t + 16, t)
        tl.store(out_ptr + xindex, tl.load(tab_ptr + (x0 + 128 * t)))

    return gather_row_index_rebuilt_on_rows_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"idx": torch.tensor([3, -2, 15, -16], dtype=torch.int64),
            "tab": torch.randn(V * C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    t = inputs["idx"]
    t = torch.where(t < 0, t + V, t)
    return {"out": inputs["tab"].view(V, C)[t].reshape(-1)}


SPEC = KernelSpec(
    name="gather_row_index_rebuilt_on_rows_kernel",
    kernel=kernel,
    signature={"idx_ptr": "*i64", "tab_ptr": "*fp32", "out_ptr": "*fp32", "XBLOCK": "constexpr"},
    constexprs={"XBLOCK": N},
    args=[
        Arg("idx", IN, "int64", (ROWS,)),
        Arg("tab", IN, "float32", (V * C,)),
        Arg("out", OUT, "float32", (N,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
    expect_ir={"06-linalg.mlir": {"sizes = array<i64: 4, 128>": 1, "sizes = array<i64: 512>": 1}},
)
