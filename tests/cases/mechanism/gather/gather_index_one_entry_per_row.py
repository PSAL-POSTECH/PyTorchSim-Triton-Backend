"""The gather index is one entry per gathered row, not one per tile element."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 2048
BM = 16
BN = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_index_one_entry_per_row_kernel(x_ptr, i_ptr, o_ptr,
                                              BM: tl.constexpr,
                                              BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + rm)
        a = tl.load(x_ptr + i[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], a)

    return gather_index_one_entry_per_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, NX // BN, (BM,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    x, i = inputs["x"].view(-1, BN), inputs["i"].long()
    return {"o": x[i].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="gather_index_one_entry_per_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # FIVE AND NOT EIGHT since 2026-09-06: the index tile used to be built in
    # one buffer and snapshotted into the gather's own, and bufferize's copy
    # named the type twice more. bufferize._write_where_the_copy_lands lands the
    # build in the gather's buffer, so the count doubles as that pin.
    # TWO SPELLINGS OF EACH TYPE: the bank view puts the lane axis in the type,
    # so the index is `memref<1x16xi32>` there and a replica would arrive as
    # `memref<1x8x16xi32>`. Counted together, both paths say five and zero.
    expect_ir={"08-passes/06-pytorchsim-bufferize.mlir": {
        ("memref<16xi32>", "memref<1x16xi32>"): 2,
        ("memref<16x8xi32>", "memref<1x8x16xi32>"): 0,
        "linalg.broadcast": 0}},
)
