"""The same one-entry index, with the tile banked on the OTHER axis."""
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
    def gather_index_survives_a_foreign_lane_axis_kernel(
            x_ptr, i_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + rm)
        a = tl.load(x_ptr + i[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rn, tl.sum(a, axis=0))

    return gather_index_survives_a_foreign_lane_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, NX // BN, (BM,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    x, i = inputs["x"].view(-1, BN), inputs["i"].long()
    return {"o": x[i].sum(0).contiguous()}


SPEC = KernelSpec(
    name="gather_index_survives_a_foreign_lane_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # FIVE AND NOT EIGHT since 2026-09-06 -- see gather_index_one_entry_per_row:
    # the snapshot copy is gone, and it named the type twice.
    # TWO SPELLINGS OF EACH TYPE: the bank view puts the lane axis in the type,
    # so the index is `memref<1x16xi32>` there and a replica would arrive as
    # `memref<1x8x16xi32>`. `indirect_dim` is counted as a token, not a value --
    # that the descriptor names an axis at all is the claim.
    expect_ir={"08-passes/06-pytorchsim-bufferize.mlir": {
        ("memref<16xi32>", "memref<1x16xi32>"): 2,
        ("memref<16x8xi32>", "memref<1x8x16xi32>"): 0,
        "indirect_dim": 1}},
)
