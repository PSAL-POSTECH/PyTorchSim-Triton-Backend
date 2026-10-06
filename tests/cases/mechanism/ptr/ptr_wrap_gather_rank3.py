"""Three axes, one of each: a wrap, a gather, and a plain affine walk.

    idx = (r % M)*S*T + j[s]*T + t

THIS IS WHERE HOLDING THE WRAP PAYS, and rank 2 could not show it. The gathered
axis gives up its dram stride and the wrapped one gives up its own, but `t`
keeps a real one -- `dram_stride = [0, 0, 1]` -- so the descriptor still WALKS
the innermost axis. At rank 2 there is no third axis left to walk, so the
descriptor comes out the same either way and only the mask carriers differ.

It was refused until 2026-08-31: a `hasModulo() && rank > 2` guard in
visitOperandExpandDims, whose only job was to keep a 2D-only memref chunk split
unreached. That lowering is deleted; ours reads the access directly (rule 21).

BR=8 over M=3, so the wrap fires on rows 3..7.

A wrap, a gather and an affine axis on one rank-3 pointer -- the affine one keeps its dram stride, which rank 2 cannot show.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BR, BS, BT, M = 8, 4, 8, 3
ASIZE = BR * BS * BT * 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_gather_rank3(a_ptr, j_ptr, o_ptr, BR: tl.constexpr,
                          BS: tl.constexpr, BT: tl.constexpr,
                          M: tl.constexpr):
        r = tl.arange(0, BR) % M
        s = tl.arange(0, BS)
        t = tl.arange(0, BT)
        j = tl.load(j_ptr + s)
        idx = (r[:, None, None] * (BS * BT) + j[None, :, None] * BT
               + t[None, None, :])
        v = tl.load(a_ptr + idx)
        out = (tl.arange(0, BR)[:, None, None] * (BS * BT)
               + s[None, :, None] * BT + t[None, None, :])
        tl.store(o_ptr + out, v)

    return wrap_gather_rank3


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32),
            "j": torch.randint(0, BS, (BS,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    a, j = inputs["a"], inputs["j"].long()
    r = torch.arange(BR) % M
    t = torch.arange(BT)
    idx = (r[:, None, None] * (BS * BT) + j[None, :, None] * BT
           + t[None, None, :])
    return {"o": a[idx].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_gather_rank3", kernel=kernel,
    signature={"a_ptr": "*fp32", "j_ptr": "*i32", "o_ptr": "*fp32",
               "BR": "constexpr", "BS": "constexpr", "BT": "constexpr",
               "M": "constexpr"},
    constexprs={"BR": BR, "BS": BS, "BT": BT, "M": M},
    args=[Arg("a", IN, "float32", (ASIZE,)),
          Arg("j", IN, "int32", (BS,)),
          Arg("o", OUT, "float32", (BR * BS * BT,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"03-ttshared.mlir": {"tts.gather": 0,
                                    "tts.make_gather_scatter_tptr": 1}},
)
