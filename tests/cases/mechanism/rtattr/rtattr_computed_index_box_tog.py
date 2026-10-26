"""Pins the snapshot of a computed-index gather whose index tile is wider than the moved tile on an inner
axis: [16, 24] moved, a [16, 32] index, so the snapshot is the [16, 24] box of it, not its first 384
entries. Breaks if lower refuses it, or the snapshot's DMA moves a run of 384 (Spike faults past the tile)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R, XBLOCK, RBLOCK, M = 12, 24, 16, 32, 40


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_computed_index_box_tog(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)[:, None]
        r0 = tl.arange(0, R0_BLOCK)[None, :]
        m = (xindex < 12) & (r0 < 24)
        tmp0 = tl.load(in_ptr0 + (xindex * 24 + r0) % 40, m)
        tl.store(out_ptr0 + (r0 + 24 * xindex), tmp0, m)

    return rtattr_computed_index_box_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"b": torch.randn(M, generator=g)}


def reference(inputs):
    import torch
    return {"o": inputs["b"][torch.arange(X * R) % M]}


SPEC = KernelSpec(
    name="rtattr_computed_index_box_tog", kernel=kernel,
    signature={"in_ptr0": "*fp32", "out_ptr0": "*fp32", "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"},
    constexprs={"XBLOCK": XBLOCK, "R0_BLOCK": RBLOCK},
    args=[Arg("b", IN, "float32", (M,)), Arg("o", OUT, "float32", (X * R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    expect_ir={"09-custom.mlir": {"pytorchsim.runtime_attribute": 1}},
)
