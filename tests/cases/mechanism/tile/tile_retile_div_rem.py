"""A 2-D box written as one divided 1-D index -- retiled back into two axes.

    base = (i // OW) * (2 * W) + (i % OW) * 2        i = arange(0, NQ)
    x    = load(a + base[:, None] * D + d[None, :])

WHAT MAKES IT A BOX. `i // OW` and `i % OW` are the row and the column of a
grid the sequence secretly is, and each is multiplied by a stride of its own --
`2 * W` and `2`. Two axes, spelled with one `arange` and a divide, which is
where a structured pointer analysis stops being affine and falls to a gather.
`A == K * B` would be the same axis split and rejoined, and that is reflattenDivRem's shape (PtrAnalysis.cpp),
not this one: 16 != 4 * 2.

WHO SPLITS IT. The linalg lowering, in PtrState -- one tensor axis becomes two state
axes and `visitOperandDiv` writes the outer one. pytorchsim-compile used to answer the same
question a stage earlier and no longer does: with `_retile_div_rem` deleted the
corpus's 07-regularized.mlir is byte-identical on all 450 kernels that produce one.
The walk is `tts.make_tptr` with three strides and every axis is one the DMA can
step. The tile comes back to the shape
the kernel wrote with a `tl.reshape`, which is a renumbering of the outer axes
and costs nothing (tile_collapse_outer_axes is the witness for that half).

The pool of sam2_qpool_attn is this index; here it is alone.

A divided index that is really two axes: retiled, so the walk is a plain strided pointer instead of one address per row.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W = 8, 8
OW = W // 2
NQ = (H // 2) * OW
D = 16
ASIZE = H * W * D


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def retile_div_rem(a_ptr, o_ptr, NQ: tl.constexpr, OW: tl.constexpr,
                       W: tl.constexpr, D: tl.constexpr):
        i = tl.arange(0, NQ)
        d = tl.arange(0, D)
        base = (i // OW) * (2 * W) + (i % OW) * 2
        x = tl.load(a_ptr + base[:, None] * D + d[None, :])
        tl.store(o_ptr + i[:, None] * D + d[None, :], x)

    return retile_div_rem


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"]
    i = torch.arange(NQ)
    base = (i // OW) * (2 * W) + (i % OW) * 2
    idx = base[:, None] * D + torch.arange(D)[None, :]
    return {"o": a[idx].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="retile_div_rem", kernel=kernel,
    signature={"a_ptr": "*fp32", "o_ptr": "*fp32", "NQ": "constexpr",
               "OW": "constexpr", "W": "constexpr", "D": "constexpr"},
    constexprs={"NQ": NQ, "OW": OW, "W": W, "D": D},
    args=[Arg("a", IN, "float32", (ASIZE,)),
          Arg("o", OUT, "float32", (NQ * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0,
                                    "tts.make_gather_scatter_tptr": 0}},
)
