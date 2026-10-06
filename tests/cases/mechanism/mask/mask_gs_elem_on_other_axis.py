"""The axis rule's cell, transposed: gather on dim 1, unstructured mask on dim 0.

mask_gs_dim_mismatch is gather-on-0 with the mask on 1, and it was the only
kernel in this cell -- one shape producing both the load and the store. This is
the mirror, so the cell is pinned on both axes rather than one.

    j     = tl.load(j_ptr + rn)      a permutation -> gather dim is 1
    rmask = tl.load(m_ptr + rm) < 5  unstructured, and it is on dim 0

The mask is a COMPARISON of a loaded value: MaskAnalysis can hold that as an
axis's mask even though it cannot fold it into an extent. A bare `flag != 0`
cannot be parsed at all and the whole access falls to the unstructured route
instead, which is a different cell.

`j` is a permutation so the store is injective; two columns landing on one would
make the answer depend on write order rather than on the mask.

The mirror of mask_gs_dim_mismatch: gather on dim 1 with the unstructured mask on dim 0, so the axis rule's cell is pinned on both axes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BN = 4, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gs_elem_on_other_axis(x_ptr, j_ptr, m_ptr, o_ptr, BM: tl.constexpr,
                              BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        j = tl.load(j_ptr + rn)                 # gather dim is 1
        rmask = tl.load(m_ptr + rm) < 5         # unstructured mask on dim 0
        v = tl.load(x_ptr + rm[:, None] * BN + j[None, :],
                    mask=rmask[:, None], other=0.0)
        tl.store(o_ptr + rm[:, None] * BN + j[None, :], v,
                 mask=rmask[:, None])

    return gs_elem_on_other_axis


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * BN, generator=g, dtype=torch.float32),
            "j": torch.tensor([2, 0, 3, 1], dtype=torch.int32),
            "m": torch.tensor([1, 9, 3, 7], dtype=torch.int32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    j = inputs["j"].to(torch.int64)
    rmask = (inputs["m"] < 5)
    out = torch.zeros(BM * BN, dtype=torch.float32)
    for i in range(BM):
        if not bool(rmask[i]):
            continue
        for k in range(BN):
            out[i * BN + int(j[k])] = x[i * BN + int(j[k])]
    return {"o": out.contiguous()}


SPEC = KernelSpec(
    name="gs_elem_on_other_axis", kernel=kernel,
    signature={"x_ptr": "*fp32", "j_ptr": "*i32", "m_ptr": "*i32",
               "o_ptr": "*fp32", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (BM * BN,)),
          Arg("j", IN, "int32", (BN,)),
          Arg("m", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
