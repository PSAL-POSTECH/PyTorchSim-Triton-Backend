"""A rank-3 row gather: one index per dim 0, broadcast over dims 1 and 2.

SIBLING: ptr_gather_2d_row.py is the same shape one dimension down. Read that
first -- it explains why a one-index-per-row gather needs the index broadcast to
tile shape at all (the DMA reads an index for every tile ELEMENT, not every row).

This file exists because that broadcast used to be lowered only for rank 2.
`_bank_broadcast` returned without doing anything for any other rank, and the op
then fell through to convert-linalg-to-loops -- which writes element (r, j, k) at
the LINEAR offset while the DMA reads it from lane r's bank. Measured before the
fix: err 5.08, compiled clean, ran clean, no diagnostic anywhere.

WHAT THE LOWERING LOOKS LIKE NOW

    affine.for %d = 0 to depth          # lane axis (dim 0)
      affine.for %j = 0 to D1           # plain loop
        affine.for %k = 0 to D2 step V  # innermost dim is the vector
          %v = affine.vector_load %src[%d]   : vector<1xT>   one per lane
          %b = vector.broadcast %v           : vector<VxT>   copies, same lane
               affine.vector_store %b, %dst[%d, %j, %k]

Any rank: dim 0 on the lanes, the innermost dim in the vector, plain loops in
between. The splat is INSIDE a lane -- no data crosses lanes -- which is what
makes the result banked without anyone computing an address.

That path also no longer declines silently: a shape it cannot express raises,
because there is no correct scalar fallback over a banked scratchpad.

torch equivalent: x.view(ROWS, W)[idx][:, :BN*BK]

rank-3 row gather. The index broadcast was rank-2 only and declined everything else silently; err 5.08 -> 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, W = 512, 16
BM, BN, BK = 128, 4, 4          # BM is the lane count; BN*BK == W


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_3d_row_kernel(x_ptr, i_ptr, o_ptr, W: tl.constexpr,
                                 BM: tl.constexpr, BN: tl.constexpr,
                                 BK: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rk = tl.arange(0, BK)
        idx = tl.load(i_ptr + rm)                       # one index per row
        v = tl.load(x_ptr + idx[:, None, None] * W
                    + rn[None, :, None] * BK + rk[None, None, :])
        tl.store(o_ptr + rm[:, None, None] * (BN * BK)
                 + rn[None, :, None] * BK + rk[None, None, :], v)

    return ptr_gather_3d_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS * W, generator=g, dtype=torch.float32),
            "i": torch.randperm(ROWS, generator=g)[:BM].to(torch.int32)}


def reference(inputs):
    import torch
    x = inputs["x"].view(ROWS, W)
    return {"o": x[inputs["i"].long()][:, :BN * BK].contiguous().reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_3d_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "W": "constexpr", "BM": "constexpr", "BN": "constexpr",
               "BK": "constexpr"},
    constexprs={"W": W, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("x", IN, "float32", (ROWS * W,)), Arg("i", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (BM * BN * BK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
