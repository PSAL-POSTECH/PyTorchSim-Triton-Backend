"""convert_accesses_to_transfer's (pointer x mask x wrap x direction) grid, and the cells only this fills.

THE GRID. Every access arriving at convert_accesses_to_transfer has one of three
POINTERS -- `tts.make_tptr`, `tts.make_gather_scatter_tptr`, or a bare
`tts.gather`/`tts.scatter` -- some set of MASK CARRIERS, a WRAP or not, and a
direction. Counted over the suite at the top of run(), before any carrier is
rewritten, MVIN/MVOUT:

    tptr    none 1080/598   box  70/40   elem 3/1   box+elem 4/1    nowrap
            none   43/12    box   5/2    elem 1/1                   wrap
    gstptr  none  122/15    box   6/2    elem 2/2   gs 37/3
                                                    box+gs 2/1      nowrap
            none    4/1     box   2/2    elem 1/1   gs  1/1         wrap
    full    none  194/12    elem 77/12

THE CARRIERS ARE NOT EXCLUSIVE, which is the first thing a grid of this shape
gets wrong. A mask sits on the ACCESS as `static_mask_dims` (a box: a low and a
high per axis), on the access as an elementwise `elem_mask`, or on the POINTER
in `make_gather_scatter_tptr`'s segment 5 -- upstream's "the box analysis gave
up on the gathered axis" flag, one bit per ROW. `box+elem` and `box+gs` are 8
accesses that carry two at once.

WHAT THIS FILE ADDS. Four cells nothing else reaches: `gstptr x box x wrap` and
`gstptr x gs x nowrap`, both in both directions.

  load 1 / store 1   wrap on axis 0, gather on axis 1, the box on axis 2
  load 2 / store 2   gather with the mask ON the gathered axis, nothing wrapped

WHY THE BOX NEEDS A THIRD AXIS. `mask=(r < K)` over a WRAPPED r is not a box: a
prefix of r is not a prefix of `r % M`. The bound has to be on an axis that is
itself plain, which with a wrap and a gather already spoken for means rank 3.

HOW THE COUNT LIED, twice, and both traps are in reading the IR rather than in
the IR itself. `_flat_shape` reads operand segment 5, which on a GATHER-SCATTER
pointer is the mask and not a flat modulus, so every gs-masked access read as
wrapped and `gs x nowrap` looked empty -- 40 accesses in the wrong row. And the
`dims` `_split_access_operands` returns are only the KDYN operands, so a box
with static extents has none and reads as unmasked, which hid most of the box
column. A third: the `0` a gathered axis carries in `static_mask_dims` is that
give-up flag, not an extent, and counting it as one turns every partial gather
into a box.

Fills four cells of convert_accesses_to_transfer's pointer x mask x wrap grid that nothing else reaches. p and s are INOUT: they are written only where the mask keeps, so a zero-filled base would let a dropped store pass.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import (KernelSpec, Arg, IN,
                                                        INOUT, OUT)

BR, BS, BT, M, KEEP = 8, 8, 4, 3, 3
ASIZE = BR * BS * BT



BR, BS, BT, M, KEEP = 8, 8, 4, 3, 3
ASIZE = BR * BS * BT


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_mask_wrap_grid_kernel(a_ptr, j_ptr, o_ptr, p_ptr, q_ptr,
                                         s_ptr, BR: tl.constexpr,
                                         BS: tl.constexpr, BT: tl.constexpr,
                                         M: tl.constexpr, KEEP: tl.constexpr):
        r = tl.arange(0, BR)
        s = tl.arange(0, BS)
        t = tl.arange(0, BT)
        j = tl.load(j_ptr + s)
        # gstptr x box x wrap: wrap on 0, gather on 1, the bound on 2.
        wrapped = ((r % M)[:, None, None] * (BS * BT)
                   + j[None, :, None] * BT + t[None, None, :])
        box = (t < KEEP)[None, None, :]
        v = tl.load(a_ptr + wrapped, mask=box, other=0.0)
        tl.store(o_ptr + (r[:, None, None] * (BS * BT) + s[None, :, None] * BT
                          + t[None, None, :]), v)
        # ALIASING IS SAFE HERE: `v` came from this same wrapped address, so
        # the rows that alias under `% M` write the same value.
        tl.store(p_ptr + wrapped, v, mask=box)
        # gstptr x gs x nowrap: the mask is on the GATHERED axis, so upstream
        # parks it on the pointer, and nothing wraps.
        row = r[:, None] * BS + j[None, :]
        keep = (j < KEEP)[None, :]
        g = tl.load(a_ptr + row, mask=keep, other=0.0)
        tl.store(q_ptr + (r[:, None] * BS + s[None, :]), g)
        tl.store(s_ptr + row, g, mask=keep)

    return ptr_gather_mask_wrap_grid_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32),
            "j": torch.randint(0, BS, (BS,), generator=g, dtype=torch.int32),
            "p": torch.randn(ASIZE, generator=g, dtype=torch.float32),
            "s": torch.randn(BR * BS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a, j = inputs["a"], inputs["j"].long()
    r = torch.arange(BR)
    sr = torch.arange(BS)
    t = torch.arange(BT)
    box = (t < KEEP)[None, None, :].expand(BR, BS, BT)
    wrapped = ((r % M)[:, None, None] * (BS * BT) + j[None, :, None] * BT
               + t[None, None, :])
    v = torch.where(box, a[wrapped], torch.zeros(()))
    pf = inputs["p"].clone()
    pf[wrapped[box]] = v[box]
    row = r[:, None] * BS + j[None, :]
    keep = (j < KEEP)[None, :].expand(BR, BS)
    g = torch.where(keep, a[row], torch.zeros(()))
    sf = inputs["s"].clone()
    sf[row[keep]] = g[keep]
    return {"o": v.reshape(-1), "p": pf, "q": g.reshape(-1), "s": sf}


SPEC = KernelSpec(
    name="ptr_gather_mask_wrap_grid_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "j_ptr": "*i32", "o_ptr": "*fp32",
               "p_ptr": "*fp32", "q_ptr": "*fp32", "s_ptr": "*fp32",
               "BR": "constexpr", "BS": "constexpr", "BT": "constexpr",
               "M": "constexpr", "KEEP": "constexpr"},
    constexprs={"BR": BR, "BS": BS, "BT": BT, "M": M, "KEEP": KEEP},
    args=[Arg("a", IN, "float32", (ASIZE,)), Arg("j", IN, "int32", (BS,)),
          Arg("o", OUT, "float32", (ASIZE,)),
          Arg("p", INOUT, "float32", (ASIZE,)),
          Arg("q", OUT, "float32", (BR * BS,)),
          Arg("s", INOUT, "float32", (BR * BS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 4}},
)
