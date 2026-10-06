"""A masked store whose positions SHARE addresses: read-modify-write is wrong.

    o[(rm % 8), :] = v[rm, :]     rm = 0..15,  masked to the first eight
    rows 8..15 land on the same eight rows as 0..7 -- the wrap REPEATS

WHY IT MATTERS. A masked store with a rectangular address is now lowered as a
dense read, a select and a dense write -- two bursts instead of one descriptor
per element. That rewrite writes the WHOLE tile back, so a masked-off position
puts what it read over whatever a live position wrote to the same address.

WHAT IT PINS: that the rewrite asks first. Here the modulo folds into the
pointer's wrap and two positions meet, so the access keeps its per-element
address and only the live ones are written.

WHAT BREAKS WITHOUT THE CHECK: rows 8..15 read the OLD o -- zeros -- and write
them over rows 0..7, which the live half had just filled. Measured on this tree
with _tile_injective forced true: 2.11522 against the reference.

Two rows, one address. The dense-write rewrite has to decline or a masked-off position erases a live one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM = 16         # rows written
BN = 4          # columns, so the tile stays structured
W = 8           # the modulus: two rows per address
# ROOM TO SPARE ON PURPOSE. The tile's addresses have to fit inside the argument
# or the bounds guard declines first and this kernel would pin nothing; 64 is
# comfortably past the 34 a wrapped axis is charged for.
NO = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_store_aliases_refuses_rmw_kernel(x_ptr, f_ptr, o_ptr,
                                              BM: tl.constexpr,
                                              BN: tl.constexpr,
                                              W: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        v = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        # A LOADED FLAG, so MaskAnalysis cannot fold it into a box -- a box on
        # the row axis would shrink the transfer and never reach this question.
        keep = (tl.load(f_ptr + rm) > 0)[:, None]
        tl.store(o_ptr + (rm % W)[:, None] * BN + rn[None, :], v, mask=keep)

    return mask_store_aliases_refuses_rmw_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(BM, dtype=torch.int32)
    f[:W] = 1
    return {"x": torch.randn(BM * BN, generator=g, dtype=torch.float32),
            "f": f}


def reference(inputs):
    import torch
    o = torch.zeros(NO)
    v = inputs["x"].reshape(BM, BN)
    for i in range(BM):
        if int(inputs["f"][i]) > 0:
            o[(i % W) * BN:(i % W) * BN + BN] = v[i]
    return {"o": o}


SPEC = KernelSpec(
    name="mask_store_aliases_refuses_rmw_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr", "W": "constexpr"},
    constexprs={"BM": BM, "BN": BN, "W": W},
    args=[Arg("x", IN, "float32", (BM * BN,)),
          Arg("f", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
