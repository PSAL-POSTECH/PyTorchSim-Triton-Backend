"""A masked store into an INOUT: the rows it turns off keep what they held.

    o[rm, :] = v[rm, :]   where flag[rm]      o arrives with data in it
    the rows without the flag must come out UNCHANGED, not zeroed

WHY THIS KERNEL EXISTS. A structured masked store is lowered as a dense read of
the destination, a select, and a dense write of the whole tile -- so the rows the
mask turns off are written back with what the read found. Every other masked
store in this suite writes an `out`, which arrives zeroed, so "kept what it held"
and "wrote zeros" are the same answer there and neither is tested.

WHAT IT PINS, twice over: that the access takes the dense path at all (the IR
count -- one indirect descriptor would mean it fell back to per-element), and
that the path preserves the destination rather than clearing or overwriting it.

THE MASK IS A LOADED FLAG on purpose. A range on the row axis is a box, and a
box shrinks the transfer instead of reaching the question this kernel asks.

The only masked store in the suite whose destination arrives with data in it, so the write-back is visible.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import (KernelSpec, Arg, IN,
                                                        INOUT)

BM, BN = 8, 16          # rows written, columns per row
KEEP = 5                # rows 0..4 carry the flag; 5..7 must survive untouched


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_store_rmw_keeps_the_destination_kernel(x_ptr, f_ptr, o_ptr,
                                                    BM: tl.constexpr,
                                                    BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        off = rm[:, None] * BN + rn[None, :]
        v = tl.load(x_ptr + off)
        keep = (tl.load(f_ptr + rm) > 0)[:, None]
        tl.store(o_ptr + off, v, mask=keep)

    return mask_store_rmw_keeps_the_destination_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(BM, dtype=torch.int32)
    f[:KEEP] = 1
    # NOT ZEROS: the destination has to hold something a lost write would show.
    return {"x": torch.randn(BM * BN, generator=g, dtype=torch.float32),
            "f": f,
            "o": torch.arange(1, BM * BN + 1, dtype=torch.float32)}


def reference(inputs):
    o = inputs["o"].clone().reshape(BM, BN)
    v = inputs["x"].reshape(BM, BN)
    for i in range(BM):
        if int(inputs["f"][i]) > 0:
            o[i] = v[i]
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="mask_store_rmw_keeps_the_destination_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (BM * BN,)),
          Arg("f", IN, "int32", (BM,)),
          Arg("o", INOUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # TWO BURSTS, NOT ONE DESCRIPTOR PER ELEMENT: the dense read of the
    # destination is the second transfer, and no descriptor is indirect.
    expect_ir={"08-transformed.mlir": {"offset_stride": 0}},
)
