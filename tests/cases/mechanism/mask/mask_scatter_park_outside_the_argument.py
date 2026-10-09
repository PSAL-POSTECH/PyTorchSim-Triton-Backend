"""A masked scatter parked FAR outside its argument, and the DMA drops it.

    o[gidx[i] * TD + j] = v      masked, so some row must go NOWHERE
    the nowhere is an INDEX, and the DMA scales it by the gathered stride TD,
    so it is rounded UP to a whole stride: 8256 bytes of buffer, landing at
    9216 -- 960 bytes past the end, and past the allocation as well

WHAT IT PINS: that the descriptor carries the argument's range and the DMA
honours it. A masked-off write does not need a place to LAND any more, only a
place to MISS -- which is why nothing is reserved behind an output now.

WHAT IT USED TO NEED. The landing spot had to exist, so every out/inout was
allocated twice over -- 2.5 GiB behind a Llama-4 output for 32 bytes of use.
Shrinking that to a constant made this kernel a refusal (the overshoot is one
gathered stride, and a stride is whatever the kernel wrote). The range removes
both the reservation and the refusal: this now runs, and its numbers are right.

AND WHAT IT CANNOT PIN, said out loud. The drop is a SAFETY property, not a
value one: a parked write lands where nothing reads, so turning the range off
leaves every number in this suite unchanged -- measured, this kernel passes
either way. What it pins is that the compiler no longer refuses the shape and
that it marks the transfer; that the DMA then honours the mark is spike's.

A park 960 bytes outside the argument. The DMA drops it because the descriptor says where the argument ends.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BN = 8, 4
TD = 256        # the row pitch, and so the gathered stride: 256 * 4 > 512
# NOT A MULTIPLE OF TD: the park is an INDEX rounded up to a whole stride, so a
# buffer whose end sits on a stride boundary would land exactly on it. 8256
# bytes puts the landing spot 960 past the end, well outside any rounding.
NT = 2064
KEEP = 5        # the gathered row this predicate turns off -- it gets parked


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_scatter_park_outside_the_argument_kernel(
            x_ptr, idx_ptr, o_ptr, TD: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, KEEP: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        gidx = tl.load(idx_ptr + rm)
        v = tl.load(x_ptr + (rm[:, None] * BN + rn[None, :]))
        # NOT A RANGE ON THE ROW AXIS -- that is a box, and a box is a clamp the
        # DMA already skips. A predicate over the GATHERED value is one
        # MaskAnalysis cannot fold into a box, so the address has to be parked.
        keep = (gidx != KEEP)[:, None]
        tl.store(o_ptr + (gidx[:, None] * TD + rn[None, :]), v, mask=keep)

    return mask_scatter_park_outside_the_argument_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * BN, generator=g, dtype=torch.float32),
            "idx": torch.arange(BM, dtype=torch.int32)}


def reference(inputs):
    import torch
    o = torch.zeros(NT)
    v = inputs["x"].reshape(BM, BN)
    for i in range(BM):
        g = int(inputs["idx"][i])
        if g != KEEP:
            o[g * TD:g * TD + BN] = v[i]
    return {"o": o}


SPEC = KernelSpec(
    name="mask_scatter_park_outside_the_argument_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "TD": "constexpr", "BM": "constexpr", "BN": "constexpr",
               "KEEP": "constexpr"},
    constexprs={"TD": TD, "BM": BM, "BN": BN, "KEEP": KEEP},
    args=[Arg("x", IN, "float32", (BM * BN,)),
          Arg("idx", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (NT,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE MARK, because the number cannot be the evidence (see the docstring).
    expect_ir={"08-transformed.mlir": {"dram_bound": 1}},
)
