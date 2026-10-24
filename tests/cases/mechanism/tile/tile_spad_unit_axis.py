"""A reduction over a COMPUTED tile, whose temporary carries a unit axis.

`tl.sum(x * x, axis=1)` reduces something that was not loaded, so the product
needs a buffer of its own -- and it arrives at the scratchpad as rank 3:

    memref.global @bufN_spad : memref<128x1x64xf32, 1> {pytorchsim.lane_axis = 0}

The middle axis is the reduced-away one, size 1 in the result's rank. It is a
NAME and not a place: the tile is still 128 rows across 128 lanes with 64
elements in each (rule 6a -- a size-1 axis moves no element between lanes).

WHAT THIS PINS. p13_lower_to_llvm sizes each reservation from the LOWERED LLVM IR, where
`pytorchsim.lane_axis` no longer exists, so it needs the spread told to it. It used to
be told only "is this ONE lane, yes or no", and for everything else it GUESSED
the spread by dividing by the narrowest dim of the type. On this buffer the
narrowest dim is the unit axis, so the guess is 1 lane: the whole 8192-element
tile reserved in EVERY lane, 32768 bytes on its own, against a 32768-byte
double-buffer budget that the kernel's other buffers then push it past.

WHAT BREAKS IT. Reverting p09_place_in_spad to tag only the ONE-LANE case, or
the reservation to fall back to the narrowest dim when a tag is present. Either way this
kernel stops at stage `binary` with

    scratchpad usage 33856 bytes/lane exceeds the double-buffer budget of 32768

Measured on 5222ec1 before the fix, against 832 after. The same cause had
layernorm at 35904, gqa_repeat_kv at 35424, fused_add_rmsnorm at 34880 and
reduce_half_sum at 66752, all four with `pytorchsim.lane_axis = 0` already stamped on
the oversized buffer.

WHY THE SHAPE IS THIS SHAPE. 128 rows is the lane count, so the unit axis is
distinguishable from the lane axis -- at 1 row they would both be 1 and the
guess would be right by accident. 64 columns makes the mis-sized buffer exactly
the whole budget, so the kernel fails on ONE buffer rather than on a total, which
is the fault being pinned. Anything narrower still mis-sizes and still fits, and
pins nothing (rule 2: a kernel that passes either way holds nothing down).

NOT A DUPLICATE of reduce/reduce_row_sum.py: that reduces the LOADED tile, which
needs no temporary at all and so never grows the unit axis.

Pins the scratchpad reservation against a temporary whose narrowest dim is a unit axis rather than the lane axis: the spread is read from pytorchsim_triton_compiler.lane_axis, not guessed from the shape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 64          # M == TORCHSIM_COMPILE_VECTORLANE_SIZE, so the lane axis is 128 long


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_spad_unit_axis_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                   N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        off = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + off)
        # The reduce's input is x*x, not x -- so it is materialised, and that
        # temporary is the [128, 1, 64] buffer this kernel exists for.
        s = tl.sum(x * x, axis=1)
        tl.store(o_ptr + off, x * s[:, None])

    return tile_spad_unit_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x * (x * x).sum(dim=1, keepdim=True)}


SPEC = KernelSpec(
    name="tile_spad_unit_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
