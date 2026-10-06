"""Inductor's conv mask, reduced: FIVE row bounds AND one column bound, in a loop.

    m = (h >= 0)[:, None] & (h < IN_H)[:, None]
      & (w >= 0)[:, None] & (w < IN_W)[:, None]
      & (c < IN_C)[None, :]

THE SHAPE COMES OFF THE GENERATED KERNEL and not from a guess about it. Inductor's
convolution template writes exactly this, once per `ijk` iteration, with `idx_x_h`
and `idx_x_w` recomputed from the loop's own induction variable:

    mask_x = (idx_n < BATCH)[:, None]
           & (idx_x_h >= 0)[:, None] & (idx_x_h < IN_H)[:, None]
           & (idx_x_w >= 0)[:, None] & (idx_x_w < IN_W)[:, None]
           & (idx_x_c < GROUP_IN_C)[None, :]

and resnet18's first conv is where it arrives. Three properties matter and all
three are load-bearing:

    OPPOSITE DIMS      the folded row bounds land on dim 0 and the channel bound
                       on dim 1, so whichever dim rides the lanes one of them has
                       its values in other lanes' banks. Rule 8: nothing crosses.
    SUB-WORD           the folded row mask is `memref<BMxi8>` -- an i1 that
                       bufferize gives a byte.
    LOOP-CARRIED       `h` and `w` move with the loop, so bufferize REUSES the
                       buffers behind the mask. That is what takes the free
                       repairs off the table: `_row_chain` requires one writer
                       per buffer and is right to, since a buffer written twice
                       no longer holds one image.

WHAT EACH RUNG SAYS, and every line of it is measured on THIS kernel:

    0' widen in place  `_widen_chain` -- the row is read narrow elsewhere
    1  widen the load  `_clone_loaded_row_wide` -- the writer is a compare, not
                       an MVIN
    2  rebuild it      `_rebuild_computed_row` -- that compare reads something
    2' clone the chain `_clone_row_chain` -- a buffer in the chain has two
                       writers
    3  round trip      the only rung left, and it declined for the row's WIDTH:
                       the scratch `_add_dram_arg` appends is `memref<*xf32>`,
                       so the descriptor's element_size is 32 bits whatever the
                       row is made of (SCRATCH_BITS)

The first four are properties of the kernel and not of the fix, so they read the
same either side of it; the fifth is the one that moves.

and then vectorize refuses the op it was never staged for:

    NotVectorisable: operands of one elementwise op are banked on different
    iteration dims [0, 1]; they share an iteration space and cannot share a
    loop nest

THE FIX IS THE TRIP AND NOT THE DESCRIPTOR. The row is widened to i32 on the way
out and truncated back on the way in, so what crosses is 32 bits wide and the
descriptor goes on telling the truth. `_travels_as_scratch` states which element
types that pair is exact for, and why signedness does not enter into it.

torch equivalent: a bounds-masked gather accumulated over the loop.

Inductor's conv bounds mask reduced: row bounds and a column bound in one loop-carried i8 row, which only the round trip can stage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM = 32          # rows of the tile
BN = 16          # columns -- the "channel" axis
IN_H = 6
IN_W = 6
IN_C = 12        # < BN, so the column bound is a real clamp
STEPS = 4
NX = IN_H * IN_W * BN


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_conv_bounds_row_over_lanes_kernel(x_ptr, w_ptr, o_ptr,
                                               BM: tl.constexpr,
                                               BN: tl.constexpr,
                                               IN_H: tl.constexpr,
                                               IN_W: tl.constexpr,
                                               IN_C: tl.constexpr,
                                               STEPS: tl.constexpr):
        om = tl.arange(0, BM)
        on = tl.arange(0, BN)
        oh = om // IN_W
        ow = om % IN_W
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for s in range(STEPS):
            # Both move with the loop, which is what makes bufferize reuse the
            # buffers behind the mask below.
            h = oh + s - 1
            w = ow + s - 1
            m = ((h >= 0)[:, None] & (h < IN_H)[:, None]
                 & (w >= 0)[:, None] & (w < IN_W)[:, None]
                 & (on < IN_C)[None, :])
            p = x_ptr + ((h * IN_W + w) * BN)[:, None] + on[None, :]
            xt = tl.load(p, mask=m, other=0.0)
            # THE DOT IS WHAT PINS THE LANE AXIS. Without it fit_layout_to_lanes is free to put
            # the lanes on dim 0, where the row mask is an in-lane splat and
            # nothing has to be staged at all -- MEASURED: the same kernel with
            # `acc += xt` passes both before and after the fix, so it pins
            # nothing. The systolic array constrains all three of its operands,
            # so the tile is banked on dim 1 and the dim-0 mask beside it is
            # then the row that must reach every lane.
            wt = tl.load(w_ptr + on[:, None] * BN + on[None, :])
            acc += tl.dot(xt, wt, allow_tf32=True)
        tl.store(o_ptr + om[:, None] * BN + on[None, :], acc)

    return mask_conv_bounds_row_over_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "w": torch.randn(BN * BN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    wt = inputs["w"].reshape(BN, BN)
    om = torch.arange(BM)
    on = torch.arange(BN).reshape(1, BN)
    oh = (om // IN_W).reshape(BM, 1)
    ow = (om % IN_W).reshape(BM, 1)
    acc = torch.zeros(BM, BN, dtype=torch.float32)
    for s in range(STEPS):
        h = oh + s - 1
        w = ow + s - 1
        m = ((h >= 0) & (h < IN_H) & (w >= 0) & (w < IN_W)) & (on < IN_C)
        # The address is only READ where the mask holds, so it is clamped here
        # for the gather and the mask decides what is kept -- the same statement
        # tl.load makes with `other=0.0`.
        idx = (((h * IN_W + w) * BN) + on).clamp(0, NX - 1)
        xt = torch.where(m, x[idx], torch.zeros(BM, BN,
                                                dtype=torch.float32))
        acc = acc + xt @ wt
    return {"o": acc.reshape(BM * BN)}


SPEC = KernelSpec(
    name="mask_conv_bounds_row_over_lanes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "BM": "constexpr",
               "BN": "constexpr", "IN_H": "constexpr", "IN_W": "constexpr",
               "IN_C": "constexpr", "STEPS": "constexpr"},
    constexprs={"BM": BM, "BN": BN, "IN_H": IN_H, "IN_W": IN_W, "IN_C": IN_C,
                "STEPS": STEPS},
    args=[Arg("x", IN, "float32", (NX,)),
          Arg("w", IN, "float32", (BN * BN,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
