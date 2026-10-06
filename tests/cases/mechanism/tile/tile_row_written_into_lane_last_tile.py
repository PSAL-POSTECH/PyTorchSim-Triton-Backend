"""A ROW of a tile banked on its LAST axis, written through a `memref.subview`.

    m = tl.reshape(tl.cat(a, b), (2, K))      # two rows, written one at a time
    t = tl.trans(m * 2.0, 1, 0) * 3.0         # the trans lands on m's lane axis

The transpose is absorbed into the map of the op that reads it (rule 8), so the
[2, K] tile is banked on axis 1 -- one COLUMN per lane, lane c holding (0, c) at
in-bank offset 0 and (1, c) at in-bank offset 1. The cat is bufferized into two
writes of one row each, and a row is a `memref.subview` of that tile:

    memref.subview %m[0, 0] [1, K] [1, 1]  ->  memref<Kxf32, strided<[1]>, 1>
    memref.subview %m[1, 0] [1, K] [1, 1]  ->  memref<Kxf32, strided<[1], offset: K>, 1>

WHAT THIS PINS IS THE OFFSET IN THAT TYPE. `offset: K` is the origin linearised
with the tile's ROW-MAJOR strides, because a memref type can carry no other
layout -- while the row the DMA laid down starts at in-bank offset 1 of every
lane. vectorize's generic path addressed the operand's own type, so the
store went K elements into each lane's bank, past the end of a reservation that
holds two; the row the reader wanted was never written, and the reader read
whatever the bank held.

    measured   THIS kernel on the tree in front of the fix: max_abs_err
               7.58001, max_rel_err 1, 8 of 16 elements returned 0.0 -- one
               whole column of the transposed result, which is the row that was
               never written. Every stage reported ok; only `verify` said
               anything.

    measured   ops/ops_shape, the nine-op family kernel this was cut down from:
               max_abs_err 19.58, verifying clean the whole way down. The tile
               is `memref<2x8xf32, 1>` on 8 lanes reserved 8 deep, so the store
               landed inside the NEXT global rather than off the end of the
               scratchpad -- silent (rule 12). The fused body ((x+1)*2+1)*3+4
               applied to the zeros left behind gives exactly 13.0, which is
               what came back.

WHAT MAKES IT FAIL AGAIN: address a spad operand by its own memref type instead
of through `_banked_view`. The strides were already re-derived from the banking
and the offset was not, so the two halves of one layout came from two different
descriptions -- and `_address_agrees`, which used to check the generic path's
addresses, could not see it: both of its sums started from the same wrong base
and agreed with each other.

A tile banked on axis 0 keeps passing whatever this pass does with the offset,
because there the bank's strides ARE the type's and the flat origin IS the
in-bank one. Two rows and a transpose are the smallest thing that separates
them.

torch equivalent: o = (torch.cat([a, b]).reshape(2, K) * 2.0).T * 3.0.

A row of a tile banked on its last axis, written through a memref.subview whose type states a row-major offset. Addressing the operand's own type stored the row K elements into each lane's bank instead of 1, and the reader read what was never written (ops_shape, max_abs_err 19.58, no diagnostic).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def row_into_lane_last_kernel(x_ptr, o_ptr, K: tl.constexpr):
        cols = tl.arange(0, K)
        a = tl.load(x_ptr + cols)
        b = tl.load(x_ptr + K + cols)
        # cat + reshape: the two halves are written into the [2, K] tile one
        # row at a time, which is where the subview comes from.
        m = tl.reshape(tl.cat(a, b, can_reorder=True), (2, K))
        # The transpose is absorbed into the reader's map, which is what puts
        # the lane axis on dim 1 of m rather than on dim 0.
        t = tl.trans(m * 2.0, 1, 0) * 3.0                       # (K, 2)
        tl.store(o_ptr + cols[:, None] * 2 + tl.arange(0, 2)[None, :], t)

    return row_into_lane_last_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(2 * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    m = x.reshape(2, K)
    t = (m * 2.0).transpose(0, 1).contiguous() * 3.0
    return {"o": t.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="row_into_lane_last_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr"},
    constexprs={"K": K},
    args=[Arg("x", IN, "float32", (2 * K,)),
          Arg("o", OUT, "float32", (2 * K,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
