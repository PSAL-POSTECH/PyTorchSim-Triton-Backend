"""The same shuffles as ops_shape, carried at i64: the round trip is sized in bytes.

SIBLING: tests/cases/mechanism/ops/ops_shape.py is this kernel at f32 and is where
the shuffle set comes from -- join / cat / reshape / trans / split, which is the
smallest thing in the suite that makes fit_layout_to_lanes mint a relayout at all
(64 bytes of scratch, the least of the 28 kernels that mint one). This file
changes ONE thing, the element type, and that is the whole of it.

WHAT IT PINS. A relayout is a DRAM round trip: lane banks cannot permute among
themselves, so a tile that has to come back under different axes is written out
by an MVOUT and read back by an MVIN. fit_layout_to_lanes appends the DRAM for it
as one more kernel argument, declared `memref<*xf32>`, and used to record its
size in ELEMENTS -- which the C wrapper multiplied by four.

The declaration is not what the DMA reads. lower_to_dma takes
element_size from the SRAM tile (`_elem_bytes` of the spad memref) and forms the
DRAM address as `base + index * that`, so an i64 tile was already striding in
eights through a buffer allocated in fours. Half of it.

    develop f6232e1   Fatal: a DRAM round trip of i64 cannot describe its own
                      elements: the scratch this pass appends is memref<*xf32>
                      and the descriptor's element_size is 32 bits
    with the fix      ALL STAGES PASSED, max_abs_err 0, scratch 128 bytes
                      -- 16 i64 elements, and 16*4 = 64 is what the old
                      arithmetic would have allocated

THE REFUSAL WAS RIGHT TO EXIST; it was the size behind it that was wrong. Its
own comment says a wrong number is worse than a refusal, and the i1 measurement
it cites (mask_col_flag_broadcast_over_rows answering 2.56723, one column of
eight arriving) is a real one. But that measurement is about SUB-BYTE elements,
where the vector unit bit-packs and the descriptor cannot say "one bit"; 64 bits
never had that problem. One test was standing for two causes.

WHY THIS SHAPE AND NOT A SIMPLER ONE. Three conditions have to hold at once, and
dropping any of them makes the kernel pass either way:

  1. the tile is not 32 bits wide
  2. a relayout is actually MINTED -- the axes must genuinely disagree.
     `_reshape_to_relayout` counts with dry=True before rewriting, so a rank-1
     reshape that no op reads under another axis mints nothing. A 16-element
     `tl.reshape` there and back passes on both trees with zero relayouts.
  3. the minter must be one that does not ask about the width first. Of the
     four, only `_replicate_computed_row` asks -- and it DECLINES, dropping the
     candidate so the ranked menu picks another plan and the kernel quietly
     succeeds. The shuffles here go through `_reshape_to_relayout`, which does
     not ask, so the width reaches `_emit_relayout` where it is a refusal.

That third one is why "use an i64 tensor" is not enough to reproduce this, and
it is worth knowing before writing the next witness for this pass.

torch equivalent: ops_shape's, with integer arithmetic (see reference).

ops_shape's shuffles at i64: pins the relayout scratch being sized in bytes rather than in f32 elements.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 4           # outer-product side
K = 8           # vector length
XN = 64
SN = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_shape_i64_round_trip_kernel(x_ptr, s_ptr, M: tl.constexpr,
                                        K: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, K)
        a = tl.load(x_ptr + cols)
        b = tl.load(x_ptr + K + cols)

        j = tl.join(a, b)                              # (K, 2)
        c = tl.cat(a + 1, b - 1, can_reorder=True)     # (2K,)
        mrs = tl.reshape(c * 2, (2, K))                # 1-D -> 2-D  EXPAND
        t = tl.trans(mrs + 1, 1, 0)                    # (2,K) -> (K,2) TRANS
        d = tl.reshape(t * 3, (2, K))                  # (K,2) -> (2,K) DYNAMIC
        f = tl.reshape(d + 4, (2 * K,))                # 2-D -> 1-D  COLLAPSE
        tl.store(s_ptr + tl.arange(0, 2 * K), f)

        p, r = tl.split(j * 2 + t)                     # (K,2) -> two (K,)
        tl.store(s_ptr + 2 * K + cols, p - r)

        ar = tl.load(x_ptr + rows)
        bc = tl.load(x_ptr + M + rows)
        outer = ar[:, None] * bc[None, :]              # (M,1) x (1,M) -> (M,M)
        tl.store(s_ptr + 3 * K + rows[:, None] * M + rows[None, :], outer)

    return ops_shape_i64_round_trip_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-100, 100, (XN,), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    import torch
    x = inputs["x"]
    out = torch.zeros(SN, dtype=torch.int64)
    a, b = x[:K], x[K:2 * K]
    j = torch.stack([a, b], dim=1)
    c = torch.cat([a + 1, b - 1])
    mrs = (c * 2).reshape(2, K)
    t = (mrs + 1).t()
    d = (t * 3).reshape(2, K)
    out[:2 * K] = (d + 4).reshape(2 * K)
    pr = j * 2 + t
    out[2 * K:3 * K] = pr[:, 0] - pr[:, 1]
    ar, bc = x[:M], x[M:2 * M]
    out[3 * K:3 * K + M * M] = (ar[:, None] * bc[None, :]).reshape(-1)
    return {"s": out}


SPEC = KernelSpec(
    name="ops_shape_i64_round_trip_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "s_ptr": "*i64",
               "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "int64", (XN,)), Arg("s", OUT, "int64", (SN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="open",
    expect_why="stage 6 refuses the kernel llc gave back: a STRIDED store of "
               "i64 is scalarised into vslidedown.vx + vmv.x.s + two sw per "
               "element, and a vector->scalar extract keeps lane 0 and drops "
               "the rest. Nothing in our IR asks for it -- the trigger is the "
               "strided layout we hand llc, so the fix is either not to ask "
               "for one or to issue the strided store ourselves",
)
