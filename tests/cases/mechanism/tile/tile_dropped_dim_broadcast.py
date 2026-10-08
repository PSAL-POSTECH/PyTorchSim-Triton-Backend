"""A broadcast whose map DROPS the dim instead of pinning it, on a banked tile.

THE SHAPE. `(rk[:, None] // 2)` inside a 2-D load offset is a column computed
once and used for every element of the row, so the linalg lowering writes it as a
rank-2 value read by a rank-3 generic:

    linalg.generic ins(%col : 32x1xi32) outs(%tile : 32x1x16xi32)
      indexing_maps = [(d0, d1, d2) -> (d0, d1), (d0, d1, d2) -> (d0, d1, d2)]

The ins map has TWO results over a THREE-dim iteration space. d2 is not in it,
and that omission is the whole statement: this operand has no position along d2,
which is what a broadcast is. It is the same sentence as the pinned spelling
`(d0, d1, d2) -> (d0, d1, 0)` over a `[32, 1, 16]` operand, and both have to
bank, because both are normal IR -- the linalg lowering emits the dropped form when
linalg-fuse-elementwise-ops collapses a chain, and
`-linalg-fold-unit-extent-dims` rewrites every pinned one into it.

The unit axis is d1, which is EXTENT ONE on both sides -- nothing crosses lanes
and nothing is renumbered (rule 6a), so nothing here is a conflict. It is the
shape rule 8b calls a unit-axis copy: a name change, not a data move.

WHAT THIS PINS, AND IT USED TO BE THIS FILE'S OWN FAILURE. vectorize
refused it, first as

    not identity-elementwise, and a scalar fallback would read the banked
    scratchpad linearly

and then, once the maps were being read rather than pattern-matched, as

    indexing map (d0, d1, d2) -> (d0, d1) over a rank-2 operand is not a
    projected permutation of a rank-3 iteration space, so this is not one
    element per element

-- a refusal whose own docstring argued that a map with fewer results was "a
different shape, not a harder one". It is not a different shape. An axis that is
absent and an axis that holds one element say the same thing, and the pass now
turns both into the same `None` before anything else reads them.

WHAT BREAKS IT AGAIN: requiring `len(map.results) == iteration rank` anywhere;
handing a load the NEST's identity affine map instead of one sized to the
operand's own rank (the verifier catches that one -- numResults against memref
rank -- so it is the safe half); or reintroducing a second code path for the
pinned spelling, at which point the two will drift.

WHAT IS STILL REFUSED, and deliberately: a map that repeats a dim (a diagonal,
which is not one element per element) or whose result is not a dim at all (an
address rather than a position). Each gets its own sentence.

THE SECOND DEFECT IN THIS FILE IS STILL LIVE, AND IT IS NOT HERE ANY MORE --
this file now passes, which is why it has moved out of `unsupported/`. In a
kernel with enough lane pressure to make fit_layout_to_lanes widen that column --
add the matmul and this file becomes dl/quant_int4_matmul -- fit_layout_to_lanes's unit-axis
widening re-points the operand at a `[32, 16]` buffer and writes

    AffineMap.get_identity(len(wide))

onto its slot. `len(wide)` is 2 and the generic's iteration space is 3, so the
op is left holding a 2-dim map beside a 3-dim one and stops verifying:

    'linalg.generic' op inferred input/output operand #1 has shape's dimension
    #1 to be 16, but found 1

which is rule 4's measured example, from an operand re-pointed without its map's
arity going with it -- or, in three of them, without the OUTS type going with
it: `ins(%widened : 8x8) outs(%narrow : 1x8)`, same defect, and the verifier
picks whichever half it notices first. dl/quant_int4_matmul still fails on it;
the list below is where it bites, and none of it is this file's shape.

EIGHT KERNELS DIE ON THAT MESSAGE, all eight from fit_layout_to_lanes. Verified by
running the stage in process and calling `module.operation.verify()` after each
pass: valid through bufferize, invalid after fit_layout_to_lanes, every time.

    dl/chunked_prefill_attn   dl/gptq_actorder_gather   dl/quant_int4_matmul
    dl/deform_conv2d          dl/moe_align_block_size   dl/varlen_packed_attn
    dl/mxfp4_shift_scale_matmul                         dl/nf4_codebook_dequant

The last of those was invisible: it was reported as `KILLED by signal 11
(SIGSEGV) -- no diagnostic was printed`, and the crash only stopped happening
once widen_broadcast_loads stopped widening a row it should have declined. A
ninth, dl/ssd_chunk, had the same message from that same defect. That pass is
now DELETED -- it decided a banking fact five passes above anything that had
chosen a lane axis -- and both crashes are gone with it; the row is staged by
fit_layout_to_lanes._clone_loaded_row_wide, which mints a buffer instead of
retyping one. ops_widen_row_narrow_reader and ops_widen_row_outs_reader survive
as the shapes, and say what they pin now.

torch equivalent: the int4 nibble unpack, ((b >> (k%2)*4) & 0xF) - 8.

Pins vectorize banking a broadcast whose indexing map DROPS the dim rather than pinning it to 0 -- the same statement as the pinned spelling, and the one -linalg-fold-unit-extent-dims writes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N = 32, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_dropped_dim_broadcast_kernel(wq_ptr, o_ptr, K: tl.constexpr,
                                           N: tl.constexpr):
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        byte = tl.load(wq_ptr + (rk[:, None] // 2) * N + rn[None, :])
        w = ((byte >> ((rk[:, None] % 2) * 4)) & 0xF) - 8
        tl.store(o_ptr + rk[:, None] * N + rn[None, :], w)

    return tile_dropped_dim_broadcast_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"wq": torch.randint(0, 255, ((K // 2) * N,), generator=g,
                                dtype=torch.int32)}


def reference(inputs):
    import torch
    b = inputs["wq"].reshape(K // 2, N)
    rk = torch.arange(K).reshape(K, 1)
    rows = (rk // 2).expand(K, N)
    cols = torch.arange(N).reshape(1, N).expand(K, N)
    byte = b[rows, cols]
    w = ((byte >> ((rk % 2) * 4)) & 0xF) - 8
    return {"o": w.reshape(-1).to(torch.int32)}


SPEC = KernelSpec(
    name="tile_dropped_dim_broadcast_kernel", kernel=kernel,
    signature={"wq_ptr": "*i32", "o_ptr": "*i32",
               "K": "constexpr", "N": "constexpr"},
    constexprs={"K": K, "N": N},
    args=[Arg("wq", IN, "int32", ((K // 2) * N,)),
          Arg("o", OUT, "int32", (K * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
