"""An elementwise generic whose INPUT map is a transpose, on a banked tile.

Four lines of triton, and the reason they are here is what absorb_layout_ops does to
them. The linalg lowering normalises `tl.max(x, axis=1)` to transpose + reduce(dim 0),
and absorb_layout_ops folds that transpose into the DRAM strides of the MVIN -- so
the tile is STAGED [N, M], with the row axis on the LANES, which is what the
fold needs. The subtraction that broadcasts the row max back still wants it as
[M, N], and what is left of the transpose is one indexing map:

    indexing_maps  (d0,d1)->(d1,d0)   (d0,d1)->(d0,0)   (d0,d1)->(d0,d1)
    pytorchsim.lane_axis       1                  0                  0

Every one of those three maps sends d0 to its operand's banked axis, so every
lane reads nothing but its own bank. The only thing that differs between the
operands is the order of the indices INSIDE a lane, and banking does not depend
on that. This kernel pins that being enough.

`m` HAS TO BE READ TWICE and that is not decoration. With `m` used only through
`m[:, None]`, absorb_layout_ops has somewhere else to put the transpose -- it
expands the tile to [M, 1, N] and folds it into the reduce's `dimensions`
instead, every map stays the identity and nothing here is exercised (measured:
that version passes on the tree this was written against). The second read is
what makes the rank-1 `m` a value in its own right, so the fold's own layout is
what the elementwise op has to live with.

WHAT BREAKS IT

  - vectorize deciding "elementwise" by whether the maps are the IDENTITY
    instead of by which iteration dim feeds each operand's lane axis. That is
    what this failed on, in as many words:
        NotVectorisable: not identity-elementwise, and a scalar fallback would
        read the banked scratchpad linearly
    and dl/cross_entropy, sort/sort_topk, dl/rmsnorm_gated, dl/convnextv2_grn,
    dl/causal_conv1d_update and ptr/ptr_memref_masked_fill all failed with that
    same sentence.
  - comparing the RAW `pytorchsim.lane_axis` numbers across operands instead of the
    iteration dims they name. This op is stamped [1, 0, 0] and all three
    operands are banked on d0; the raw comparison reads that as two lane axes
    ("banked on different axes [0, 1]") and refuses an op with no conflict in it.
  - handing every operand the nest's loop variables positionally. The map has to
    be composed with the nest and the result reordered into the operand's own
    in-bank stride order. Get that wrong and this COMPILES and VERIFIES and
    answers with the wrong elements -- so M != N here, and make_inputs varies
    the value along BOTH axes with different periods, precisely so a swapped
    index cannot come out right by coincidence.

M is the lane count, so the depth loop is one trip and a bug in the lane index
hides; N is 64, so the in-lane extent is not the lane count and the two cannot
be confused. Both are dl/cross_entropy's, which is where this shape was found.

Pins vectorize accepting an elementwise generic whose ins map is a transpose: the condition is that every operand's lane axis is fed by the same ITERATION DIM, not that the maps are the identity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_transposed_ins_map_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                       N: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, N)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        m = tl.max(x, axis=1)                 # the fold that turns the staging
        e = x - m[:, None]                    # elementwise, between two folds
        s = m + tl.sum(e, axis=1)             # ...and m read again, as itself
        tl.store(o_ptr + rows, s)

    return tile_transposed_ins_map_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Different along the rows and along the columns, with different periods and
    # no symmetry: an index swapped anywhere in the chain moves the answer
    # instead of leaving it alone.
    x = torch.randn(M, N, generator=g, dtype=torch.float32)
    x += torch.arange(M, dtype=torch.float32)[:, None]
    x += 0.125 * torch.arange(N, dtype=torch.float32)[None, :]
    return {"x": x.reshape(-1).contiguous()}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, N)
    m = x.max(dim=1).values
    return {"o": (m + (x - m[:, None]).sum(dim=1)).contiguous()}


SPEC = KernelSpec(
    name="tile_transposed_ins_map_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
