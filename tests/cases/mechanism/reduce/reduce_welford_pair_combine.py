"""A `tl.reduce` whose combiner is not one of the known ones -- Welford's shape.

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER REDUCE IN THIS DIRECTORY
`reduce_row_sum` and friends reduce ONE tensor with ONE known operator, and
the linalg lowering's ReduceConverter matches exactly that: a body holding a single op
from a fixed list (addf, maxf, ...). Inductor's LOOPED variance does not have
that shape. It carries a (mean, m2, weight) triple and combines two triples with
a six-argument body -- `welford_combine`. No single op names it, so the pattern
does not match and the `tt.reduce` survives the whole linalg lowering.

What it survives INTO is this backend, which has no lowering for `tt.reduce` at
all: bufferize refuses it by dialect ("no lowering for 'tt.reduce'").

This kernel is the two-operand form of that -- the smallest body that is not one
op over one tensor. It reduces a pair (value, weight) with a four-argument
combiner, which fails the single-op match for the same reason the six-argument
one does, without needing the whole variance recurrence to read it.

WHAT FIXES IT, AND WHERE
Not here. `the linalg lowering` 46d70d7 adds `ReduceGeneralConverter` at benefit 0:
below every named converter, so a body that IS one known op still takes the fast
path, and anything else is lowered generically -- seed the accumulator from
slice 0 along the reduced axis with one `tensor.extract_slice` per operand, then
fold the remaining slices through the body. That is true for anyone lowering
this IR, not just for this backend, so it belongs up there (rule 11) and this
side needs no pass change at all.

WHAT PROVES IT FIRED
  03-linalg.mlir has no `tt.reduce`, and instead holds two
  `tensor.extract_slice` per operand feeding a two-result `linalg.reduce`.
  Against a triton-shared built before 46d70d7 the same file still has the
  `tt.reduce`, and the run stops in bufferize with "no lowering for 'tt.reduce'" --
  the dialect refusal, which is the case that refusal was written for.

THE SECOND HALF WAS OURS AND IS FIXED. With 46d70d7 the kernel compiled the whole
way to Spike and the values were WRONG (measured: o max_abs_err 10.06, p 14.99,
against a torch reference that is a plain column sum). Neither the upstream
lowering nor `_bank_reduce_general` was at fault, and the near-misses that would
have named a small slip in either were all measured and rejected -- it was not
the sum with row 0 dropped (seed lost), not row 0 alone (fold lost), not the
other operand's sum (results crossed), not a doubled seed.

What it WAS: every one of the 128 outputs is the sum of ROW L of the tile
instead of column L, exact to 9.5e-07. Lane L folded sixteen consecutive
elements of its own bank, and its bank held a row -- because the buffer and the
op that reads it disagreed about which axis the lanes are:

    %alloc   = memref.alloc()  {torchsim.lane_axis = 0}   memref<16x128xf32>
    %subview = memref.subview %alloc[1, 0] [15, 128] [1, 1]
    linalg.reduce ins(%subview, ...) dimensions = [0]
                                 {torchsim.lane_axis = array<i64: 1, 1, 0, 0>}

The reduce is right: folding dim 0 of a tile banked on dim 0 would have to cross
lanes, so axis 1 is the only executable answer. The buffer kept axis 0, and that
is what place_in_spad transcribes into the symbol name and lower_to_dma into
the MVIN descriptor, so the DMA banked one way and vectorize addressed the
other. `memref.subview` had no rule in lib_axis_graph._maps_of and is not in
NO_EDGE, so the two ends were ORPHANED rather than related -- the exact
distinction relates_no_axes exists to make. Slices are new here because the
general lowering is: a reduce used to read its whole buffer, and now every
multi-operand one arrives behind two subviews.

Reduces over axis 0 rather than 1 on purpose: the multi-operand seed-and-fold is
what is under test, and axis 0 is the axis the linalg lowering normalises reductions
to, so the lowering is read without absorb_transpose moving first.

Multi-operand tl.reduce with a body that is not one known op. Covers the linalg lowering's ReduceGeneralConverter (crashes in bufferize against a linalg lowering without it) AND the axis edge across the slices it emits -- without that edge the buffer and the reduce that reads it pick different lane axes and every column sum is a row sum.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 16, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _pair_combine(v0, w0, v1, w1):
        # Two tensors in, two out -- the shape ReduceConverter cannot name. A
        # weighted running sum, which is one step of Welford without the m2 term.
        w = w0 + w1
        return v0 + v1, w

    @triton.jit
    def reduce_welford_pair_combine_kernel(x_ptr, w_ptr, o_ptr, p_ptr,
                                           M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        off = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + off)
        w = tl.load(w_ptr + off)
        v_out, w_out = tl.reduce((x, w), 0, _pair_combine)
        tl.store(o_ptr + rn, v_out)
        tl.store(p_ptr + rn, w_out)

    return reduce_welford_pair_combine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32),
            "w": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sum(dim=0), "p": inputs["w"].sum(dim=0)}


SPEC = KernelSpec(
    name="reduce_welford_pair_combine_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "p_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("w", IN, "float32", (M, N)),
          Arg("o", OUT, "float32", (N,)), Arg("p", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
