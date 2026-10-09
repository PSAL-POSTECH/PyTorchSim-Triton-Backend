"""A transpose stuck between two reassociation groups, taken by the DESCRIPTOR.

WHAT IT PINS. `absorb_layout_ops`'s reassociation arm -- `_absorb_reassoc` and the
`_split_strides` half of the pair. The shape is the one the pass could not
finish for three kernels: expand_shape, transpose, collapse_shape, with the
permutation moving axes that live INSIDE one reassociation group. Both crossings
refuse and both are right to: on the narrow side there is no answer at all,
because the destination index is not affine in the source index and no pair of
strides expresses it.

THE ANSWER IS ON THE WIDE SIDE. Move the reassociation instead of the
permutation: split the MVIN's descriptor so the DMA stages a rank-4 tile, and
the transpose that was stuck between two groups is a plain permutation of four
axes that the existing `dram_stride` sink takes. Then the collapse on the far
side goes the same way, into the MVOUT.

WHAT IT PINS IS A COUNT, NOT A VERDICT, and that has to be said plainly: this
kernel PASSES with the arm and PASSES without it. Nothing refuses the transpose
downstream -- `fit_layout_to_lanes` materialises it, which is a correct answer and
an expensive one. So the regression this file guards is the number of DMAs, read
off `08-passes/10-psto-place-lane-banks.mlir`:

    no arm              8 transfers   the transpose is a DRAM round trip, and
                                      the collapse is a SECOND one
    arm, transposes     4 transfers   the transpose is gone; the collapse still
    only                              copies the tile out and straight back in
    arm, every          2 transfers   one MVIN, one MVOUT, dram_stride
    reassociation                     [4, 128, 32, 1] and [32, 16, 4, 1]

and the third row is the whole argument for absorbing a reassociation that no
transpose is standing next to: the round trip's own MVOUT descriptor is
[32, 16, 4, 1], byte for byte the one `_split_strides` computes, so the copy was
writing the tile out in the layout it was already in.

Spike answers max_abs_err = 0 in all three, which is the point of rtol=atol=0
here: this kernel has no arithmetic in it at all, so any difference is a byte in
the wrong place rather than a rounding one.

WHY THREE SUB-AXES AND NOT TWO. `_split_strides` computes the sub-strides as a
running product read from the inside out. A group of two cannot tell that apart
from any other rule that happens to put the larger stride first -- 64 splitting
into 2x4x8 can, because the middle stride is the only one that has to be a
product of two things.

WHAT THIS DOES NOT PIN. `_merge_strides` SUCCEEDING. It is reached here and it
refuses, which is how the collapse ends up going forwards at all: asked to merge
[4, 128, 32] over extents [8, 2, 4] it finds 4 != 128 * 2 and says so, the
backward plan comes back None, and `_reshape_fwd` into the MVOUT is what lands.
A descriptor those axes ARE one contiguous run of is the case that would take
the other branch, and no kernel in the suite has one -- said rather than left to
look covered.

A permutation inside a reassociation group is absorbed by splitting the DMA descriptor, not by crossing the reassociation.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

G0, G1, G2, W = 2, 4, 8, 4
M = G0 * G1 * G2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reassoc_into_descriptor_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                           W: tl.constexpr, G0: tl.constexpr,
                                           G1: tl.constexpr, G2: tl.constexpr):
        rm = tl.arange(0, M)
        rw = tl.arange(0, W)
        x = tl.load(x_ptr + rm[:, None] * W + rw[None, :])        # (M, W)
        # One axis split three ways, then the innermost group axis pulled to the
        # front: the permutation moves axes that are all inside group 0, so
        # neither crossing of the reassociation can carry it.
        g = tl.reshape(x, (G0, G1, G2, W))
        t = tl.trans(g, 2, 0, 1, 3)                                # (G2,G0,G1,W)
        o = tl.reshape(t, (M, W))
        tl.store(o_ptr + rm[:, None] * W + rw[None, :], o)

    return ops_reassoc_into_descriptor_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * W, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(G0, G1, G2, W)
    return {"o": x.permute(2, 0, 1, 3).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ops_reassoc_into_descriptor_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "W": "constexpr", "G0": "constexpr", "G1": "constexpr",
               "G2": "constexpr"},
    constexprs={"M": M, "W": W, "G0": G0, "G1": G1, "G2": G2},
    args=[Arg("x", IN, "float32", (M * W,)),
          Arg("o", OUT, "float32", (M * W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
