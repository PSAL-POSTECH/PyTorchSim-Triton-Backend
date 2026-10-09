"""A bare `tl.reshape` between two shapes with no common grouping, read by a
reduce. It REFUSED until convert_unsupported_ops learned to re-spell one as a
collapse and an expand, and the history below is why it was aimed here.

RETIRED AS A FORBIDS PIN, 2026-09-17. Its subject, absorb_layout_ops, is gone,
and no stage-4 pass sets FORBIDS_DEFAULT = False any more -- every declaration
is enforced on every run, which is the state the scaffolding below existed to
reach. WHAT IS LOST IS THE MECHANISM'S OWN PIN: with no pass leaving a forbidden
op, `_check_forbids` never fires, so a break in that chain is silent again. That
wants a harness kernel of its own and this one can no longer be it.


THIS FILE IS ABOUT THE CHECK, NOT ABOUT THE RESHAPE. Every stage-4 pass declares
FORBIDS -- the ops that may not exist in the module it hands on -- and
lowering/passes/transform.py:_check_forbids reads that declaration against the pass's own output
the moment it returns. This is the kernel that fails when any link in that chain
is missing.

WHAT BREAKS IT, in order:

    pytorchsim_triton_compiler/lowering/passes/p06_absorb_layout_ops.py       FORBIDS, the layout family
    pytorchsim_triton_compiler/lowering/passes/__init__.py                forbids / forbids_enabled / forbidden
    pytorchsim_triton_compiler/lowering/passes/transform.py:_check_forbids   called from run()'s dump()
    SPEC.extra["forbids"] below            the per-kernel override

Take any one away and this kernel COMPILES AND PRODUCES THE RIGHT NUMBERS --
measured, all eight stages green, max_abs_err 2.4e-07 -- so run.py reports
EXPECTED A REFUSAL, BUT EVERY STAGE PASSED and the sweep goes red. That is the
whole evidence: without the mechanism there is nothing here to fail on, which is
precisely the state the mechanism exists to end.

WHY IT NEEDS THE OVERRIDE AT ALL. absorb_layout_ops declares the whole layout family
-- transpose, expand_shape, collapse_shape, reshape, each in its tensor and its
memref spelling -- and does not yet ENFORCE it: 30 of those ops are still
standing across the suite, in nine kernels. Switching it on for everyone today would turn kernels
that produce correct numbers red for a reason nobody can act on yet, so the
declaration is written now and enforced per kernel, here.

THE SHAPE, AND IT IS THE THIRD ONE THIS FILE HAS HAD. It was written as
`expand_shape -> transpose -> collapse_shape` and that sandwich learned to
absorb. It was then written as two bare `tl.reshape`s with a multiply between
them, and that learned to absorb too: absorb_layout_ops restates an ELEMENTWISE-IDENTITY
generic on the other shape whole -- every operand, the result, the maps and the
iterators together -- so a reshape with only elementwise work around it has a
neighbour that takes it. Both times the kernel reported its own refusal as
stale, which is the signal this file is built to give, and this is that signal
being acted on for the third time: re-pointed at a shape the pass still leaves,
not answered by weakening absorb_layout_ops's FORBIDS (rule 5).

THIS FILE IS NOT RE-AIMED AGAIN; IT IS RETIRED WHEN THE COUNT REACHES ZERO.
Aiming a kernel at a shape absorb_layout_ops cannot absorb is a losing strategy and not a
maintenance cost -- that shape IS absorb_layout_ops's work, so every time the work lands the
kernel ages out, three times so far. The declaration is scaffolding for the
period while FORBIDS_DEFAULT is False; once it is True the WHOLE SUITE is the
pin, because any pass that leaves a forbidden op turns whichever kernel hit it
red with the pass and the op named. See the FORBIDS comment in
pytorchsim_triton_compiler/lowering/passes/p06_absorb_layout_ops.py.

WHAT IS LEFT STANDING HERE, AND THE REASON CHANGED UNDER IT. It used to say the PRODUCER was half the refusal:
the tile is loaded with a row stride of 16 out of a row 8 wide, so its 32
elements are not one contiguous run of DRAM and `_merge_strides` had nothing to
return. That was a stronger question than the machine asks. A descriptor is
sizes and strides and expresses any strided access for free; what costs anything
here is elements moving between LANES, and a relinearisation moves none. absorb_layout_ops now
carries a third candidate -- the COMMON REFINEMENT, `4x8` and `8x4` both split
into `4x2x4` -- so both descriptors restate by splitting, which never refuses.
ops/ops_reshape_refinement is this kernel with a `tl.store` where the `tl.sum`
is, and it absorbs completely.

So what is left standing here is the READER, alone, and that is now the whole of
absorb_layout_ops's only refusal:

    the op            `tl.reshape` between 4x8 and 8x4. Neither side has a
                      common grouping, so upstream cannot write a reassociation
                      and emits a bare `tensor.reshape` with a shape OPERAND.
                      absorb_layout_ops reads it as a WEIGHT PER RESULT AXIS -- rowmajor of
                      the result shape -- and the axes do not land one-to-one,
                      so the only reading left is the linear one.

    the reader        `linalg.reduce`, whose `dimensions` is a list of axis
                      NAMES. A name is not linear in the index, and unlike a
                      per-axis LIST it cannot be split either: an axis that
                      became two has no name to give. That is the same wall a
                      generic's `indexing_maps` and a slice's three arrays hit,
                      and the reason `_restate` refuses them in one place
                      instead of three. It is also why the refinement does not
                      rescue this one: the refinement makes both sides SPLIT,
                      and a split is exactly what a name cannot take.

    the producer      absorbs now, and is no longer part of the refusal. Kept in
                      the kernel because the strided load is what makes the
                      refinement the only plan on that side, which is what
                      ops_reshape_refinement pins.

TWO OF THEM, ONE NAME, AND THAT IS DELIBERATE: the report has to say
`tensor.reshape x2`, so the COUNT is checked and not only the name. They come
from two separate loads rather than two views of one tile, because two reshapes
of the SAME buffer trip a downstream aliasing bug in place_in_spad ("'linalg.reduce'
op init dimensions [16] doesn't match input dimensions after reduction [8]") and
this kernel would then be red for something that is not the check.

THE SAME SHAPE IN THE WILD is dl/hadamard_rotate and sort/sort_bitonic, which
leave twelve of these between them: `insert_slice -> tensor.reshape ->
extract_slice`, where the producer and the readers are slices instead of a DMA
and a reduce. Their wall is NOT quite this one, and the difference is measured
rather than assumed -- instrumenting `_plan_into` shows the slice arm answering
YES every time an exact move reaches it and never being ASKED under a split,
because the branch above it refuses everything with no axis correspondence in
one place. Unlike `dimensions`, a slice's three arrays CAN take a split (each
entry becomes a box on the sub-axes) -- what it needs is the slice's `src` to
move with them, and that is a group reaching back into the generics that fill
it. This file uses the DMA-and-reduce spelling because it is four lines and it
verifies end to end, which hadamard_rotate does not.

WHEN THIS LANDS, THIS FILE GOES RED, AND THAT IS THE END OF IT. run.py reports
the refusal as stale; the answer is to DELETE this file, not to re-point it and
not to weaken absorb_layout_ops's FORBIDS (rule 5). The count in absorb_layout_ops goes from 30 to 28 the
moment these two absorb, and to 0 it turns FORBIDS_DEFAULT on and the suite
takes over the job.

torch equivalent: two strided 4x8 windows, each read back as 8x4 and row-summed

A bare reshape between 4x8 and 8x4 -- no common grouping, so upstream writes no reassociation -- read by a reduce whose dimensions is a list of axis numbers. It compiles now: convert_unsupported_ops re-spells it as a collapse and an expand, both of which state their own correspondence.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 4, 8
S = 16          # the DRAM row stride. S > C is what makes each tile a strided
                # read rather than a line -- no longer the refusal (the
                # refinement takes it), but what makes the plan the refined one.


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def forbids_transpose_below_absorb_kernel(x_ptr, o_ptr, p_ptr,
                                              R: tl.constexpr,
                                              C: tl.constexpr,
                                              S: tl.constexpr):
        r = tl.arange(0, R)
        c = tl.arange(0, C)
        # Two 4x8 windows of a 4x16 buffer, so neither tile is one contiguous
        # run and neither descriptor can take a reshape.
        a = tl.load(x_ptr + r[:, None] * S + c[None, :])
        b = tl.load(x_ptr + C + r[:, None] * S + c[None, :])
        # 4x8 -> 8x4: no grouping expresses it, so each is a bare tensor.reshape,
        # and a reduce is the only thing reading it.
        tl.store(o_ptr + tl.arange(0, C), tl.sum(tl.reshape(a, (C, R)), axis=1))
        tl.store(p_ptr + tl.arange(0, C), tl.sum(tl.reshape(b, (C, R)), axis=1))

    return forbids_transpose_below_absorb_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    m = inputs["x"].reshape(R, S)
    a = m[:, :C].contiguous()
    b = m[:, C:2 * C].contiguous()
    return {"o": a.reshape(C, R).sum(dim=1).contiguous(),
            "p": b.reshape(C, R).sum(dim=1).contiguous()}


SPEC = KernelSpec(
    name="forbids_transpose_below_absorb_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "p_ptr": "*fp32",
               "R": "constexpr", "C": "constexpr", "S": "constexpr"},
    constexprs={"R": R, "C": C, "S": S},
    args=[Arg("x", IN, "float32", (R * S,)),
          Arg("o", OUT, "float32", (C,)),
          Arg("p", OUT, "float32", (C,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
