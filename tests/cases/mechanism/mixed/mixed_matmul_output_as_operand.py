"""One matmul's OUTPUT is the next matmul's B, and both share A.

    t = tl.dot(x, y)          x is A: (m, k).  t is C: (m, n)
    u = tl.dot(x, t)          x is A again.    t is now B: (k, n)
    s = tl.sum(u, axis=0)     the fold refuses the class that cycle collapsed

WHAT THIS PINS -- A MATMUL FED ALONG THE WRONG AXIS, WITH NO DIAGNOSTIC
This kernel COMPILES CLEAN through every stage and returns numbers that are
wrong in every element (max_abs_err 468.552, 32/32 outside tolerance). Nothing
refuses it, which is the worst failure a layout pass has and is the reason this
file exists; the module docstring of p08_fit_layout_to_lanes says so in as many
words about this exact pass.

WHY. lib_axis_graph's node is (value, axis) and a class is "one logical axis".
Follow this program's correspondences round the cycle:

    t axis 0   is  m of the first matmul   (it is that matmul's C)
           and     k of the second         (it is that matmul's B)
    x axis 0   is  m of both               (it is A in both)
    x axis 1   is  k of both

so m == k and x's TWO axes are ONE class. `axis_of` then cannot tell x's axis 0
from its axis 1 -- they are the same class -- and hands back the lower one, so
the MUST "the matmul streams A along the axis B is held by", which names axis 1,
lands on axis 0. Measured stamp, `08-passes/10-psto-place-lane-banks.mlir`:

    linalg.matmul {torchsim.lane_axis = array<i64: 0, 1, 1>} ins(%alloc_2, %alloc_3)
    linalg.matmul {torchsim.lane_axis = array<i64: 0, 1, 1>} ins(%alloc_2, %alloc_5)

A on 0 while B is held on 1: the array is streamed A along its M and multiplies
the wrong rows. `_demands_met`, which is what `_resolve` accepts on, checks only
the NOT demands -- "NOT ONLY, deliberately", its own docstring -- so a MUST
landing wrong passes it silently and the pass reports success.

AND THE MEMORY SAYS EXACTLY THAT, which is how this diagnosis was confirmed
rather than inferred from the IR (rule 12). Reading `runtime/o.raw` back off
spike and comparing against numpy on the same `x.raw` / `y.raw`:

    vs  (x @ (x @ y)).sum(0)        max_abs_err 468.552   -- the kernel's meaning
    vs  (x @ (x @ y)).sum(1)        max_abs_err 401.0     -- not a transposed read
    vs  (x.T @ (x.T @ y)).sum(0)    max_abs_err 0.0001068 -- THIS is what ran

Both matmuls used A transposed and nothing else moved: not a layout that has to
be read differently, an arithmetically different program.

AXIS 0 AND NOT AXIS 1, AND THE AXIS IS THE WHOLE KERNEL. u's axis 0 is the
matmuls' m, which the cycle put in the same class as x's k -- the class a MUST
pins. Folding axis 1 instead refuses the OTHER class, every buffer still has an
answer, and the kernel gets as far as a vectorize refusal ("reduce over dim
1 of a tile banked on that same axis") instead of a wrong number. One character,
two completely different failures.

HOW BIG THIS IS, MEASURED OVER tests/cases/ AT 083800f. In the state
fit_layout_to_lanes SHIPS, at least one MUST demand lands off the axis it names in
NINE of the 277 functions that reach the pass. Eight already fail, and three of
those fail with a NUMERICAL MISMATCH -- dl/block_scale_2d_matmul,
dl/lightning_attn, dl/rel_shift_attn -- which is this defect at full size in
real kernels. This file is the smallest form of it.

WHY IT IS NOT SIMPLY FIXED BY ADDING MUST TO `_demands_met`, measured rather
than assumed. The ninth kernel is dl/stft_frame, which PASSES today with two
MUSTs landing wrong on square [64, 64] tiles; refusing on "a MUST landed wrong"
turns a working kernel into a refusal. Nor is "the three MUSTs of one matmul
disagree with each other" the discriminator: stft_frame has two such matmuls and
verifies, while dl/lightning_attn has none and answers wrongly. A test that
tells this case from stft_frame is what is missing, and it is missing because
`axis_of` has no way to say "this buffer's two axes are one class, so I cannot
answer which of them a demand meant" -- that is the fact to report, and it is a
capability the graph does not have.

WHAT BREAKS IT / WHAT FIXES IT
Fixed when the pass either finds x an arrangement in which A's K really is on
the lanes, or REFUSES. Broken again by anything that goes back to answering the
lane axis out of `mine` alone.

STATUS: FIXED, and by a smaller thing than this file predicted. It said the
answer was to widen `_demands_met` or to teach `axis_of` to say "I cannot tell
these two apart", and neither was needed -- a `Demand` has carried the AXIS it
is about all along and nothing read it. `axis_of` now prefers the axis a MUST
about THIS buffer names over `mine`'s, and the two can only disagree inside a
merged class, which is exactly this shape. See `_axis_in`.

    before   468.552 against the reference, 0.000107 against
             (x.T @ (x.T @ y)).sum(0) -- both matmuls ran with A transposed
    after    9.16e-05, all eight stages

The three dl kernels this file said to check with it: dl/rel_shift_attn and
dl/varlen_packed_attn went green with it; dl/block_scale_2d_matmul and
dl/lightning_attn did not, so their wrong numbers are something else and this
file is no longer their reproducer.

A matmul's C used as the next matmul's B while both share A puts m and k in ONE axis class, so axis_of cannot tell A's two axes apart and the array is fed A along its M. Compiles clean, every element wrong. Three dl kernels fail numerically for this reason.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mixed_matmul_output_as_operand_kernel(x_ptr, y_ptr, o_ptr,
                                              N: tl.constexpr):
        r = tl.arange(0, N)
        off = r[:, None] * N + r[None, :]
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        # t is the first matmul's C -- (m, n) -- and the second matmul's B --
        # (k, n). x is A in both. That cycle is the whole kernel: it makes m
        # and k one axis class, and x's two axes indistinguishable inside it.
        t = tl.dot(x, y)
        u = tl.dot(x, t)
        # Axis 0 is m, which is the class the cycle collapsed and a MUST pins.
        tl.store(o_ptr + r, tl.sum(u, axis=0))

    return mixed_matmul_output_as_operand_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N * N, generator=g, dtype=torch.float32),
            "y": torch.randn(N * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(N, N)
    y = inputs["y"].reshape(N, N)
    return {"o": (x @ (x @ y)).sum(dim=0).contiguous()}


SPEC = KernelSpec(
    name="mixed_matmul_output_as_operand_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[
        Arg("x", IN, "float32", (N * N,)),
        Arg("y", IN, "float32", (N * N,)),
        Arg("o", OUT, "float32", (N,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
