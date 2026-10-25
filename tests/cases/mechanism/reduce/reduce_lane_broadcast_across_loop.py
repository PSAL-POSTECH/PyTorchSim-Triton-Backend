"""A lane-broadcast operand inside a reduction loop is wrong after the first pass.

An operand loaded as `c_ptr + r0_index` is masked on the REDUCTION axis only and
is constant across the lane axis -- one row, splatted to every lane. Add it to a
tile inside a loop whose accumulator crosses the back edge and the reduction
comes out wrong.

IT USED TO COMPILE CLEAN AND ANSWER WRONG, with nothing refused and nothing
warned -- the failure rule 12 exists for. It PASSES now, at 9.53674e-06, and
not because of anything in this file's own line of work: the fix arrived with
develop-e2e-gpt2's wide-tile gather and lane-index work. What is below is kept
because it is the diagnosis that was paid for, and because the shape it
describes is the one a regression would come back through.

A GUARD FOR IT WAS TRIED AND TAKEN BACK OUT, and that is worth more than the
guard was. `_bank_reduce` refuses "reduce over dim N of a tile banked on that
same axis" by the OP's stamp; making it ask the BUFFER's stamp too catches this
kernel. It also refuses convnextv2, which is CORRECT -- its 59th kernel is
    linalg.reduce ins(memref<2x1024xf32>) outs(memref<2xf32>) dimensions=[1]
with the op's stamp 0 and the buffer's 1, and the model runs end to end at
1.7881e-06 when it is allowed through. So a stamp mismatch is REAL here and
HARMLESS there, and it is not the discriminator. Adding conditions until only
this kernel is caught would be rule 5's case rule. Left red instead.

FOUND BY e2e (rule 13a). `tests/models/BERT/test_bert.py --preset small` stops
at kernel 9, `..._add_addmm_native_layer_norm_view_9` -- the residual add plus
the second LayerNorm, buf21, 24575/24576 elements over tolerance. The lane
broadcast there is LayerNorm's own bias/weight row (`in_ptr0 + r0_1`), which
every token in the tile reads the same way. It is not exotic: every LayerNorm
and every bias-add in a transformer has one.

TWO INGREDIENTS, EACH PROVED BY LEAVE-ONE-OUT against this exact file:

    the broadcast removed, loop still 6 passes      PASSES
    the broadcast kept, R0_BLOCK 1024 so ONE pass   PASSES
    both                                            FAILS   <- this file

THE CAUSE, read off vectorize's own decisions (TORCHSIM_COMPILE_DBG instrumentation, reverted).
The pass count is a CORRELATE, not the mechanism; what actually moves is which
axis gets the lanes.

  1. `c` is constant along the token axis. Put the lanes on that axis (dim 0)
     and every lane needs the whole row, which takes a replicating transfer
     this operand does not get -- so vectorize REFUSES, correctly, with
     "operand [1, N] spans iteration dims [1] of a tile ... banked on dim 0,
     and it was staged one element per lane". That guard was added for
     mask_gs_dim_mismatch and it is doing its job.
  2. So the lane axis goes on dim 1 instead -- the reduction axis, which `c`
     DOES span, so the guard is silent. Measured, same file, one knob:

         no broadcast     every op and every buffer   lane_dim 0   PASSES
         broadcast        the generics move to        lane_dim 1   FAILS

  3. The generics move; the `linalg.fill` that zeroes their loop-carried
     accumulator does NOT. It is still decided at lane_dim 0. So the buffer
     the loop carries enters banked on dim 0 and is yielded banked on dim 1:

         %6 = get_global @buf1 ... {pytorchsim.lane_axis = 0}    iter_args(%arg10 = %6)
         %1 = get_global @buf6 ... {pytorchsim.lane_axis = 1}    scf.yield %1

     Both are `memref<128x128xf32, 1>`. The lane axis lives in an attribute on
     the global, not in the type, so the verifier sees one type, agrees, and
     nothing is refused.
  4. The reduce that follows then has to cross lanes, and the answer is wrong
     with no diagnostic.

  Two ways out, and they are at different levels: give the broadcast operand
  the replicating transfer step 1 asks for, so dim 0 stays viable and every op
  agrees; or make the lane axis something a loop carry is CHECKED on, so step 3
  refuses instead of miscompiling. The first removes the cause, the second
  removes the silence.

  NEITHER IS DONE. The second was tried -- `_bank_reduce` asking the buffer's
  stamp as well as the op's -- and it refuses convnextv2, which is correct; see
  the head of this file. The first is still the real repair, and it is a
  transfer decision, not a vectoriser one: vectorize's own refusal says so, "a
  decision for the transfer, not for this pass".

  A DEAD END, RECORDED SO NOBODY RE-WALKS IT. The first repair tried was a
  loop-carry check in fit_layout_to_lanes: `iter_args` enters on axis 0 and `scf.yield` leaves
  on axis 1, same `memref<128x128xf32, 1>` on both ends, invisible to the
  verifier. It is real and it is NOT the defect -- `dl/attention.py` carries the
  identical mismatch and answers correctly, and with the reduce replaced by a
  raw dump of the accumulator THIS kernel answers correctly too (9.5e-07). The
  loop is fine; the fold is not.

RULED OUT, measured on the model's own kernel so nobody re-walks them:

    Welford's three-way carry   NOT it -- a plain `acc += v` fails identically
                                (243.398 abs err), and this file uses the sum
                                because it is the smaller statement
    depth per lane              NOT it -- R0_BLOCK 1024 is EIGHT deep per lane
                                and passes; R0_BLOCK 128 is one deep and fails.
                                R0_BLOCK moves the answer because it moves
                                which axis can take the lanes, not because of
                                depth and not because of the pass count
    a gather                    there is none. The model kernel's transfers are
                                all plain masked MVINs, no `indirect` anywhere
    a second 2-D operand        NOT it -- cutting the model kernel's other 2-D
                                input leaves it failing
    in-place read/write         NOT it -- the model kernel writes its input
                                buffer, but the FIRST output block is already
                                wrong, so the reduction is wrong before any
                                store could alias it

NOT THE SAME DEFECT as gather/gather_masked_deep_in_loop.py (develop-e2e-gpt2
82871d0), though both are reductions in a loop that come back wrong. That one
needs a GATHER, needs depth > 1 per lane, and passes at depth 1; this one has no
gather, fails at depth 1, and passes at depth 8. Fixing either, run the other.

BUILDING UP FOUND NOTHING -- this file is what DELETING from the model's kernel
left. Four hand-written kernels of the same shape (masked on both axes, looped,
Welford or sum, no broadcast) all passed. What worked was running the model's
own generated `_spec.py` with a numpy reference bolted on and cutting one load
at a time. Bisect down from the failure, not up from a guess.

WHAT MAKES THIS PASS: staging a broadcast operand with a replicating transfer,
so the lane axis can stay on the axis the operand is constant along and every op
in the loop agrees. Until then fit_layout_to_lanes stops it, which is the honest failure.

A row splatted to every lane, added inside a reduction loop that runs more than once. Reduced from BERT-small kernel 9 by deleting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# BERT-small's own shape: 32 tokens under a 128-lane block, so 96 lanes are
# masked off, and 768 hidden reduced 128 at a time -- six passes.
ROWS = 32
N = 768
XBLOCK = 128
R0_BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_lane_broadcast_across_loop_kernel(
            x_ptr, c_ptr, o_ptr,
            xnumel: tl.constexpr, r0_numel: tl.constexpr,
            XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex

        acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            v = tl.load(x_ptr + (r0_index + r0_numel * x0), r0_mask & xmask,
                        other=0.0)
            # THE INGREDIENT: masked on r0 only, constant across the lanes.
            v = v + tl.load(c_ptr + (r0_index), r0_mask, other=0.0)
            acc = tl.where(r0_mask & xmask, acc + v, acc)

        tl.store(o_ptr + tl.arange(0, XBLOCK), tl.sum(acc, 1),
                 tl.arange(0, XBLOCK) < xnumel)

    return reduce_lane_broadcast_across_loop_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS * N, generator=g, dtype=torch.float32),
            "c": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    a = inputs["x"].reshape(ROWS, N).double() + inputs["c"].double()
    return {"o": a.sum(dim=1).float()}


SPEC = KernelSpec(
    name="reduce_lane_broadcast_across_loop_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "c_ptr": "*fp32", "o_ptr": "*fp32",
               "xnumel": "constexpr", "r0_numel": "constexpr",
               "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"},
    constexprs={"xnumel": ROWS, "r0_numel": N,
                "XBLOCK": XBLOCK, "R0_BLOCK": R0_BLOCK},
    args=[Arg("x", IN, "float32", (ROWS * N,)),
          Arg("c", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (ROWS,))],
    grid=(1, 1, 1),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3,
    atol=1e-3,
    expect="pass",
)
