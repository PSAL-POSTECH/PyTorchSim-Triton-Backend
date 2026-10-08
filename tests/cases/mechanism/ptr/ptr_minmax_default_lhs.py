"""min/max where BOTH sides are gathers -- the clamp goes INTO the index.

Split out of the old tests/cases/ptr_reject.py.  Two loads in one file: min and
max reach the same answer by the same route, and separating them would mean two
files whose only difference is three characters.

WHAT IT PINS.  Neither operand is a number and neither is a readable range, so
`visitOperandClamp` takes its third answer for both: one address per element,
built over `min(ix, jx)` and over `max(jx*2, ix*2)` rather than over one side.

  and, from the `* 2` on the max line, mulState's unstructured arms stay
  covered: the per-dim loop that multiplies a NOT-structured dim's stride by
  the scalar, and the first pass that propagates offset and stride unchanged.
  (Those two are also claimed by the gather files; they are kept alive here
  because the monolithic ptr_reject.py reached them through exactly this line
  and dropping the `* 2` would be a silent regression.)

WHAT IT USED TO PIN.  Every earlier arm fell through to `state = lhsState`, so
the LHS won by position and the clamp went nowhere.  The experiment was to run
the same two tensors through min and max in OPPOSITE order and watch the answer
follow the left-hand side.  `ix < jx` everywhere, so both discarded operands
were no-ops and the number never said a thing.

min and max with both operands unstructured: the clamp reaches the index and the address is a table, run in opposite order so a left-hand-side answer would show. The `* 2` on the max line also keeps mulstate-unstruct-mul-stride and mulstate-unstruct-propagate alive.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_minmax_default_lhs_kernel(in_ptr, idx_ptr, o_ptr,
                                      BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        ix = tl.load(idx_ptr + r)
        jx = tl.load(idx_ptr + BLOCK + r)

        # min, BOTH sides unstructured -> fall through, keep the lhs (ix)
        p = tl.load(in_ptr + tl.minimum(ix, jx))
        # max, BOTH sides unstructured -> fall through, keep the lhs (jx).
        # The `* 2` is load-bearing, see MULTIPLYING A GATHER in the docstring.
        q = tl.load(in_ptr + tl.maximum(jx * 2, ix * 2))

        tl.store(o_ptr + r, p + q)

    return ptr_minmax_default_lhs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(BLOCK)
    idx = torch.cat([i, i + 1]).to(torch.int32)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": idx}


def reference(inputs):
    import torch
    x = inputs["in"]
    idx = inputs["idx"].long()
    ix, jx = idx[0:BLOCK], idx[BLOCK:2 * BLOCK]
    # ix < jx everywhere, so both discarded operands are no-ops
    return {"o": x[torch.minimum(ix, jx)] + x[torch.maximum(jx * 2, ix * 2)]}


SPEC = KernelSpec(
    name="ptr_minmax_default_lhs_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("idx", IN, "int32", (2 * BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    # THE NUMBER CANNOT SAY IT: ix < jx everywhere, so either side answers the
    # same. The claim is that both addresses are index tables built over the
    # clamp, not over whichever operand came first.
    expect_ir={"03-linalg.mlir": {"tts.make_gather_scatter_tptr": 2}},
    reference=reference,
    make_inputs=make_inputs,
)
