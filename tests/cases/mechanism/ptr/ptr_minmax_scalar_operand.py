"""min/max against a RUNTIME bound -- and only the min side can be proved.

Split out of the old tests/cases/ptr_reject.py.  Four loads in one file, one
per arm, because they differ ONLY in which operand carries the scalar (and in
min vs max): same pointer, same rank, same everything else.  Reading them as
four labelled lines side by side is the point.

WHAT IT PINS.  `visitOperandClamp` keeps a clamp only where it can PROVE the
clamp never fires, and THE TWO SIDES NEED DIFFERENT PROOFS ABOUT THE BOUND:

    min(v, c) is v      where v never REACHES c   -> a LOWER bound on c is enough
    max(v, c) is v      where v never FALLS to c  -> needs an UPPER bound on c

Both bounds here are runtime, and a program id has a lower bound (0) and no
upper one.  So the two minima keep their stride and the two maxima do not:

    hi = pid + 4096     lower bound 4096, and 4096 > 11*127 = 1397
    lo = pid - 1        no upper bound, however obviously it is below every lane

    load                address              answer
    minsi lhs-scalar    minimum(hi, r*3)     stride 3
    minsi rhs-scalar    minimum(r*5, hi)     stride 5
    maxsi lhs-scalar    maximum(lo, r*7)     index table
    maxsi rhs-scalar    maximum(r*11, lo)    index table

THE UNIQUE-MULTIPLIER TRICK -- keep 3 / 5 / 7 / 11.  All four loads read the
same `in_ptr`, so the only thing in the IR that says which load is which is the
multiplier its tensor operand carries, used nowhere else in the kernel.  Don't
renumber them.

WHAT IT USED TO PIN, and why the file is worth keeping.  This was a
SILENT-MISCOMPILE shape: each arm picked one operand's state and returned
success, so `tl.minimum(limit, offs)` -- a real bounds guard -- came out as an
unclamped `tts.make_tptr` whose stride was the tensor operand's multiplier and
whose offset was 0.  The four distinct strides were the proof that four
different arms each threw a clamp away.  With grid=(1,) every clamp here is a
no-op on every lane, which is why the wrong address still gave the right number
and reference() can be an honest torch expression.

minsi/maxsi against a RUNTIME bound on either side: a program id has a lower bound and no upper one, so the two minima keep their stride and the two maxima take an index table. The primes 3/5/7/11 are what identify the four arms in the IR.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 2048
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_minmax_scalar_operand_kernel(in_ptr, o_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        hi = pid + 4096          # runtime scalar, above every lane
        lo = pid - 1             # runtime scalar, below every lane

        # min, SCALAR ON THE LEFT -> state = rhs, so stride 3 survives
        a = tl.load(in_ptr + tl.minimum(hi, r * 3))
        # min, SCALAR ON THE RIGHT -> state = lhs, so stride 5 survives
        b = tl.load(in_ptr + tl.minimum(r * 5, hi))
        # max, SCALAR ON THE LEFT -> state = rhs, so stride 7 survives
        c = tl.load(in_ptr + tl.maximum(lo, r * 7))
        # max, SCALAR ON THE RIGHT -> state = lhs, so stride 11 survives
        d = tl.load(in_ptr + tl.maximum(r * 11, lo))

        tl.store(o_ptr + r, a + b + c + d)

    return ptr_minmax_scalar_operand_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["in"]
    r = torch.arange(BLOCK)
    hi = torch.tensor(4096)
    lo = torch.tensor(-1)
    # every clamp below is a no-op by construction, so this matches both the
    # source and the (clamp-free) code the compiler actually emits
    return {"o": x[torch.minimum(hi, r * 3)] + x[torch.minimum(r * 5, hi)]
                 + x[torch.maximum(lo, r * 7)] + x[torch.maximum(r * 11, lo)]}


SPEC = KernelSpec(
    name="ptr_minmax_scalar_operand_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    # THE NUMBER CANNOT SAY IT: every clamp here is a no-op, so a dropped one
    # answers the same. The rungs are the claim -- two strides survive their
    # proof and two clamps become one address per element.
    expect_ir={"03-ttshared.mlir": {"tts.make_gather_scatter_tptr": 2}},
    reference=reference,
    make_inputs=make_inputs,
)
