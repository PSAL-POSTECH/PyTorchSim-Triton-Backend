"""The branches nothing else in the set reaches.

Added after measuring, not before: `the branch census` put every census
branch under a gdb breakpoint, and these came back with zero hits while 365 of
the other 370 targets were confirmed. Each needs one specific thing that no
existing kernel happens to do.

    gsv-splat-walk                TritonStructuredOps.cpp:89
        getScalarValue() walking THROUGH a triton::SplatOp: operand =
        op.getSrc(), loop again. Only happens when `other=` is a RUNTIME scalar,
        which the frontend splats to tensor shape. Every other masked load here
        passes `other=0.0`, a constant the frontend folds into a dense attribute,
        so the walk has nothing to step through.

    t2u-getbitwidth-scalar        TritonToUnstructuredPass.cpp:212
        getBitWidth() on a plain IntegerType. Needs a SCALAR i64 offset added to
        a pointer -- the tt.ptr_to_int/addptr offset chain then carries an i64,
        and TritonToUnstructured has to ask its width. All existing offsets are
        i32 or index.

torch equivalent: see `reference` -- three unrelated computations, deliberately
so, because the point is which compiler path each takes.

AND ONE THIS KERNEL STILL DOES NOT REACH
----------------------------------------
`visitoperand-blockarg-known` (PtrAnalysis.cpp:1378 worktree / 1373 HEAD,
`state = knownPtrs[operand];`). The nested loop in section 1 was written for it
and does not get there. What the measurements narrowed it to:

    :1365  `} else if (!operand.getDefiningOp())`   this kernel 2, ptr_loops 11
    :1366  `if (!knownPtrs.contains(operand))`      this kernel 0, ptr_loops  6
    :1367  `  return failure();`                    this kernel 0, ptr_loops  6
    :1373  `state = knownPtrs[operand];`            BOTH 0

So ptr_loops.py does reach the block-argument arm eleven times, and every single
time the value is NOT in knownPtrs -- it takes the failure path. Carrying the
POINTER through both loops rather than an offset tensor (which is what this
kernel now does) did not change that either: :1366 is not even reached, meaning
the operands visited at that point still have defining ops.

The census says the value must have been "populated during rewriteForOp of the
parent loop", so the shape needed is narrower than "any two nested loops". Left
uncovered deliberately rather than guessed at; the numbers above are the start
of the next attempt.

WHY THE OTHER TWO GAPS ARE NOT KERNEL PROBLEMS
-----------------------------------------------
Five branches came back uncovered. Two of them no kernel can fix:

  * `NOT-REGISTERED-monolithic-only` -- those patterns were registered only
    by the MONOLITHIC triton-to-linalg pass, which this backend never ran. They
    are GONE now: the pass was deleted, and the three converters it was the
    only registrant of went with it. No kernel could reach code nothing
    registers, and there is no longer any code there to reach.

  * `cmp-newdim-slt-ult` (MaskAnalysis.cpp:566) -- already covered by
    ptr_structured.py; the measurement could not see it. The census line is the
    `} else if (predicate == slt || ult) {` condition and carries no
    instruction, but :567 (the `|| ult` half) and :581-582 (the arm's body) are
    hit 1 and 25 times. the branch census credited a branch when either its
    own line or the next one runs, which is what makes this one -- and
    gsv-splat-walk and t2u-getbitwidth-scalar, whose census lines are the same
    `} else if` shape -- resolve.

Fills two measured coverage gaps: runtime-scalar `other` splat walk and scalar i64 offset width. The nested loop in section 1 targets a third (visitoperand-blockarg-known) and does NOT reach it -- see the module docstring for how far it got.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
OUTER = 2
INNER = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gap_fill_kernel(x_ptr, i_ptr, o_ptr, fill, bigoff,
                        BLOCK: tl.constexpr, OUTER: tl.constexpr,
                        INNER: tl.constexpr):
        r = tl.arange(0, BLOCK)

        # -- 1. nested loop carrying the POINTER itself, not an offset.
        #       knownPtrs is keyed on pointer-typed values, so the inner loop's
        #       iter-arg is only found there when the OUTER rewriteForOp put a
        #       pointer in it. Carrying `offs` instead takes the
        #       !knownPtrs.contains -> failure arm (measured: 6/6 on
        #       ptr_loops.py). ---------------------------------------------
        p = x_ptr + r
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for _i in range(0, OUTER):
            for _j in range(0, INNER):
                acc += tl.load(p)
                p = p + BLOCK

        # -- 2. `other` is a RUNTIME scalar, so the frontend splats it and
        #       getScalarValue has a SplatOp to walk through ----------------
        idx = tl.load(i_ptr + r)
        g = tl.load(x_ptr + idx, mask=r < BLOCK, other=fill)

        # -- 3. a SCALAR i64 offset, so getBitWidth sees a plain IntegerType
        s = tl.load(x_ptr + bigoff)

        tl.store(o_ptr + r, acc + g + s)

    return gap_fill_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(N, generator=g, dtype=torch.float32),
        "i": torch.randint(0, N, (BLOCK,), generator=g, dtype=torch.int32),
    }


def reference(inputs):
    import torch
    x, i = inputs["x"], inputs["i"]
    # 1. nested loop reads BLOCK-sized slices at 0, 128, 256, 384
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(OUTER * INNER):
        acc += x[k * BLOCK:(k + 1) * BLOCK]
    # 2. mask is r < BLOCK, i.e. always true -> `fill` is never selected
    gathered = x[i.long()]
    # 3. scalar load at bigoff
    scalar = x[BIGOFF]
    return {"o": acc + gathered + scalar}


FILL = 1.5
BIGOFF = 7


SPEC = KernelSpec(
    name="gap_fill_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "fill": "fp32", "bigoff": "i64",
               "BLOCK": "constexpr", "OUTER": "constexpr", "INNER": "constexpr"},
    constexprs={"BLOCK": BLOCK, "OUTER": OUTER, "INNER": INNER},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("i", IN, "int32", (BLOCK,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("fill", "float"), ("bigoff", "int64_t")],
           "scalar_values": {"fill": FILL, "bigoff": BIGOFF}},
)
