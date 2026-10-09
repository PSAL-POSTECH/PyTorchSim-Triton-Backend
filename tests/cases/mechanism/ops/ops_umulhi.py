"""A linalg.generic that yields TWO values, and the store that only made one.

`tl.umulhi(a, b)` is the high half of the 64-bit product -- 32x32 does not fit
in 32, and this is the half that a division-by-multiplication, a hash, or a
fixed-point multiply wants.  The linalg lowering spells it exactly right:

    %5:2 = linalg.generic ins(%3, %3) outs(%3, %3) {
    ^bb0(%a: i32, %b: i32, %o0: i32, %o1: i32):
      %6:2 = arith.mului_extended %a, %b : i32
      linalg.yield %6#0, %6#1        // low, high
    }
    tts.store(%4, %5#1)              // the kernel reads #1, the HIGH half

TWO results, two outs, two yielded values.  vectorize took
`yield.operands[0]` and stored it to `memrefs[-1]` regardless of arity, so the
LOW half was written into the HIGH half's buffer.  Every element came out as the
low word of the product where the high word belonged -- measured on this shape,
128/128 elements, `got` equal to the low half to the bit.

NO DIAGNOSTIC ANYWHERE, and the evidence removed itself: with the high result
now unused, the canonicalize at the end of vectorize folded
arith.mului_extended to arith.muli, so the final IR read as though a plain
truncating multiply had been asked for.  Reading
08-passes/08-torchsim-vectorize.mlir gave no hint that a second result had
ever existed.

Nothing was missing from the target. RVV has the instruction and llc picks it
unaided -- `vmulhu.vv` for the high result, `vwmulu.vv` + `vnsrl.wi` for the low
one.  The gap was the arity of one store.

COVERS
  the multi-result store path in vectorize._vectorise (the loop that walks
  the generic's body and the AffineVectorStoreOp after it), which no other
  kernel in the suite reaches: every other elementwise body yields exactly one
  value.  arith.mului_extended is the only two-result op triton can emit.

WHAT PROVES IT FIRED   work/ops_umulhi_kernel/

  08-passes/08-torchsim-vectorize.mlir keeps the two-result op and stores BOTH:

      %low, %high = arith.mului_extended %3, %3 : vector<8xi32>
      affine.vector_store %low,  %1[%arg9] : memref<128xi32, 1>, vector<8xi32>
      affine.vector_store %high, %0[%arg9] : memref<128xi32, 1>, vector<8xi32>

  and 11-kernel.s contains vmulhu.vv.  Before the fix that file had
  `arith.muli` and a single store, and vmulhu.vv appeared nowhere.

  The low store is redundant here -- nothing reads %1 afterwards, and llc still
  computes the low half through vwmulu.vv/vnsrl.wi to satisfy it.  Left in
  rather than guessed at: at the memref level the generic has no results, so
  "this yielded value is dead" is not a question this pass can answer, and
  storing what the op says it stores is the only thing that is certainly right.

ARITHMETIC
  umulhi(n, n) with n in [2^20, 2^30).  Squaring is deliberate -- one input
  buffer, so bufferize is free to reuse it as an outs, which is what produced
  the aliased `ins(%1, %1) outs(%1, %0)` this store had to be correct under.
  The range makes the product genuinely exceed 32 bits (so the high half is
  non-zero and a low/high mix-up cannot pass) while each input stays valid as
  an i32 and unambiguous under umulhi's UNSIGNED reading.

POSITIVE KERNEL: every stage succeeds and the result matches torch.

Witness for the multi-result store in vectorize: a generic that yields (low, high) needs one store per yielded value, to its own outs buffer. Storing operands[0] into memrefs[-1] put the low half of umulhi where the high half was read, on every element, silently. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_umulhi_kernel(n_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        n = tl.load(n_ptr + r)
        tl.store(o_ptr + r, tl.umulhi(n, n))

    return ops_umulhi_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"n": torch.randint(1 << 20, 1 << 30, (N,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    # In int64 so the product is exact, then take the top word. int64 rather
    # than uint32 arithmetic because the inputs are positive and below 2^30, so
    # signed and unsigned readings agree and int64 cannot overflow.
    w = inputs["n"].to(torch.int64) * inputs["n"].to(torch.int64)
    return {"o": (w >> 32).to(torch.int32).contiguous()}


SPEC = KernelSpec(
    name="ops_umulhi_kernel",
    kernel=kernel,
    signature={"n_ptr": "*i32", "o_ptr": "*i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("n", IN, "int32", (N,)), Arg("o", OUT, "int32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
