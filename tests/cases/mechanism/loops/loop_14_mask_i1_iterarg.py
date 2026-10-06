"""The MASK ITSELF is the iter-arg: the prepass skips i1, so there is no state to find.

parseLoopIterArg rejection #2 of 4, and the one that needs the least code. Every
other mask file carries an index tensor and compares it inside the loop; this
one carries the tensor<128xi1> boolean. PtrAnalysis's prepass only wraps values
that can take part in pointer arithmetic -- pointers and int/index tensors -- so
an i1 gets no tts.get_structured_state at all, and parseLoopIterArg's lookup
finds a plain block argument.

COVERS (1 id, lib/Analysis/MaskAnalysis.cpp)

  loopiter-initarg-not-getstate     :663 (HEAD :663)
      The matching init arg is not defined by a tts::GetStructuredStateOp, so
      control falls through to the `return failure()` at :691.

PROOF, from 01-structured.mlir -- this is the only scf.for in the whole loop set
with a SINGLE iter-arg and no index companions, and the mask block argument is
handed straight to the raw triton ops:

    %2 = arith.cmpi slt, %0, %1 : tensor<128xi32>
    %9 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %2)
             -> (tensor<128xi1>)  : i32 {
      %10 = tt.load %5, %arg5, %cst : tensor<128x!tt.ptr<f32>>
      tt.store %8, %10, %arg5 : tensor<128x!tt.ptr<f32>>
      %14 = arith.andi %arg5, %13 : tensor<128xi1>
      scf.yield %14 : tensor<128xi1>
    }

No get_structured_state anywhere in the module, no offset/stride iter-args, and
tt.load/tt.store keep their mask operand. Compare loop_13, where the op is
present but fails, and loop_11, where it succeeds and the loop signature grows
by (index, index).

The census's own trigger for this branch is a different shape (`idx =
tl.arange(0, BLOCK) * 2`, not recognised as structured). The i1 route is used
here instead because it also reaches the branch and cannot be confused with
anything else in the file -- there is exactly one iter-arg to reason about.

VERIFIED
  loopiter-initarg-not-getstate: 2 hits. This file reaches only 8 of the loop
  census's 41 ids, the fewest of any of them, which is the intended result:
  nothing about pointers happens here at all.

RUNNABLE. o[j] = x[j] for j < n - (iters - 1), zero elsewhere; the mask is
narrowed by one bound per trip and the last write wins.

POSITIVE lowering, NEGATIVE analysis: an i1 iter-arg is invisible to the tts.get_structured_state prepass, so MaskAnalysis bails at the 'init arg is not a GetStructuredStateOp' check.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
NBOUND = 96
ITERS = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop14_mask_i1_kernel(x_ptr, o_ptr, n, iters, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        msk = r < n                         # the MASK ITSELF is the iter-arg
        for it in range(iters):
            v = tl.load(x_ptr + r, mask=msk, other=0.0)
            tl.store(o_ptr + r, v, mask=msk)
            msk = msk & (r < n - it)

    return loop14_mask_i1_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(BLOCK, dtype=torch.float32)
    r = torch.arange(BLOCK)
    msk = r < NBOUND
    for it in range(ITERS):
        o[msk] = x[:BLOCK][msk]
        msk = msk & (r < NBOUND - it)
    return {"o": o}


SPEC = KernelSpec(
    name="loop14_mask_i1_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "iters": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("iters", "int32_t")],
           "scalar_values": {"n": NBOUND, "iters": ITERS}},
)
