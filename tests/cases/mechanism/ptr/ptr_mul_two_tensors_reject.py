"""`x_ptr + r * r` -- the address PtrAnalysis gives up on, and the six branches
that one failure walks through.

Split out of the old tests/cases/ptr_structured.py section A. NEGATIVE
KERNEL: this file exists to be REFUSED. mulState can only multiply two pointer
states when one of them is effectively a scalar; `r * r` is two tensor states,
so the address cannot be described by (offset, stride, size) at all. The
failure then propagates outward through four more branches -- the addptr, the
walk dispatch, and finally the load and the store that use the un-mapped
pointer.

"Works" means: --triton-to-structured emits four remarks, leaves the tt.addptr
and its users alone, and the later TritonToUnstructured pass turns them into a
tts.gather / tts.scatter. The pipeline as a whole still SUCCEEDS -- a refused
pointer is a fallback, not an error.

COVERS (6)   source root /workspace/triton_shared/, worktree lines

  mulstate-no-scalar-fail        lib/AnalysisStructured/PtrAnalysis.cpp:551
      mulState: !lhsState.scalar && !rhsState.scalar ->
      "PtrAnalysis: only support multiplying pointer states when one of them
       represent a scalar".
  visitaddptr-offset-visit-fail  PtrAnalysis.cpp:1133
      visitOperandAddptr: the ptr operand parsed fine, visiting the OFFSET
      operand is what failed. Anything already emitted for the ptr side stays
      in the IR.
  addptr-visit-fail              PtrAnalysis.cpp:1400
      rewriteAddptrOp's early return: no tts.make_tptr, and knownPtrs / ptrMap
      are left without an entry for this addptr.
  dispatch-addptr-fail-remark    PtrAnalysis.cpp:1975
      the walk's AddPtrOp arm: "PtrAnalysis: Failed to rewrite AddPtrOp",
      then advance() so nested ops are still visited.
  load-ptr-not-mapped            PtrAnalysis.cpp:1774
      rewriteLoadOp: ptrMap.lookupOrNull(op.getPtr()) == null ->
      "pointer is not replace with tts.make_tptr so loadOp cannot be
       rewritten". This is the DOWNSTREAM consequence of the addptr failure,
      not a second independent decision.
  store-ptr-not-mapped           PtrAnalysis.cpp:1917
      the same for tt.store. Kept in this file rather than a separate one
      because it needs the very same refused address to exist -- splitting the
      store off would mean duplicating the whole failure.

WHAT PROVES IT FIRED
  01-structured.mlir still contains the raw triton ops -- there is no
  tts.make_tptr anywhere in the module:
      %2 = arith.muli %1, %1 : tensor<32xi32>
      %3 = tt.addptr %0, %2 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %4 = tt.load %3 : tensor<32x!tt.ptr<f32>>
      tt.store %6, %4 : tensor<32x!tt.ptr<f32>>
  tools/lower_steps.py --debug prints, in this order:
      remark: PtrAnalysis: only support multiplying pointer states when one of
              them represent a scalar
      remark: PtrAnalysis: Failed to rewrite AddPtrOp
      remark: PtrAnalysis: pointer is not replace with tts.make_tptr so loadOp
              cannot be rewritten
      remark: PtrAnalysis: Failed to rewrite LoadOp
      ... and the storeOp pair after it.
  02-unstructured.mlir then shows the fallback that makes the module still
  compile:
      %2 = "tts.gather"(%arg0, %1) ... -> tensor<32xf32>
      "tts.scatter"(%2, %arg1, %1) ...
  i.e. the quadratic index tensor survives as a real gather index.

torch equivalent: o[i*i] = x[i*i] for i in 0..31, zero elsewhere.

NEGATIVE: PtrAnalysis must refuse this address. Four remarks, zero tts.make_tptr, and the fallback gather/scatter is the observable. Numerically runnable through the fallback path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024        # r*r reaches 31*31 = 961
BLOCK = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_mul_two_tensors_reject_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        bad = tl.load(x_ptr + r * r)     # mulState: neither side is a scalar
        tl.store(o_ptr + r * r, bad)     # same refused address on the store

    return ptr_mul_two_tensors_reject_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(N, dtype=torch.float32)
    idx = torch.arange(BLOCK) * torch.arange(BLOCK)
    o[idx] = x[idx]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_mul_two_tensors_reject_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
