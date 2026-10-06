"""A pointer advanced by a SCALAR: the one address shape PtrAnalysis maps to
itself and refuses to rewrite.

Split out of the old tests/cases/ptr_structured.py section A. Everything
here hangs off `sp = x_ptr + pid`, a tt.addptr whose result is a plain
!tt.ptr<f32> rather than a tensor of pointers. rewriteAddptrOp does not create
any op for it -- it maps the result to ITSELF -- and both the scalar load and
the scalar store off it are then rejected with a remark. The rejection is the
point: those two tt ops survive --triton-to-structured untouched and are picked
up much later by TritonToUnstructured as tts.gather / tts.scatter with a SCALAR
offset.

NEGATIVE KERNEL, PARTLY. Loads 1/2 and store 1 are SUPPOSED to be refused by
PtrAnalysis; "works" means the two remarks below appear and the ops survive
into 02-unstructured.mlir. The pipeline as a whole still succeeds.

COVERS (12)   source root /workspace/triton_shared/, worktree lines

  addptr-scalar-selfmap        lib/AnalysisStructured/PtrAnalysis.cpp:1442
      !isa<RankedTensorType>(op.getPtr().getType()) -> ptrMap.map(res, res).
      No tts op is created and the tt.addptr stays in the output.
  addstate-rank0-scalar-propagate   PtrAnalysis.cpp:291
      addState with lhsState.getRank() == 0 and only one side carrying a
      scalar -> the scalar is propagated as-is, no arith.addi.  `x_ptr + pid`.
  addstate-two-scalars-addi         PtrAnalysis.cpp:287
      both sides have a scalar -> one arith.addi index combines them.
      `x_ptr + pid + n`, i.e. addptr(addptr(x_ptr, pid), n).
  load-scalar-ptr                   PtrAnalysis.cpp:1782
      rewriteLoadOp: pointee is not a ShapedType -> bail with
      "PtrAnalysis: scalar loadOp will not be rewritten".
  store-scalar-ptr                  PtrAnalysis.cpp:1924
      the same for tt.store: "scalar storeOp will not be rewritten".
  dispatch-load-fail-advance        PtrAnalysis.cpp:1998
      the TypeSwitch LoadOp arm taking WalkResult::advance() because
      rewriteLoadOp failed (on success it takes skip()).
  dispatch-store-fail-advance       PtrAnalysis.cpp:2006
      same for the StoreOp arm.
  visitoperand-ptr-addptr           PtrAnalysis.cpp:1321
      visitOperand on a SCALAR tt.ptr whose defining op is an AddPtrOp ->
      recurse into visitOperandAddptr.  See the note below: only load 4 can
      reach this, load 3 cannot.
  tts-load-scalar-not-rewritten     TritonStructuredDialect.td:318
  tts-store-scalar-not-rewritten    TritonStructuredDialect.td:347
      the ODS side of the same two rejections: TTS_LoadOp's result and
      TTS_StoreOp's $value are constrained to TT_Tensor, so a scalar load or
      store can never become a tts op at all.
  tts-gather-offset-scalar          TritonStructuredDialect.td:273
  tts-scatter-offset-scalar         TritonStructuredDialect.td:297
      where the two refused ops DO end up: TT_IntLike:$offset is a scalar i32,
      handled by ScalarLoadConverter / ScalarStoreConverter (memref.load /
      memref.store) rather than the linalg gather loop.

WHY LOAD 3 AND LOAD 4 ARE BOTH HERE (they look redundant, they are not)
  Load 3 `tl.load(sp + r)` is the census's own trigger for
  visitoperand-ptr-addptr and it does NOT reach it. rewriteAddptrOp stores a
  state for EVERY addptr it analyses, scalar ones included
  (`knownPtrs[op.getResult()] = state;` at :1404), and the scalar addptr is
  walked first, so by the time the tensor addptr visits it the knownPtrs cache
  at :1297 answers and returns before the pointer arm at :1318 is ever reached.
  Load 4 is the way in: the cache lookup is guarded by
  `!isAnalysisingUnstructured`, and the gather rebuild deliberately re-visits
  the whole address with that flag SET, so the second walk falls through to
  :1321 and recurses. Measured: :1321 is reached twice, and dropping load 4
  drops the branch. The old ptr_structured.py claimed this id with load 3 only
  and never actually covered it.

WHAT PROVES IT FIRED
  01-structured.mlir keeps the refused ops verbatim:
      %6  = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32   <- self-mapped, no tts op
      %7  = tt.load %6 : !tt.ptr<f32>                  <- still a tt.load
      tt.store %8, %7 : !tt.ptr<f32>                   <- still a tt.store
      %9  = arith.index_cast %arg3 : i32 to index
      %10 = arith.addi %4, %9 : index                  <- the two-scalar addi
  and with tools/lower_steps.py --debug the pass prints
      remark: PtrAnalysis: scalar loadOp will not be rewritten
      remark: PtrAnalysis: Failed to rewrite LoadOp
      remark: PtrAnalysis: scalar storeOp will not be rewritten
      remark: PtrAnalysis: Failed to rewrite StoreOp
  02-unstructured.mlir is where the two refused ops land:
      %2 = tts.gather %arg0[%0] : (<f32>, i32) -> f32
      tts.scatter %2 into %arg2[%0] : f32 into (<f32>, i32)
  -- scalar offset, scalar value, no tensor anywhere, which is exactly what
  tts-gather-offset-scalar / tts-scatter-offset-scalar name.
  Load 4's proof that the recursion at :1321 happened is in 01-structured:
      %24 = tts.make_gather_scatter_tptr %arg0 to sizes: [32]
            gather_scatter_dim: 0 gather_scatter_offset: %23, strides: [1],
            offsets: [0]
      %23 = arith.addi %22, %20   (%22 = splat of pid, %20 = the index load)
  The base is %arg0 and the scalar `pid` was FOLDED INTO the gather offset --
  the only way to get there is to have walked through the scalar tt.addptr.

torch equivalent: see reference().

Scalar-pointer family. The scalar load/store are DELIBERATELY refused by PtrAnalysis and re-emerge as tts.gather/tts.scatter with scalar offsets. Runtime i32 scalar n=3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32
NIDX = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_scalar_base_kernel(x_ptr, i_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)

        sp = x_ptr + pid          # scalar addptr: mapped to itself
        d = tl.load(sp)           # 1 SCALAR load  -> refused, becomes tts.gather
        tl.store(o_ptr + pid, d)  #   SCALAR store -> refused, becomes tts.scatter

        s2 = x_ptr + pid + n      # 2 two scalars -> arith.addi
        e = tl.load(s2)

        t = tl.load(sp + r)       # 3 tensor off the scalar base: knownPtrs cache
        i = tl.load(i_ptr + r)
        w = tl.load(sp + i)       # 4 gather off the scalar base: no cache

        tl.store(o_ptr + BLOCK + r, t + w + d + e)

    return ptr_scalar_base_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": torch.randint(0, N - BLOCK, (NIDX,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    x, i = inputs["x"], inputs["i"]
    o = torch.zeros(N, dtype=torch.float32)
    # grid is (1,) so pid == 0, and the runtime scalar n is 3.
    o[0] = x[0]
    o[BLOCK:2 * BLOCK] = x[0:BLOCK] + x[i.long()] + x[0] + x[3]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_scalar_base_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "n": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (NIDX,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"n": 3}},
)
