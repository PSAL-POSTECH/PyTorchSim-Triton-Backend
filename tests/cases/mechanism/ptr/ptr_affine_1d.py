"""The five ways a 1-D affine address is built, all of which PtrAnalysis accepts.

Split out of the old tests/cases/ptr_structured.py section A. This is the
base case every other ptr_*.py file is a deviation from: one tensor of
pointers, offsets that are pure index arithmetic, no mask, no modulo. Every
load here ends as a single `tts.make_tptr` + `tts.load`.

COVERS (14)   source root /workspace/triton_shared/, worktree lines

  visitoperand-ptr-blockarg-source   lib/AnalysisStructured/PtrAnalysis.cpp:1338
      the tt.ptr operand has no defining op (it is the kernel argument), so
      state.source = operand.  Load 1.
  makerange-state                    PtrAnalysis.cpp:986
      tl.arange -> offset = start, size = shape[0], stride = 1.  Load 1.
  splat-int-index-ptr                PtrAnalysis.cpp:1090
      tt.splat of the BASE POINTER contributes offset 0 / stride 0 per dim.
      Every load here goes through it.
  addptr-structured-make-tptr        PtrAnalysis.cpp:1407
      tensor-of-pointers addptr whose state.isStructured() -> emit tts.make_tptr.
  maketptr-op-emitted                PtrAnalysis.cpp:660
      PtrState::createTTSMakeTensorPtrOp itself (all sizes static attrs).
  load-no-mask                       PtrAnalysis.cpp:1795
      op.getMask() == null -> `dims` stays empty, MaskState never runs.
  load-success                       PtrAnalysis.cpp:1818
      tts::LoadOp::create + replaceAllUsesWith + erase.
  store-no-mask                      PtrAnalysis.cpp:1938
  store-success                      PtrAnalysis.cpp:1950
      the same two for the tt.store at the end.
  visitoperand-int-index-cast        PtrAnalysis.cpp:1303
      an i32 SCALAR operand (here `pid * BLOCK`) -> arith.index_cast, inserted
      after that operand's OWN defining op under an InsertionGuard, not down at
      the addptr.  Load 2.
  splat-scalar-offset0               PtrAnalysis.cpp:1104
      that scalar becomes offsets[0] of the state -> the dim-0 offset of
      tts.make_tptr.  Load 2.
  visitoperand-knownptrs-cache       PtrAnalysis.cpp:1297
      `q = p + BLOCK` where p was already analysed: the cached PtrState is
      reused and the offset subtree is NOT walked again.  Load 3.
  constsplat-state                   PtrAnalysis.cpp:1153
      a dense splat integer constant (the `+ 7`) becomes state.scalar and folds
      into offsets[0] -- no arith op survives for it.  Load 4.
  extsi-passthrough                  PtrAnalysis.cpp:892
      visitOperandExtSI delegates straight to its input, so an i64 offset
      tensor produces exactly the same state as the i32 one.  Load 5.

WHAT PROVES IT FIRED (work/ptr_affine_1d_kernel/triton-shared-detail/01-structured.mlir)

      %0 = tts.make_tptr %arg0 to sizes: [32], strides: [1],
           offsets: [0], shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
      %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>,
           static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f32>>) -> tensor<32xf32>

  * load 1 is the quote above -- offsets `[0]`, strides `[1]`, no mask operands.
  * load 2 is `offsets: [%7]` where
        %6 = arith.muli %0, %c32_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
    and the index_cast is emitted next to %6, above the pointer op.
  * load 3 is `offsets: [%c32]` with `%c32 = arith.constant 32 : index`: the
    `p + 32` folded into the CACHED state of p, and the arange arithmetic is
    NOT walked or re-emitted a second time.
  * load 4 is `offsets: [%c7]` for the same reason (const splat -> scalar).
  * load 5 is `offsets: [0], strides: [1]` identical to load 1 -- the
    `arith.extsi ... to tensor<32xi64>` is left dead in the TTIR and never
    reaches the pointer op.
  Total in 01-structured: 5 tts.make_tptr + 5 tts.load + 1 tts.store, no
  tt.addptr, no tts.gather, no remark of any kind.

torch equivalent: see reference() -- five overlapping slices of x, summed.

Base case of the affine family: 5 loads, 5 tts.make_tptr, no mask, no modulo, no gather. Numerically runnable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_affine_1d_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)

        a = tl.load(x_ptr + r)                    # 1 static offset 0
        b = tl.load(x_ptr + (pid * BLOCK + r))    # 2 dynamic dim-0 offset
        p = x_ptr + r
        q = p + BLOCK                             # 3 p is already in knownPtrs
        c = tl.load(q)
        g = tl.load(x_ptr + (r + 7))              # 4 constant splat, folded
        h = tl.load(x_ptr + r.to(tl.int64))       # 5 arith.extsi is transparent
        tl.store(o_ptr + r, a + b + c + g + h)

    return ptr_affine_1d_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    # grid is (1,) so pid == 0 and load 2 reads the same slice as load 1.
    return {"o": x[0:BLOCK] + x[0:BLOCK] + x[BLOCK:2 * BLOCK]
                 + x[7:7 + BLOCK] + x[0:BLOCK]}


SPEC = KernelSpec(
    name="ptr_affine_1d_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
