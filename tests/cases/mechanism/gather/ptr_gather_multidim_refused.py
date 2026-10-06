"""ONE non-contiguous dimension, or nothing: the four ways to break that rule.

tts.make_gather_scatter_tptr has exactly one gather_scatter_dim. Every rebuild
helper in PtrAnalysis therefore starts by asserting the state has at most one
dim with extent > 1, and gives up otherwise. This file is the four refusals,
side by side, because the rule is the same rule each time and the interesting
part is which CHECK catches each shape.

NOT A FAILING KERNEL. All four loads are refused by --triton-to-structured and
then picked up by --triton-to-unstructured as plain tts.gather ops, so the
pipeline succeeds end to end. "Refused" here means "no tts.make_gather_scatter_
tptr was built", not "the compile broke".

COVERS (4)   lib/AnalysisStructured/PtrAnalysis.cpp
  rebuild-unsupported-multidim-fail   PtrAnalysis.cpp:212
      rebuildAsUnsupportedOp: isNotSingleDim(operand) -- the loaded index is
      itself 8x2, so two dims have extent > 1 and there is no single dim to
      hand the tensor to.  [load 1]
  rebuild-gs-multidim-fail            PtrAnalysis.cpp:247
      rebuildAsGatherScatter: the same test, reached from the MODULO path
      instead: `(i[:, None]*16 + arange(16)[None, :]) % 64` is an 8x16
      remainder, so the rebuild that would have cleared the modulo cannot pick
      a dim either.  [load 2]
  addstate-different-nonstructured-dim-fail   PtrAnalysis.cpp:295
      addState with BOTH sides unstructured and
      lhs.getNonStructuredDim() != rhs.getNonStructuredDim(): two independent
      loaded index tensors, one on each axis. Each side is individually legal;
      their sum is not.  [load 3]
  addptr-unstruct-reanalysis-fail     PtrAnalysis.cpp:1423
      the caller: the structured pass already decided the address is not
      structured, so rewriteAddptrOp re-ran visitOperandAddptr with
      isAnalysisingUnstructured = true -- and THAT failed. This is the arm that
      turns any of the three failures above into "leave the tt.addptr alone".
      [loads 1-3]

WHAT PROVES IT
  triton-shared-opt --triton-to-structured --debug-only=triton-ptr-analysis
  prints, once per refused load:
      remark: PtrAnalysis: Failed to analyze ptr of tt.addptr for unstructured
              state
      remark: PtrAnalysis: Failed to rewrite LoadOp
  and for load 3 the more specific
      remark: PtrAnalysis: do not support adding two pointer states that have
              different non-continuous dimension
  In 01-structured.mlir all three addresses are still raw Triton:
      %25 = tt.addptr %24, %23 : tensor<8x2x!tt.ptr<f32>>, tensor<8x2xi32>
      %26 = tt.load %25 : tensor<8x2x!tt.ptr<f32>>
  and the module holds exactly ONE tts.make_gather_scatter_tptr:
      %38 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 1]
            gather_scatter_dim: 0 gather_scatter_offset: %7,
            strides: [%c16_3, 0], offsets: [0, 0]
  -- the dead 8x1 intermediate for load 3's `i8[:, None] * 16`, built before the
  `+ j16[None, :]` that killed it. It has no consumer. Three tt.load survive
  next to it; that ratio (3 raw loads : 1 dead pointer : 0 live gather pointers)
  is the observable. By 02-unstructured.mlir each load has become
      %7  = tts.gather %arg0[%6]  : (<f32>, tensor<8x2xi32>)  -> tensor<8x2xf32>
      %15 = tts.gather %arg0[%14] : (<f32>, tensor<8x16xi32>) -> tensor<8x16xf32>
      %19 = tts.gather %arg0[%18] : (<f32>, tensor<8x16xi32>) -> tensor<8x16xf32>
  with the full N-D offset tensor kept, which is precisely the capability
  tts.make_gather_scatter_tptr lacks. The two STORES are unaffected and stay
  structured tts.store -- only the addresses that needed two gather dims fell
  back.

WHY IT MATTERS
  The fallback is not free: a make_gather_scatter_tptr lowers to an scf.for of
  memref.copy over whole ROWS, while a tts.gather lowers to a linalg.generic
  doing one tensor.extract per ELEMENT. Load 1 versus
  tests/cases/ptr_gather_offset_collapse.py (an 8x1 index, accepted) is a
  one-character difference in the source and that difference in the output.

torch equivalent: compile-time probe (`reference` returns zeros); the file
exists for the refusals, and load 2's modulo makes the addresses uninteresting
numerically.

Three shapes with more than one non-contiguous dim, all refused by PtrAnalysis and all recovered as element-wise tts.gather.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NI = 1024
M = 64
BM = 8
BN = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_multidim_refused_kernel(x_ptr, i_ptr, o_ptr, o2_ptr,
                                           M: tl.constexpr, BM: tl.constexpr,
                                           BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        r2 = tl.arange(0, 2)
        i8 = tl.load(i_ptr + rm)                 # 8 indices, axis 0
        j16 = tl.load(i_ptr + 16 + rn)           # 16 indices, axis 1

        # 1. the loaded index tensor is itself 8x2 -- two dims of extent > 1
        i2d = tl.load(i_ptr + rm[:, None] * 2 + r2[None, :])
        a = tl.load(x_ptr + i2d)
        # 2. a modulo whose operand is an 8x16 gather expression
        b = tl.load(x_ptr + ((i8[:, None] * BN + rn[None, :]) % M))
        # 3. two independent gathers, one per axis
        c = tl.load(x_ptr + i8[:, None] * BN + j16[None, :])

        tl.store(o_ptr + rm[:, None] * 2 + r2[None, :], a)
        tl.store(o2_ptr + rm[:, None] * BN + rn[None, :], b + c)

    return ptr_gather_multidim_refused_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, M, (NI,), generator=g, dtype=torch.int32)}


def reference(inputs):
    """The three gathers, as written.

    This was zeros() -- a placeholder from when the file could not be compiled
    and the numbers were never reached. PtrAnalysis declining these shapes is
    what the file is here for; it still lowers, through tts.gather.
    """
    import torch
    x, i = inputs["x"], inputs["i"].long()
    rm, rn, r2 = torch.arange(BM), torch.arange(BN), torch.arange(2)
    i8, j16 = i[:BM], i[16:16 + BN]
    a = x[i[rm[:, None] * 2 + r2[None, :]]]                    # 2-D index tensor
    b = x[(i8[:, None] * BN + rn[None, :]) % M]                # modulo of a gather
    c = x[i8[:, None] * BN + j16[None, :]]                     # one gather per axis
    return {"o": a.reshape(-1), "o2": (b + c).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_multidim_refused_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "o2_ptr": "*fp32", "M": "constexpr", "BM": "constexpr",
               "BN": "constexpr"},
    constexprs={"M": M, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("o", OUT, "float32", (BM * 2,)),
          Arg("o2", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
