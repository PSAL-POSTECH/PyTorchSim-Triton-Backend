"""tl.cumsum: CumSumConverter, and every shape of the ttx.cumsum op it emits.

Split out of the old tests/cases/ops.py (which had merged the old
tts_cumsum.py into it).  One file, because seven of these eight ids are ODS
facts about a single op and the only way to check an ODS fact is to look at
several instances of the op side by side -- rank 1 f32, rank 1 i32, rank 2
axis 1 -- plus one scan that must NOT become a ttx.cumsum at all.

tt.scan is a different op from tt.reduce with a different converter, which is
why this is not part of tests/cases/ops_reduce.py.

COVERS -- 8 branch ids.

  scan-cumsum-ttx   lib/Conversion/TritonArithToLinalg/
                    TritonArithToLinalg.cpp:80   (triton-shared WORKING TREE)
      CumSumConverter on triton::ScanOp.  Requires isCumSum()
      (ConversionPatterns.hpp:2082-2109): the scan region body has exactly ONE
      op, that op is arith::AddFOp or arith::AddIOp, its result is the
      tt.scan.return operand, and its operand set equals the block-argument set.
      Then a rank/axis guard at ConversionPatterns.hpp:2122-2128.  This is the
      ONLY pattern in the whole file that emits a TritonTilingExt (ttx.*) op.

  and seven ODS ids on the op it emits, all in
  include/triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtOps.td.
  These are TableGen, not code: gdb has nothing to break on, so every one of
  them is verified by reading the emitted op, quoted below.

  ttx-cumsum-only-op                  :205  TritonTilingExt_CumSumOp is the ONLY
        concrete op in the entire ttx dialect.  Nothing on it is optional --
        no Optional<> operands, no optional attributes.
  ttx-cumsum-from-scan-pattern-match  :205  the isCumSum() gate above, seen from
        the op's side: a non-add scan produces NO ttx op and leaves a raw
        tt.scan region behind.
  ttx-cumsum-rank1-axis0              :209  UI32Attr:$axis == 0 with rank 1.
        CumSumOp::verify (CumSum.cpp:63-70) enforces rank in {1,2} AND
        axis == rank-1; getLoopIteratorTypes (CumSum.cpp:89-94) then yields zero
        parallel + one reduction iterator.
  ttx-cumsum-rank2-axis1              :209  axis == 1 with rank 2 -- the only
        legal 2-D case.  axis == 0 on a rank-2 tensor is rejected twice over, by
        the verifier at CumSum.cpp:68-70 and by the pattern's own guard.
  ttx-cumsum-tensor-operands          :207  verify() accepts RankedTensorType OR
        MemRefType; the conversion pattern always builds the TENSOR form, with a
        tensor.empty feeding outs (ConversionPatterns.hpp:2131-2135).
  ttx-cumsum-float-vs-int             :206  Variadic<AnyType>:$inputs element
        type is float (addf body) or integer (addi body).  The op carries no
        signedness or kind attribute, so the accumulate kind is READ OFF the
        element type and nothing else.
  ttx-cumsum-variadic-but-arity-one   :206  $inputs/$outputs are declared
        Variadic with AttrSizedOperandSegments, but the only builder
        (lines 216-223, skipDefaultBuilders = 1) takes exactly one input and one
        output and the accessors hard-index [0] (lines 230-236).

WHAT PROVES IT FIRED   work/ops_cumsum_kernel/triton-shared-detail/

  01-structured.mlir has four tt.scan regions, three with an arith.addf/addi
  body and one with arith.mulf.  03-arith-linalg.mlir has exactly THREE
  ttx.cumsum -- so scan-cumsum-ttx fired three times and refused once:

      %11 = tensor.empty() : tensor<32xf32>
      %12 = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>}
              ins(%7 : tensor<32xf32>) outs(%11 : tensor<32xf32>)
              -> tensor<32xf32>
      %14 = tensor.empty() : tensor<32xi32>
      %15 = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>}
              ins(%5 : tensor<32xi32>) outs(%14 : tensor<32xi32>)
              -> tensor<32xi32>
      %17 = tensor.empty() : tensor<8x32xf32>
      %18 = ttx.cumsum {axis = 1 : ui32, operandSegmentSizes = array<i32: 1, 1>}
              ins(%9 : tensor<8x32xf32>) outs(%17 : tensor<8x32xf32>)
              -> tensor<8x32xf32>

  (that linalg-style ins/outs spelling is the op's custom printer,
  TritonTilingExtOps.td:89.)  Read straight off those three ops:

    * ttx-cumsum-rank1-axis0    -- axis = 0 : ui32 on tensor<32x...>, twice
    * ttx-cumsum-rank2-axis1    -- axis = 1 : ui32 on tensor<8x32xf32>
    * ttx-cumsum-float-vs-int   -- tensor<32xf32> vs tensor<32xi32>, same op,
                                   no attribute distinguishing them
    * ttx-cumsum-tensor-operands-- RankedTensorType in and out, and each `outs`
                                   is a tensor.empty (%11, %14, %17)
    * ttx-cumsum-variadic-but-arity-one -- operandSegmentSizes = array<i32: 1, 1>
                                   on all three, never anything else, and one
                                   ins / one outs every time
    * ttx-cumsum-only-op        -- `ttx.` appears nowhere else in the module;
                                   grep finds exactly these three lines.

  None of the seven is measurable with a debugger: their census entries point at
  TritonTilingExtOps.td, and TableGen op definitions have no line to break on.
  Reading the emitted op IS the measurement for them.

  ttx-cumsum-from-scan-pattern-match is the negative half, and it is the reason
  the cumprod is in this kernel.  Its tt.scan has a mulf body, isCumSum()
  refuses it, and it survives the pass as a raw tt.scan region -- sitting in
  03-arith-linalg.mlir next to the three converted ones:

      %22 = "tt.scan"(%21) <{axis = 0 : i32, reverse = false}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %23 = arith.mulf %arg10, %arg11 : f32
        tt.scan.return %23 : f32
      }) : (tensor<32xf32>) -> tensor<32xf32>

  The i32 cumsum's input is worth one more look: it is `%5`, the linalg.generic
  that materialises tl.arange.  An arange used only as an address is folded into
  a tts.make_tptr by pass 1, so feeding `cols` to a cumsum is what keeps it a
  live value -- and it is the only source of an INTEGER cumsum here.

  The census entry for scan-cumsum-ttx predicts that an unmatched tt.scan
  "remains illegal -> conversion failure".  MEASURED, it does not: tt.scan has
  no ConversionTarget entry in this pass, so the unconverted op is simply legal
  and the pass reports success.  Recorded here because it contradicts the
  census text, and because it is what makes the negative witness cheap -- the
  cumprod costs nothing but a store.

WHY run.py STOPS AT STAGE 2 -- expected, and NOT a failure of the linalg lowering
  All six passes of the linalg lowering succeed (tools/lower_steps.py emits every stage
  with no error, warning or remark, and 98-full.mlir is byte-identical to
  99-composite.mlir).  What fails is the step after them: ttx.cumsum is
  the linalg lowering's own TritonTilingExt dialect, and the backend MLIR that run.py
  hands 03-linalg.mlir to does not have it registered --

      [2/7] ttshared  FAIL
        error: "-":25:10: Dialect `ttx' not found for custom op 'ttx.cumsum'

  which is the same shape of gap as tests/cases/ops_tensor_of_ptrs.py's
  tptr.type_offset, and is inherited unchanged from the old ops.py.  Check this
  file with

      python tools/lower_steps.py tests/cases/ops_cumsum.py --debug

  The reference below is kept exact so the file stays usable if that gap closes.

WHY THE CUMPROD INPUT IS NEAR 1.0
  A running product of 32 standard normals underflows fast, and triton's scan
  associates differently from torch's sequential cumprod, so the two would
  disagree well outside rtol.  `x*0.1 + 1.0` keeps every partial product near 1
  where the two orders agree.  This matters only for the numeric reference; the
  branch fires either way.

ARITHMETIC
  Four scans of the same 32 or 8x32 elements. Exact except as noted above.

SCOPE
  IR-census kernel.  scan-cumsum-ttx was confirmed by a gdb line hit
  (the branch census); the seven ttx-cumsum-* ids are TableGen and have
  no line to break on, so reading the three emitted ops above IS their
  measurement.  This file additionally stops at run.py stage 2 -- see above.

POSITIVE KERNEL: every pass of the linalg lowering succeeds, including on the refused
scan; run.py stage 2 fails on the backend's side, by design -- see above.

CumSumConverter plus every ODS shape of ttx.cumsum (rank-1 f32, rank-1 i32, rank-2 axis-1) and one tl.cumprod that must NOT match. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 8
K = 32
XN = M * K       # 256
ON = 512
QN = K
PN = K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_cumsum_kernel(x_ptr, o_ptr, q_ptr, p_ptr,
                          M: tl.constexpr, K: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, K)
        a = tl.load(x_ptr + cols)
        x2 = tl.load(x_ptr + rows[:, None] * K + cols[None, :])

        # rank 1, axis 0, FLOAT -- scan body is arith.addf
        tl.store(o_ptr + cols, tl.cumsum(a, axis=0))
        # rank 1, axis 0, INT   -- scan body is arith.addi, same op, no attr
        tl.store(q_ptr + cols, tl.cumsum(cols, axis=0))
        # rank 2, axis 1        -- the only legal 2-D case
        tl.store(o_ptr + K + rows[:, None] * K + cols[None, :],
                 tl.cumsum(x2, axis=1))
        # NOT a cumsum: a mulf scan body, which isCumSum() refuses.  The raw
        # tt.scan surviving pass 4 is the witness -- see the module docstring.
        tl.store(p_ptr + cols, tl.cumprod(a * 0.1 + 1.0, axis=0))

    return ops_cumsum_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(XN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    a = x[:K]
    x2 = x.reshape(M, K)
    o = torch.zeros(ON, dtype=torch.float32)
    o[0:K] = a.cumsum(0)
    o[K:K + M * K] = x2.cumsum(dim=1).reshape(-1)
    q = torch.cumsum(torch.arange(K, dtype=torch.int32), 0).to(torch.int32)
    p = (a * 0.1 + 1.0).cumprod(0)
    return {"o": o, "q": q, "p": p}


SPEC = KernelSpec(
    name="ops_cumsum_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "q_ptr": "*i32",
               "p_ptr": "*fp32", "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "float32", (XN,)),
          Arg("o", OUT, "float32", (ON,)), Arg("q", OUT, "int32", (QN,)),
          Arg("p", OUT, "float32", (PN,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
