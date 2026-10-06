"""ReduceConverter and its four sub-branches, plus the two ArgMinMax converters.

Split out of the old tests/cases/ops.py.  One file for the whole reduction
family because ReduceConverter's sub-branches are chosen by the SHAPE of the
reduce, and the only way to read them is as a set of reduces over the same
tensor: rank 1 -> rank 0, rank 2 axis 0, rank 2 axis 1, rank 3 axis 1, plus one
whose element type is narrower than f32.  Separate files would be five copies of
the same eight lines differing in one integer.

tl.cumsum is NOT here -- tt.scan is a different op with a different converter
(CumSumConverter) and its own file, tests/cases/ops_cumsum.py.

COVERS -- 7 branch ids.  Line numbers are the triton-shared WORKING TREE,
source root /workspace/triton_shared/.  "CP.hpp" =
include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp.

  reduce-vector           lib/Conversion/TritonArithToLinalg/
                          TritonArithToLinalg.cpp:97
      ReduceConverter with isVectorReduce == (sourceType.getRank() == 1)
      (CP.hpp:1336, 1375-1385).  The body must contain exactly ONE op from the
      supported set {addf, addi, andi, maximumf, mulf, muli, maxnumf, minimumf,
      minnumf, minsi, minui, maxsi, maxui, ori, xori}.

  reduce-tensor           TritonArithToLinalg.cpp:97
      the rank > 1 arm (CP.hpp:1386-1394): tensor.empty + linalg.fill of the
      per-op identity constant + linalg.reduce, and NO trailing tensor.extract.

  reduce-transpose-rank0  TritonArithToLinalg.cpp:97
      sub-branch: transposeReduceToRank0 == true AND !isVectorReduce AND
      axis != 0 (CP.hpp:1343-1357).  An EXTRA tensor.empty + linalg.transpose is
      emitted BEFORE the reduce and the reduce's `dimensions` becomes [0].

  opt-transpose-reduce-to-rank0   include/triton-shared/Conversion/
                                  TritonArithToLinalg/Passes.td:26
      Option<"transposeReduceToRank0", default=true>, threaded into nothing but
      ReduceConverter's constructor.  It picks between the two transpose
      insertion policies at CP.hpp:1343 and CP.hpp:1358.  A .td option: gdb
      cannot break on it, so it is verified by the permutation attribute on the
      rank-3 reduce below -- see WHY THERE IS A RANK-3 REDUCE.

  reduce-f32-upconvert    TritonArithToLinalg.cpp:97
      sub-branch requiresF32Conversion(resType, rop) (CP.hpp:1278-1284, 1366):
      the result element type is a FloatType NARROWER than f32 and the reduction
      op is arith::AddFOp.

  argmax-linalg-reduce    TritonArithToLinalg.cpp:96
      ArgMaxConverter : ArgMinMaxBaseConverter (CP.hpp:1710-1737).  Matches only
      the exact 8-op body triton emits for tl.argmax: cmpf OEQ / cmpi SLT /
      andi / cmpf OGT / ori / select / select / tt.reduce.return.

  argmin-linalg-reduce    TritonArithToLinalg.cpp:95
      ArgMinConverter, the identical matcher with cmpf OLT (CP.hpp:1443-1707,
      1740-1761).

WHAT PROVES IT FIRED   work/ops_reduce_kernel/triton-shared-detail/

  reduce-vector -- and the give-away is `bufferization.alloc_tensor`, which the
  converter uses INSTEAD of tensor.empty specifically so it survives CSE, plus
  the trailing tensor.extract that pulls the rank-0 result back to a scalar:

      %10 = bufferization.alloc_tensor() : tensor<f32>
      %inserted = tensor.insert %cst_1 into %10[] : tensor<f32>
      %reduced = linalg.reduce ins(%3 : tensor<32xf32>)
                               outs(%inserted : tensor<f32>) dimensions = [0]
        (%in: f32, %init: f32) { %38 = arith.addf %in, %init : f32 ... }
      %extracted = tensor.extract %reduced[] : tensor<f32>

  reduce-tensor -- rank 2, axis 0.  tensor.empty + linalg.fill of the identity
  constant (0.0 for addf), dimensions = [0], and NO transpose and NO extract:

      %13 = tensor.empty() : tensor<32xf32>
      %14 = linalg.fill ins(%cst_1 : f32) outs(%13 : tensor<32xf32>)
      %reduced_5 = linalg.reduce ins(%9 : tensor<8x32xf32>)
                                 outs(%14 : tensor<32xf32>) dimensions = [0]

  reduce-transpose-rank0 -- rank 2, axis 1.  Same converter, same input (%9),
  and the only difference is the extra transpose and dimensions staying [0]:

      %15 = tensor.empty() : tensor<32x8xf32>
      %transposed = linalg.transpose ins(%9 : tensor<8x32xf32>)
                                     outs(%15 : tensor<32x8xf32>)
                                     permutation = [1, 0]
      %reduced_6 = linalg.reduce ins(%transposed : tensor<32x8xf32>)
                                 outs(%17 : tensor<8xf32>) dimensions = [0]

  reduce-f32-upconvert -- the f16 sum.  The accumulator and the alloc_tensor are
  built in f32 rather than in the source element type, the body extends each
  incoming element before adding, and the result is truncated back:

      %11 = bufferization.alloc_tensor() : tensor<f32>
      %inserted_2 = tensor.insert %cst_1 into %11[] : tensor<f32>
      %reduced_3 = linalg.reduce ins(%5 : tensor<32xf16>)
                                 outs(%inserted_2 : tensor<f32>) dimensions = [0]
        (%in: f16, %init: f32) {
          %38 = arith.extf %in : f16 to f32
          %39 = arith.addf %38, %init : f32
        }
      %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
      %12 = arith.truncf %extracted_4 : f32 to f16

  argmax / argmin -- TWO tensor.empty + linalg.fill pairs each, feeding a
  two-result linalg.reduce whose body is a clone of the tt.reduce body.  Note
  these do NOT take the transpose policy: both reduce `dimensions = [1]`
  directly, so the transpose above really is ReduceConverter's and not something
  every axis-1 reduction does.  The pair is told apart by the value
  accumulator's fill constant, exactly as the census says -- -inf for argmax,
  +inf for argmin -- and by the comparison in the body; the index accumulator is
  -1 in both:

      %cst_0 = arith.constant 0xFF800000 : f32     (-inf)   argmax
      %cst   = arith.constant 0x7F800000 : f32     (+inf)   argmin
      %c-1_i32 = arith.constant -1 : i32           (index, both)
      %reduced_7:2 = linalg.reduce ins(%9, %7 ...) outs(%19, %21 ...)
                                   dimensions = [1]
        ( ... %41 = arith.cmpf ogt, %in, %init : f32 ... )     argmax
      %reduced_8:2 = linalg.reduce ins(%9, %7 ...) outs(%23, %25 ...)
                                   dimensions = [1]
        ( ... %41 = arith.cmpf olt, %in, %init : f32 ... )     argmin

WHY THERE IS A RANK-3 REDUCE
  This is the only reason the rank-3 tensor exists, and deleting it silently
  costs a branch.  Every RANK-2 reduce emits byte-identical IR under both
  transposeReduceToRank0 policies -- transposing a 2-D tensor with permutation
  [1,0] and reducing dimension [0] is exactly what the legacy policy's
  dimensions=[1] does -- so a rank-2 reduce cannot distinguish them.  A rank-3
  reduce over a MIDDLE axis can: the true policy emits

      %transposed_15 = linalg.transpose ins(%expanded : tensor<8x4x8xf32>)
                                        outs(%26 : tensor<4x8x8xf32>)
                                        permutation = [1, 0, 2]
      %reduced_16 = linalg.reduce ins(%transposed_15 ...) dimensions = [0]

  where the legacy policy would emit no transpose at all and dimensions = [1].
  The permutation [1, 0, 2] in the IR above is the observable for
  opt-transpose-reduce-to-rank0, and there is no other one.

WHY THE f16 BUFFER IS SEPARATE
  requiresF32Conversion needs a FloatType narrower than f32 with an addf body,
  so the upconvert arm needs a genuinely f16 tensor.  float16 is a storable
  dtype in this harness, so it is a real kernel argument rather than a cast.

ARITHMETIC
  Sums and arg-extrema of x viewed as 8x32, all exact.  randn produces no ties,
  so argmax/argmin are unambiguous.

SCOPE
  IR-census kernel.  The six code ids were confirmed by gdb line hits
  (the branch census).  opt-transpose-reduce-to-rank0 is a TableGen
  option with no line to break on and is verified by the permutation = [1, 0, 2]
  quoted above -- that IS the measurement for a .td entry.  Like every file
  under tests/cases/ this does not reach run.py's numeric stages.

POSITIVE KERNEL: every triton-shared pass succeeds.

ReduceConverter's four sub-branches (vector / tensor / transpose-to-rank0 / f32 upconvert), the rank-3 middle-axis shape that makes the transposeReduceToRank0 option observable, and both ArgMinMax converters. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 8           # rows of the 2-D view
K = 32          # cols of the 2-D view, and the length of the 1-D vectors
XN = M * K      # 256
ON = 128
PN = M * M      # 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reduce_kernel(x_ptr, h_ptr, o_ptr, p_ptr,
                          M: tl.constexpr, K: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, K)
        a = tl.load(x_ptr + cols)                            # rank 1, f32
        h = tl.load(h_ptr + cols)                            # rank 1, f16
        x2 = tl.load(x_ptr + rows[:, None] * K + cols[None, :])   # rank 2

        s1 = tl.sum(a)              # rank 1 -> rank 0        reduce-vector
        sh = tl.sum(h)              # f16 body, f32 accum     reduce-f32-upconvert
        s0 = tl.sum(x2, axis=0)     # axis == 0, NO transpose reduce-tensor
        s2 = tl.sum(x2, axis=1)     # axis != 0, transposed   reduce-transpose-rank0
        am = tl.argmax(x2, axis=1)
        an = tl.argmin(x2, axis=1)

        # The one shape that separates the two transposeReduceToRank0 policies:
        # a rank-3 reduce over a MIDDLE axis. See the module docstring.
        x3 = tl.reshape(x2, (M, 4, K // 4))
        s3 = tl.sum(x3, axis=1)                              # -> (M, K//4)

        tl.store(o_ptr + cols, s0)                                   # [0:32]
        tl.store(o_ptr + K + rows, s2 + (am + an).to(tl.float32))    # [32:40]
        tl.store(o_ptr + K + M, s1 + sh.to(tl.float32))              # [40]
        tl.store(p_ptr + rows[:, None] * (K // 4) + tl.arange(0, K // 4)[None, :],
                 s3)

    return ops_reduce_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(XN, generator=g, dtype=torch.float32),
            "h": torch.randn(K, generator=g, dtype=torch.float32
                             ).to(torch.float16)}


def reference(inputs):
    import torch
    x, h = inputs["x"], inputs["h"]
    x2 = x.reshape(M, K)
    a = x[:K]

    o = torch.zeros(ON, dtype=torch.float32)
    o[0:K] = x2.sum(dim=0)
    o[K:K + M] = x2.sum(dim=1) + (x2.argmax(dim=1) + x2.argmin(dim=1)
                                  ).to(torch.float32)
    # tl.sum on f16 returns f16 (accumulated in f32 inside the converter), so
    # the reference rounds the h-sum to f16 before widening it again.
    o[K + M] = a.sum() + h.sum().to(torch.float16).to(torch.float32)

    p = x2.reshape(M, 4, K // 4).sum(dim=1).reshape(-1)
    return {"o": o, "p": p}


SPEC = KernelSpec(
    name="ops_reduce_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "h_ptr": "*fp16", "o_ptr": "*fp32",
               "p_ptr": "*fp32", "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "float32", (XN,)), Arg("h", IN, "float16", (K,)),
          Arg("o", OUT, "float32", (ON,)), Arg("p", OUT, "float32", (PN,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
