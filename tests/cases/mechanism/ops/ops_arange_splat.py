"""The four converters that MATERIALISE a tensor: tt.make_range (both start
arms), tt.splat, tt.unsplat, and a dense splat arith.constant.

Split out of the old tests/cases/ops.py.  These four sit on consecutive
registration lines (TritonArithToLinalg.cpp:64, 77, 78, 79) and answer one
question between them -- how does a value that is not already a tensor become
one, and how does it get back out.  Reading them side by side is the point:
a RUNTIME scalar goes through tt.splat, a COMPILE-TIME one never becomes a
tt.splat at all, and the reverse direction is its own op again.

COVERS -- 5 branch ids.  Line numbers are the triton-shared WORKING TREE,
source root /workspace/triton_shared/.

  makerange-no-start    lib/Conversion/TritonArithToLinalg/
                        TritonArithToLinalg.cpp:64
      MakeRangeConverter with op.getStart() == 0 (falsy,
      ConversionPatterns.hpp:792).  Hard-asserts rank == 1 and a 32-bit element
      type.

  makerange-with-start  TritonArithToLinalg.cpp:64
      the non-zero-start arm: the SAME linalg.generic plus an arith.constant
      <start> created OUTSIDE the region (via `rewriter`, not `nestedBuilder`)
      and an arith.addi inside the body before the yield.

  splat-fill            TritonArithToLinalg.cpp:77
      SplatConverter, unconditional: tensor.empty + linalg.fill
      ins(adaptor.getSrc()).  Needs a RUNTIME scalar -- a compile-time one is
      folded into a dense attribute by the frontend and never becomes a
      tt.splat at all.

  unsplat-extract       TritonArithToLinalg.cpp:78
      UnsplatConverter, with a rank sub-branch (ConversionPatterns.hpp:660-665):
      rank > 0 builds an `arith.constant 0 : index` per dimension, rank 0 passes
      an empty index list.  This file takes the rank-1 arm.

  denseconst-fill       TritonArithToLinalg.cpp:79
      DenseConstantConverter on arith::ConstantOp.  The pattern is NOT
      self-guarded -- it does an unchecked cast<DenseElementsAttr> +
      getSplatValue -- and is gated entirely by the ConversionTarget predicate
      at TritonArithToLinalgPass.cpp:130-143, which marks a constant illegal
      only when the result is a RankedTensorType AND the attribute is a splat of
      FloatType/IntegerType.

WHAT PROVES IT FIRED   work/ops_arange_splat_kernel/triton-shared-detail/

  01-structured.mlir still has the raw ops -- pass 1 folded the arange that is
  used as an ADDRESS into the tts.make_tptr, but the two used as VALUES survive:

      %cst = arith.constant dense<2.000000e+00> : tensor<128xf32>
      %0 = tt.make_range {end = 128 : i32, start = 0 : i32}   : tensor<128xi32>
      %1 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32>
      %8 = tt.unsplat %7 : tensor<1xf32>
      %9 = tt.splat %8 : f32 -> tensor<128xf32>

  03-arith-linalg.mlir, makerange-no-start -- ZERO inputs, one output, body is
  just linalg.index + index_cast:

      %4 = tensor.empty() : tensor<128xi32>
      %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]}
             outs(%4 : tensor<128xi32>) {
      ^bb0(%out: i32):
        %20 = linalg.index 0 : index
        %21 = arith.index_cast %20 : index to i32
        linalg.yield %21 : i32
      } -> tensor<128xi32>

  and makerange-with-start (%7) is the same generic with two extra lines, the
  constant hoisted OUT of the region exactly as the census describes:

      %c128_i32 = arith.constant 128 : i32          <- outside the body
      %7 = linalg.generic ... outs(%6 : tensor<128xi32>) {
      ^bb0(%out: i32):
        %20 = linalg.index 0 : index
        %21 = arith.index_cast %20 : index to i32
        %22 = arith.addi %21, %c128_i32 : i32       <- inside the body
        linalg.yield %22 : i32
      } -> tensor<128xi32>

  unsplat-extract is a bare tensor.extract with one index constant, and
  splat-fill the tensor.empty + linalg.fill pair that consumes it:

      %extracted = tensor.extract %10[%c0] : tensor<1xf32>
      %11 = tensor.empty() : tensor<128xf32>
      %12 = linalg.fill ins(%extracted : f32) outs(%11 : tensor<128xf32>)

  denseconst-fill is the same SHAPE from a different source -- note the scalar
  constant was materialised out of the dense<2.0> attribute, and the fill is
  built at the top of the function, before any load:

      %cst = arith.constant 2.000000e+00 : f32
      %2 = tensor.empty() : tensor<128xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128xf32>)

  ONE THING THAT LOOKS WRONG AND IS NOT: the `tl.load(x_ptr + tl.arange(0, 1))`
  becomes `%10 = tts.gather %arg0[%1]`, not a tts.make_tptr.  A one-element
  arange is constant-folded to dense<0> by the frontend, so PtrAnalysis sees a
  splat offset rather than a range and takes the gather path.  That is
  incidental -- the witness this file needs from that line is the tt.unsplat on
  its result, and that is unaffected.

WHY BOTH RANGES ARE ADDED INTO THE STORED VALUE
  This is the trap in the arange case.  An arange consumed only as an ADDRESS
  never reaches pass 4: PtrAnalysis folds it into the offsets/strides of a
  tts.make_tptr during pass 1 and erases the tt.make_range.  Both ranges here
  are therefore also added into the stored value, which keeps a live
  tt.make_range for the converter to rewrite.  Delete the `.to(tl.float32)`
  term and this file covers neither arange branch.

ARITHMETIC
  o = x*2 + x[0] + (i + (i+128)), exact.

SCOPE
  IR-census kernel.  All 5 ids above were confirmed by gdb line hits
  (the branch census) and the IR was read from tools/lower_steps.py.
  Like every file under tests/cases/ it does not reach run.py's numeric
  stages -- stage 3 (pytorchsim_triton_compiler/lowering/passes/) rejects a static reinterpret_cast
  offset, and tests/cases/harness/gap_fill.py stops in the same place.

POSITIVE KERNEL: every triton-shared pass succeeds.

MakeRangeConverter (both start arms), SplatConverter (runtime scalar), UnsplatConverter (rank-1 arm) and DenseConstantConverter (compile-time splat). POSITIVE.
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
    def ops_arange_splat_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)                  # start == 0
        rr = tl.arange(BLOCK, 2 * BLOCK)         # start == 128
        x = tl.load(x_ptr + r)

        one = tl.load(x_ptr + tl.arange(0, 1))   # tensor<1xf32>
        sc = one.item()                          # tt.unsplat -> tensor.extract
        sp = tl.full((BLOCK,), sc, tl.float32)   # RUNTIME scalar -> tt.splat
        k = tl.full((BLOCK,), 2.0, tl.float32)   # CONSTANT -> dense arith.constant

        # both ranges must appear in a VALUE, not only in an address -- see the
        # module docstring.
        tl.store(o_ptr + r, x * k + sp + (r + rr).to(tl.float32))

    return ops_arange_splat_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r = torch.arange(0, BLOCK, dtype=torch.int32)
    rr = torch.arange(BLOCK, 2 * BLOCK, dtype=torch.int32)
    return {"o": x * 2.0 + x[0] + (r + rr).to(torch.float32)}


SPEC = KernelSpec(
    name="ops_arange_splat_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
