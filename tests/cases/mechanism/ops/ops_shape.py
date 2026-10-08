"""The shape-changing converters: reshape (all three arms), trans, expand_dims,
broadcast, join, split, cat.

Split out of the old tests/cases/ops.py.  One file for the whole family
because they share one setup and one trap -- see WHY EVERY SHAPE OP IS ON A
VALUE -- and because the reshape arms are only readable as a set: `outputRank <
inputRank` vs `>=` vs `no valid reassociation at all` is one if/else chain and
three adjacent lines of triton source here.

Nothing in this file emits linalg except broadcast; the rest are pure tensor
dialect (expand_shape / collapse_shape / reshape / concat / insert_slice /
extract_slice / linalg.transpose), which is what makes the whole family cheap to
read in one pass of 03-arith-linalg.mlir.

COVERS -- 9 branch ids.  Line numbers are the triton-shared WORKING TREE,
source root /workspace/triton_shared/, all in
lib/Conversion/TritonArithToLinalg/TritonArithToLinalg.cpp.

  reshape-expand         :81   ReshapeConverter, reassociation found and
                               outputRank >= inputRank (CP.hpp:2281)
  reshape-dynamic        :81   fall-through: static output shape but NO valid
                               reassociation -- a genuine data shuffle
                               (CP.hpp:2287-2292)
  reshape-collapse       :81   reassociation found and outputRank < inputRank
                               (CP.hpp:2277)
  trans-order-explicit   :63   TransposeConverter -> getTransposedValue with a
                               NON-empty order attr (CP.hpp:128-135); asserts
                               order.size() == sourceRank
  expanddims-expandshape :65   ExpandDimsConverter, builds ReassociationIndices
                               by walking srcRank
  broadcast-generic      :62   BroadcastConverter; asserts a single result, and
                               getBroadcastDims asserts if nothing broadcasts
  join-insert-slice      :73   JoinConverter
  split-extract-slice    :72   SplitConverter, loops i in {0,1} popping the LAST
                               offset/size and pushing offset=i, size=1
  cat-tensor-concat      :71   CatConverter, concat dimension HARDCODED to 0

  ("CP.hpp" = include/triton-shared/Conversion/TritonArithToLinalg/
   ConversionPatterns.hpp.)

WHAT PROVES IT FIRED   work/ops_shape_kernel/triton-shared-detail/

  The three reshape arms, all present, all different, in source order:

      reshape-expand    (16,) -> (2,8), outputRank 2 >= inputRank 1
        %expanded = tensor.expand_shape %20 [[0, 1]] output_shape [2, 8]
                      : tensor<16xf32> into tensor<2x8xf32>

      reshape-dynamic   (8,2) -> (2,8), same rank, NO valid reassociation, so
                        the converter falls through to a real data shuffle --
                        note the shape OPERAND, which no other arm has:
        %cst = arith.constant dense<[2, 8]> : tensor<2xi64>
        %reshape = tensor.reshape %23(%cst)
                      : (tensor<8x2xf32>, tensor<2xi64>) -> tensor<2x8xf32>

      reshape-collapse  (2,8) -> (16,), outputRank 1 < inputRank 2
        %collapsed = tensor.collapse_shape %24 [[0, 1]]
                      : tensor<2x8xf32> into tensor<16xf32>

  trans-order-explicit -- tensor.empty of the PERMUTED shape plus a
  linalg.transpose carrying the order attribute as its permutation:

      %22 = tensor.empty() : tensor<8x2xf32>
      %transposed = linalg.transpose ins(%21 : tensor<2x8xf32>)
                                     outs(%22 : tensor<8x2xf32>)
                                     permutation = [1, 0]

  join-insert-slice -- one tensor.empty then TWO chained insert_slice at
  trailing offsets 0 and 1, size 1:

      %16 = tensor.empty() : tensor<8x2xf32>
      %inserted_slice   = tensor.insert_slice %13 into %16[0, 0] [8, 1] [1, 1]
                            : tensor<8xf32> into tensor<8x2xf32>
      %inserted_slice_4 = tensor.insert_slice %15 into %inserted_slice[0, 1]
                            [8, 1] [1, 1] : tensor<8xf32> into tensor<8x2xf32>

  split-extract-slice -- exactly TWO extract_slice, offsets [0,0] and [0,1],
  sizes [8,1], both rank-reducing to tensor<8xf32>:

      %extracted_slice   = tensor.extract_slice %27[0, 0] [8, 1] [1, 1]
                             : tensor<8x2xf32> to tensor<8xf32>
      %extracted_slice_7 = tensor.extract_slice %27[0, 1] [8, 1] [1, 1]
                             : tensor<8x2xf32> to tensor<8xf32>

  cat-tensor-concat -- the census's note made visible: CatConverter builds
  tensor.concat dim(0), and then runOnOperation immediately runs
  applyTensorConcatDecomposition (TritonArithToLinalgPass.cpp:79-90), so there
  is no tensor.concat in the final IR at all, only its decomposition.  Told
  apart from the join above by being RANK-PRESERVING -- offsets [0] and [8] on
  one dimension, not [0,0] and [0,1] on two:

      %19 = tensor.empty() : tensor<16xf32>
      %inserted_slice_5 = tensor.insert_slice %17 into %19[0] [8] [1]
      %inserted_slice_6 = tensor.insert_slice %18 into %inserted_slice_5[8] [8] [1]

  expanddims-expandshape -- a single tensor.expand_shape, no linalg at all, one
  per axis:

      %expanded_8 = tensor.expand_shape %31 [[0, 1]] output_shape [4, 1]
                      : tensor<4xf32> into tensor<4x1xf32>
      %expanded_9 = tensor.expand_shape %33 [[0, 1]] output_shape [1, 4]
                      : tensor<4xf32> into tensor<1x4xf32>

  broadcast-generic -- two of them, one per side of the outer product, and the
  only linalg.generic anywhere carrying the distinctive broadcastDims attribute.
  The size-1 source dim is mapped to affine constant 0 in the INPUT map, and
  which dim that is flips between the two:

      #map1 = affine_map<(d0, d1) -> (d0, d1)>       the output map, both
      #map2 = affine_map<(d0, d1) -> (d0, 0)>        ar[:, None]
      #map3 = affine_map<(d0, d1) -> (0, d1)>        bc[None, :]

      %35 = linalg.generic {indexing_maps = [#map2, #map1],
                            iterator_types = ["parallel", "parallel"]}
              ins(%expanded_8 : tensor<4x1xf32>) outs(%34 : tensor<4x4xf32>)
              attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4x4xf32>
      %37 = ... ins(%expanded_9 : tensor<1x4xf32>) ...
              attrs =  {broadcastDims = array<i64: 0>}

WHY EVERY SHAPE OP IS ON A VALUE, NEVER ONLY ON AN ADDRESS
  `rows[:, None] * K + cols[None, :]` inside a tl.load is NOT a witness for
  expanddims or broadcast: PtrAnalysis consumes the expand_dims and the
  broadcast during pass 1 and folds them into the offsets/strides of a
  tts.make_tptr, so no tt.expand_dims or tt.broadcast survives to pass 4.  The
  outer product at the bottom of this kernel therefore expands and broadcasts
  the LOADED DATA, and the 2-D store address is written the ordinary way.  The
  same rule is why `t` and `j` are consumed arithmetically before the split.

WHY reshape-dynamic IS THE (8,2) -> (2,8) ONE
  getReassociationIndicesForReshape can group (16,) into (2,8) and can flatten
  (2,8) into (16,), but there is no grouping that turns (8,2) into (2,8) -- the
  elements genuinely move.  That is the only way to reach the fall-through arm;
  any same-rank reshape whose dims happen to be groupable takes reshape-expand
  instead.  (If the output shape were dynamic the pattern would return failure
  outright, which is a fourth, failing branch that no kernel here provokes.)

ARITHMETIC
  Trivial and exact: a cat/reshape/transpose/reshape/reshape chain over the
  first 16 elements of x, a join/split difference, and a 4x4 outer product.
  The +1.0 in `tl.trans(mrs + 1.0, 1, 0)` is applied BEFORE the transpose -- the
  reference has to match that order.

SCOPE
  IR-census kernel.  All 9 ids above were confirmed by gdb line hits
  (the branch census) and the IR was read from tools/lower_steps.py.
  Like every file under tests/cases/ it does not reach run.py's numeric
  stages; tests/cases/harness/gap_fill.py stops in the same place.

POSITIVE KERNEL: every pass of the linalg lowering succeeds.

Shape-changing converter family: Reshape (expand/dynamic/collapse), Transpose, ExpandDims, Broadcast, Join, Split, Cat. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 4          # outer-product side
K = 8          # vector length
XN = 64
SN = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_shape_kernel(x_ptr, s_ptr, M: tl.constexpr, K: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, K)
        a = tl.load(x_ptr + cols)
        b = tl.load(x_ptr + K + cols)

        # -- join / cat, then the three reshape arms and a transpose ---------
        j = tl.join(a, b)                                # (K, 2)
        c = tl.cat(a + 1.0, b - 1.0, can_reorder=True)   # (2K,)
        mrs = tl.reshape(c * 2.0, (2, K))                # 1-D -> 2-D  EXPAND
        t = tl.trans(mrs + 1.0, 1, 0)                    # (2,K) -> (K,2) TRANS
        d = tl.reshape(t * 3.0, (2, K))                  # (K,2) -> (2,K) DYNAMIC
        f = tl.reshape(d + 4.0, (2 * K,))                # 2-D -> 1-D  COLLAPSE
        tl.store(s_ptr + tl.arange(0, 2 * K), f)

        p, r = tl.split(j * 2.0 + t)                     # (K,2) -> two (K,)
        tl.store(s_ptr + 2 * K + cols, p - r)

        # -- expand_dims + broadcast, on DATA and not on an address ----------
        ar = tl.load(x_ptr + rows)
        bc = tl.load(x_ptr + M + rows)
        outer = ar[:, None] * bc[None, :]                # (M,1) x (1,M) -> (M,M)
        tl.store(s_ptr + 3 * K + rows[:, None] * M + rows[None, :], outer)

    return ops_shape_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(XN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    a, b = x[:K], x[K:2 * K]
    c = torch.cat([a + 1.0, b - 1.0])
    mrs = (c * 2.0).reshape(2, K)
    t = (mrs + 1.0).transpose(0, 1).contiguous()    # the +1.0 is BEFORE the trans
    d = (t * 3.0).reshape(2, K)                     # a real data shuffle
    f = (d + 4.0).reshape(2 * K)
    j = torch.stack([a, b], dim=1)                  # tl.join
    sp = j * 2.0 + t
    ar, bc = x[:M], x[M:2 * M]
    outer = ar.unsqueeze(1) * bc.unsqueeze(0)

    s = torch.zeros(SN, dtype=torch.float32)
    s[0:2 * K] = f                                          # [0:16]
    s[2 * K:3 * K] = sp[:, 0] - sp[:, 1]                    # [16:24]
    s[3 * K:3 * K + M * M] = outer.reshape(-1)              # [24:40]
    return {"s": s}


SPEC = KernelSpec(
    name="ops_shape_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "float32", (XN,)), Arg("s", OUT, "float32", (SN,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
