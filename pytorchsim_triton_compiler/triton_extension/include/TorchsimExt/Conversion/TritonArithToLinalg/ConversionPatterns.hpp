#ifndef TRITON_CONVERSION_PATTERNS
#define TRITON_CONVERSION_PATTERNS

//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "TorchsimExt/Analysis/MaskAnalysis.h"
#include "TorchsimExt/Analysis/OpFoldResultUtils.h"
#include "TorchsimExt/Conversion/TritonArithToLinalg/ConversionTools.h"
#include "TorchsimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "TorchsimExt/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "TorchsimExt/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace triton;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// getScalarValue IS tts::utils::getScalarValue AND NOT A COPY OF IT. A static
// duplicate stood here -- sixty-two lines, identical but for the builder
// parameter's type -- and a ConversionPatternRewriter IS an OpBuilder, so the
// shared one takes this caller unchanged. What the copy cost was the include it
// was written to avoid; what it risked is that "how do I read a scalar out of a
// splat/sitofp/truncf chain" had two answers (philosophy 13, rule 9).

// if order is empty, transpose the last two dimensions
// otherwise, use the provided order.
// The order must be a permutation of the source rank.
static Value getTransposedValue(Value source, const Location loc,
                                ConversionPatternRewriter &rewriter,
                                llvm::ArrayRef<int32_t> order = {},
                                Type resultType = nullptr) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(sourceRank);
  SmallVector<int64_t> transposedShape(sourceType.getShape());
  if (order.empty()) {
    std::iota(std::begin(perm), std::end(perm), 0);
    std::swap(perm[sourceRank - 1], perm[sourceRank - 2]);
    std::swap(transposedShape[sourceRank - 1], transposedShape[sourceRank - 2]);
  } else {
    // Use the provided order
    assert(order.size() == sourceRank && "Order size must match source rank");
    for (unsigned i = 0; i < sourceRank; ++i) {
      perm[i] = order[i];
      transposedShape[i] = sourceType.getShape()[order[i]];
    }
  }

  // A transposed tile keeps its layout, transposed: the dialect that owns the
  // encoding says what that is. Without an encoding there is nothing to carry.
  // The op that asked for the transpose already SAYS its result type; only a
  // transpose nobody spelled (a reduce's) infers one.
  Attribute transposedEncoding;
  if (auto want = dyn_cast_or_null<RankedTensorType>(resultType)) {
    transposedEncoding = want.getEncoding();
  } else if (auto encoding = sourceType.getEncoding()) {
    SmallVector<int32_t> order32(perm.begin(), perm.end());
    auto *inferLayout = encoding.getDialect()
                            .getRegisteredInterface<triton::DialectInferLayoutInterface>();
    if (!inferLayout ||
        failed(inferLayout->inferTransOpEncoding(encoding, sourceType.getShape(),
                                                 order32, transposedEncoding, loc))) {
      transposedEncoding = nullptr;
    }
  }
  Value transposeInit = tensor::EmptyOp::create(
      rewriter, loc,
      RankedTensorType::get(transposedShape, sourceType.getElementType(),
                            transposedEncoding),
      ValueRange{});

  Value transpose =
      linalg::TransposeOp::create(rewriter, loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

// for IntLike and FloatLike types
static std::optional<unsigned> getBitWidth(Type a) {
  if (auto type = dyn_cast<TensorType>(a)) {
    auto elementType = type.getElementType();
    if (elementType.isIntOrFloat()) {
      return type.getElementType().getIntOrFloatBitWidth();
    }
    return std::nullopt;
  }

  if (a.isIntOrFloat())
    return a.getIntOrFloatBitWidth();

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = cast<RankedTensorType>(op.getType());
    auto loc = op.getLoc();

    auto init = tensor::EmptyOp::create(rewriter, loc, opType, ValueRange{});

    auto filledTensor =
        linalg::FillOp::create(rewriter, loc, ValueRange{adaptor.getSrc()},
                               ValueRange{init})
            .result();

    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

struct UnsplatConverter : public OpConversionPattern<triton::UnsplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::UnsplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = op.getSrc().getType();

    // Only generate indices for non-zero rank tensors.
    SmallVector<Value, 1> indices(tensorType.getRank());
    if (indices.size() > 0) {
      auto zeroIdx =
          rewriter.createOrFold<arith::ConstantIndexOp>(op.getLoc(), 0);
      llvm::fill(indices, zeroIdx);
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, adaptor.getSrc(),
                                                   indices);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
private:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op->getNumResults() == 1 && "code assumes single result!");
    RankedTensorType sourceType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());

    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(),
                        rewriter.getMultiDimIdentityMap(resultRank));

    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto init = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, op->getResultTypes(), ValueRange{adaptor.getSrc()},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = blockArgs[0];
          linalg::YieldOp::create(nestedBuilder, loc, opResult);
        });

    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = cast<RankedTensorType>(src.getType()).getRank();
    auto resType = cast<RankedTensorType>(op->getResultTypes()[0]);
    SmallVector<ReassociationIndices> reassoc;
    int64_t c = 0;
    for (int64_t i = 0; i < srcRank; i++) {
      ReassociationIndices g;
      g.push_back(c++);
      if (op.getAxis() == i) {
        g.push_back(c++);
      } else if (op.getAxis() == i + 1 && i == srcRank - 1) {
        g.push_back(c++);
      }
      reassoc.push_back(g);
    }

    auto expandShapeOp = tensor::ExpandShapeOp::create(rewriter, op.getLoc(),
                                                       resType, src, reassoc);

    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

struct TransposeConverter : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res = getTransposedValue(adaptor.getSrc(), op.getLoc(), rewriter,
                                  op.getOrder(), op.getType());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<TensorType>(op.getResult().getType());
    auto shape = type.getShape();
    auto elementType = type.getElementType();
    auto context = rewriter.getContext();

    assert(type.getShape().size() == 1 &&
           type.getElementType().getIntOrFloatBitWidth() == 32 &&
           "make range can only return 1D int32 tensor");

    SmallVector<AffineMap> indexingMaps{AffineMap::get(
        /* dimCount */ 1, /* symbolCount */ 0,
        SmallVector<AffineExpr>{mlir::getAffineDimExpr(0, context)}, context)};

    auto init = tensor::EmptyOp::create(
        rewriter, loc, cast<RankedTensorType>(op.getType()), ValueRange{});
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, op->getResultTypes(), /* operands */ ValueRange{},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value index = linalg::IndexOp::create(nestedBuilder, loc, 0);
          Value res = arith::IndexCastOp::create(nestedBuilder, loc,
                                                 type.getElementType(), index);
          if (op.getStart()) {
            auto start = mlir::arith::ConstantIntOp::create(
                rewriter, op.getLoc(), op.getStart(),
                type.getElementType().getIntOrFloatBitWidth());
            res = arith::AddIOp::create(nestedBuilder, loc, res, start);
          }
          linalg::YieldOp::create(nestedBuilder, loc, res);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct AssertConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value condVal = op.getCondition();

    auto assertMessage =
        llvm::formatv("Assertion `{0}` failed", op.getMessage());

    // The condition can only be I1 or I1Tensor (integer or tensor) from
    // TritonOps.td. Tensors will always be RankedTensorType.
    if (isa<mlir::IntegerType>(condVal.getType())) {
      // handle scalar case
      mlir::cf::AssertOp::create(rewriter, op.getLoc(), condVal,
                                 assertMessage.str());
    } else if (auto tensorType =
                   dyn_cast<RankedTensorType>(condVal.getType())) {
      // handle tensor case
      int64_t rank = tensorType.getRank();

      // create identity mapping for access pattern
      SmallVector<AffineMap, 3> indexingMaps{
          AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())};

      // loops do not depend on each other
      SmallVector<utils::IteratorType, 3> iteratorTypes(
          rank, utils::IteratorType::parallel);

      linalg::GenericOp::create(
          rewriter, op.getLoc(), TypeRange{}, condVal, ValueRange{},
          ArrayRef<AffineMap>{indexingMaps},
          ArrayRef<utils::IteratorType>{iteratorTypes},
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // obtain the element in the tensor
            Value element = args[0];

            // make a cf.assert for the current element
            mlir::cf::AssertOp::create(b, loc, element, assertMessage.str());

            linalg::YieldOp::create(b, loc);
          });
    } else {
      op.emitError("Unexpected type in triton::AssertOp");
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // arith::bitcast does not support casting pointers
    if (triton::isPtrTypeLike(op.getType())) {
      return failure();
    }

    auto arithBitcast = arith::BitcastOp::create(rewriter, op.getLoc(),
                                                 op.getType(), op.getOperand());

    rewriter.replaceOp(op, arithBitcast.getResult());
    return success();
  }
};

struct CallConverter : public OpConversionPattern<triton::CallOp> {
  using OpConversionPattern<triton::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> args = adaptor.getOperands();

    // We need to pass extra arguments added by addProgramInfo which are
    // num_programs and program_ids
    if (FuncOp parentFunc = op->getParentOfType<triton::FuncOp>()) {
      SymbolRefAttr calleeAttr = op.getCalleeAttr();
      StringRef calleeName = calleeAttr.getRootReference();

      if (ModuleOp module = op->getParentOfType<ModuleOp>()) {
        if (FuncOp calleeFunc = module.lookupSymbol<FuncOp>(calleeName)) {
          size_t argsNeed = calleeFunc.getFunctionType().getInputs().size();
          Block &entryBlock = parentFunc.front();
          auto parentInputs = entryBlock.getArguments();
          size_t argsParent = parentInputs.size();

          if (argsNeed > args.size()) {
            int missing = argsNeed - args.size();
            int missingArgsStart = argsParent - missing;
            for (int i = 0; i < missing; i++) {
              args.push_back(parentInputs[missingArgsStart + i]);
            }
          }
        }
      }
    }

    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     op.getResultTypes(), args);

    if (!call) {
      op.emitError("Failed to create func::CallOp");
      return failure();
    }

    rewriter.replaceOp(op, call);
    return success();
  }
};

struct FpToFpConverter : public OpConversionPattern<triton::FpToFpOp> {
  using OpConversionPattern<triton::FpToFpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundingMode = triton::RoundingMode::RTNE; // default

    auto roundingModeAttr = op.getRounding();
    if (roundingModeAttr.has_value()) {
      roundingMode = roundingModeAttr.value();
    }

    assert(roundingMode != triton::RoundingMode::RTZ &&
           "Rounding Towards Zero is not supported");

    Type resultType = op.getResult().getType();

    auto operandWidth = getBitWidth(op.getOperand().getType());
    auto resultWidth = getBitWidth(resultType);

    assert(operandWidth.has_value() && resultWidth.has_value() &&
           "Not a float-like operand or result");

    if (operandWidth.value() > resultWidth.value()) {
      Value truncatedValue = arith::TruncFOp::create(
          rewriter, op.getLoc(), resultType, op.getOperand());
      rewriter.replaceOp(op, truncatedValue);
      return success();
    }

    Value extendedValue = arith::ExtFOp::create(rewriter, op.getLoc(),
                                                resultType, op.getOperand());
    rewriter.replaceOp(op, extendedValue);

    return success();
  }
};

struct ClampConverter : public OpConversionPattern<triton::ClampFOp> {
  using OpConversionPattern<triton::ClampFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool propagateNan = op.getPropagateNan() == triton::PropagateNan::ALL;

    Location loc = op.getLoc();
    Value x = adaptor.getOperands()[0];
    Value min = adaptor.getOperands()[1];
    Value max = adaptor.getOperands()[2];

    Value clamp;
    if (propagateNan) {
      Value maxMin = arith::MaximumFOp::create(rewriter, loc, x, min);
      clamp = arith::MinimumFOp::create(rewriter, loc, maxMin, max);
    } else {
      Value maxMin = arith::MaxNumFOp::create(rewriter, loc, x, min);
      clamp = arith::MinNumFOp::create(rewriter, loc, maxMin, max);
    }
    rewriter.replaceOp(op, clamp);

    return success();
  }
};

struct PreciseSqrtConverter
    : public OpConversionPattern<triton::PreciseSqrtOp> {
  using OpConversionPattern<triton::PreciseSqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        math::SqrtOp::create(rewriter, op.getLoc(), adaptor.getOperands());

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct PreciseDivConverter : public OpConversionPattern<triton::PreciseDivFOp> {
  using OpConversionPattern<triton::PreciseDivFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PreciseDivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        arith::DivFOp::create(rewriter, op.getLoc(), adaptor.getOperands());

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

// tt.cat, tt.split and tt.join, and the pytorchsim ops a target lowers them to where its layout port lets
// them take any layout: one lowering each, the same tensor slices.
template <typename OpTy>
struct CatConverterOf : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The result keeps the op's own type: its layout encoding, when the
    // kernel arrived with layouts, is not something the inputs can say.
    auto replacement = tensor::ConcatOp::create(
        rewriter, op.getLoc(), op.getType(), rewriter.getI64IntegerAttr(0),
        adaptor.getOperands());

    rewriter.replaceOp(op, replacement);

    return success();
  }
};

template <typename OpTy>
struct SplitConverterOf : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op->getOperand(0);
    auto inputType = cast<RankedTensorType>(input.getType());

    Type resultType = op.getResults().front().getType();
    auto resultTensor = cast<RankedTensorType>(resultType);
    auto shape = inputType.getShape();

    SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = llvm::to_vector(
        llvm::map_range(shape, [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    SmallVector<Value> results;

    for (int i = 0; i < 2; ++i) {
      offsets.pop_back();
      sizes.pop_back();

      offsets.push_back(rewriter.getIndexAttr(i));
      sizes.push_back(rewriter.getIndexAttr(1));
      Value slice = tensor::ExtractSliceOp::create(
          rewriter, loc, resultTensor, input, offsets, sizes, strides);
      results.push_back(slice);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

template <typename OpTy>
struct JoinConverterOf : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = op->getOperands();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    auto loc = op.getLoc();
    Value result = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});

    auto shape = resultType.getShape();

    SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = llvm::to_vector(
        llvm::map_range(shape, [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    for (int i = 0; i < 2; ++i) {
      offsets.pop_back();
      sizes.pop_back();

      offsets.push_back(rewriter.getIndexAttr(i));
      sizes.push_back(rewriter.getIndexAttr(1));
      result = tensor::InsertSliceOp::create(rewriter, loc, inputs[i], result,
                                             offsets, sizes, strides);
    }

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct MulHiUIOpConverter : public OpConversionPattern<triton::MulhiUIOp> {
  using OpConversionPattern<triton::MulhiUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto mulResult =
        arith::MulUIExtendedOp::create(rewriter, loc, adaptor.getOperands());
    rewriter.replaceOp(op, mulResult.getHigh());

    return success();
  }
};

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  // true means tensor elements are zeros
  // false means not zero or it cannot be determined
  bool isZeroTensor(Value &v, bool integers) const {
    if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
      if (auto constOp = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) {
          return val.getValueAsDouble() == 0.;
        }
        if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) {
          return val.getValue() == 0;
        }
      }
      return false;
    }

    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          if (integers)
            return denseAttr.getSplatValue<APInt>().isZero();
          return denseAttr.getSplatValue<APFloat>().isZero();
        }
      }
    }

    return false;
  }

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto opa = op.getA();
    auto opb = op.getB();
    auto opc = op.getC();

    auto dstType = cast<RankedTensorType>(op.getType());
    auto elementType = dstType.getElementType();
    bool integers = elementType.isInteger();
    bool skipC = isZeroTensor(opc, integers);

    // tt.dot's C is the accumulator, and linalg.matmul's outs is exactly that:
    // the result is outs + A*B. A zero C starts from a zero fill instead.
    if (!skipC) {
      rewriter.replaceOp(op, linalg::MatmulOp::create(rewriter, loc,
                                                      ValueRange{opa, opb},
                                                      ValueRange{opc})
                                 .getResult(0));
      return success();
    }

    auto init = tensor::EmptyOp::create(rewriter, loc, dstType, ValueRange{});
    TypedAttr constantAttr =
        integers
            ? static_cast<TypedAttr>(rewriter.getIntegerAttr(elementType, 0))
            : static_cast<TypedAttr>(rewriter.getFloatAttr(elementType, 0));

    auto zero = mlir::arith::ConstantOp::create(rewriter, op.getLoc(),
                                                elementType, constantAttr);

    auto zeroes = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                         ValueRange{init})
                      .result();

    rewriter.replaceOp(op, linalg::MatmulOp::create(rewriter, loc,
                                                    ValueRange{opa, opb},
                                                    ValueRange{zeroes})
                               .getResult(0));
    return success();
  }
};

struct ReduceConverter : public OpConversionPattern<triton::ReduceOp> {

  ReduceConverter(MLIRContext *context, bool transposeToRank0 = true,
                  PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit),
        transposeToRank0(transposeToRank0) {}

private:
  bool transposeToRank0;

  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const {
    auto reduceBlock = redOp.getBody();
    return llvm::map_to_vector(reduceBlock->without_terminator(),
                               [](Operation &op) { return &op; });
  }

  bool isReductionOpSupported(Operation *redOp) const {
    return isa<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::MaximumFOp,
               arith::MulFOp, arith::MulIOp, arith::MaxNumFOp,
               arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
               arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::OrIOp,
               arith::XOrIOp>(redOp);
  }

  arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr =
        llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
            .Case([&](arith::AddFOp) {
              return rewriter.getFloatAttr(constantType, 0.f);
            })
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, -std::numeric_limits<float>::infinity());
            })
            .Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, std::numeric_limits<float>::infinity());
            })
            .Case([&](arith::MinSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxIntN(bitWidth));
            })
            .Case([&](arith::MinUIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxUIntN(bitWidth));
            })
            .Case([&](arith::MaxSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::minIntN(bitWidth));
            })
            .Case<arith::MaxUIOp, arith::XOrIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 0); })
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .Case<arith::MulIOp, arith::AndIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 1); })
            .Case([&](arith::OrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Default([](Operation *op) {
              op->dump();
              llvm_unreachable("Reduction op not yet supported");
              return nullptr;
            });

    return arith::ConstantOp::create(rewriter, redOp->getLoc(), constantType,
                                     attr);
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    unsigned width =
        cast<FloatType>(Float32Type::get(elemType.getContext())).getWidth();
    return isa<FloatType>(elemType) &&
           elemType.getIntOrFloatBitWidth() < width &&
           isa<arith::AddFOp>(redOp);
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case<arith::AddFOp, arith::MulFOp>([&](auto redOp) {
          if (convertLhsToF32Precision) {
            lhs = arith::ExtFOp::create(b, loc,
                                        Float32Type::get(b.getContext()), lhs);
          }
          using OpT = std::remove_pointer_t<decltype(redOp)>;
          return OpT::create(b, loc, lhs, rhs);
        })
        .Case<arith::AddIOp, arith::AndIOp, arith::XOrIOp, arith::MaximumFOp,
              arith::MaxNumFOp, arith::MulIOp, arith::MinimumFOp,
              arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
              arith::MaxUIOp, arith::OrIOp>([&](auto redOp) {
          using OpT = std::remove_pointer_t<decltype(redOp)>;
          return OpT::create(b, loc, lhs, rhs);
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
                        typename triton::ReduceOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto source = adaptor.getOperands().front();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto elemType = sourceType.getElementType();
    auto resType = op.getResult().front().getType();
    auto loc = op.getLoc();
    auto reductionOps = getRedOps(op);

    // Reduction of arbitrary operations isn't supported because using the first
    // element across the reduction dimension requires us to iterate over a
    // subview that skips over each first element.
    if (reductionOps.size() != 1 ||
        !isReductionOpSupported(reductionOps.front())) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering reduction with body "
              "containing 1 max(i/f), addf, ori, or mulf.");
    }

    auto rop = reductionOps.front();
    auto axis = op.getAxis();
    auto rank = sourceType.getRank();
    auto isVectorReduce = (rank == 1);

    // For now we are transposing reductions from Triton Shared as an
    // optimization. This should not be the job of Triton Shared so moving
    // forward this will be removed. Doing the transpose here lacks a wider
    // scope of analysis that might indicate that the transpose to a given axis
    // is not optimal.
    if (transposeToRank0) {
      // if it is not a vector reduce, we can transpose the source
      // so that the reduction axis is the first dimension.
      if (!isVectorReduce && axis != 0) {
        SmallVector<int32_t> order;
        order.reserve(rank);
        order.push_back(axis);
        for (int i = 0; i < rank; ++i) {
          if (i != axis) {
            order.push_back(i);
          }
        }
        source = getTransposedValue(source, op.getLoc(), rewriter, order);
        axis = 0;
      }
    } else {
      // preserving old behavior until we remove the transpose entirely.
      if (axis == rank - 1 && !isVectorReduce) {
        source = getTransposedValue(source, op.getLoc(), rewriter);
        axis = rank - 2;
      }
    }

    bool convertToF32Precision = requiresF32Conversion(resType, rop);

    auto constantType = convertToF32Precision
                            ? Float32Type::get(rewriter.getContext())
                            : elemType;

    auto accBaseConstOp = getRedBaseConstOp(rewriter, rop, constantType);
    Value initTensor;

    if (isVectorReduce) {
      // The affine vectorizer cannot vectorize affine loops generated from
      // linalg.reduce for the vector reduce case, so we must rewrite the
      // linalg.reduce to affine loops manually. Here we lower to AllocTensor
      // directly instead of EmptyOp so that the subsequent pass can recognize
      // the patterns (EmptyOp is susceptible to being CSE'd away, making it
      // harder to match the patterns correctly).
      initTensor = bufferization::AllocTensorOp::create(
          rewriter, loc, RankedTensorType::get({}, constantType), ValueRange{});
      initTensor = tensor::InsertOp::create(rewriter, loc, accBaseConstOp,
                                            initTensor, ValueRange{});
    } else {
      auto resTensorType = cast<RankedTensorType>(resType);
      Value init = tensor::EmptyOp::create(
          rewriter, loc,
          RankedTensorType::get(resTensorType.getShape(), constantType,
                                resTensorType.getEncoding()),
          ValueRange{});
      initTensor =
          linalg::FillOp::create(rewriter, loc, ValueRange{accBaseConstOp},
                                 ValueRange{init})
              .result();
    }

    Value finalResult =
        linalg::ReduceOp::create(
            rewriter, loc, ValueRange{source}, ValueRange{initTensor},
            SmallVector<int64_t>{axis},
            [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
              assert(inputs.size() == 2);
              Value result = getRedElement(inputs[0], inputs[1], loc, rop,
                                           opBuilder, convertToF32Precision);
              linalg::YieldOp::create(opBuilder, loc, result);
            })
            .getResult(0);

    if (isVectorReduce) {
      finalResult =
          tensor::ExtractOp::create(rewriter, loc, constantType, finalResult);
    }

    if (convertToF32Precision) {
      finalResult =
          arith::TruncFOp::create(rewriter, loc, resType, finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        cast<RankedTensorType>(adaptor.getOperands().front().getType());
    assert(sourceType.hasRank() && "Expected input is "
                                   "ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction "
           "axis is within "
           "operand's rank");

    return convertToLinalgReduce(op, adaptor, rewriter);
  }
};

template <typename T>
class ArgMinMaxBaseConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  // We're looking for an op that looks like this:
  //
  // %9:2 = "tt.reduce"(%8, %3) <{axis = 0 : i32}> ({
  // ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
  // -------------------------------------------------
  // `matchTieBreakValue`                                |
  //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32         |
  //   %12 = arith.cmpi slt, %arg10, %arg12 : i32        |   1.
  //   %13 = arith.andi %11, %12 : i1                    |
  // -------------------------------------------------   |-> `matchShouldUpdate`
  // `matchUpdateCondition`                              |
  //   %14 = arith.cmpf ogt, %arg9, %arg11 : f32         |   2.
  // -------------------------------------------------   |
  //   %15 = arith.ori %14, %13 : i1                     |
  // -------------------------------------------------
  //   %16 = arith.select %15, %arg9, %arg11 : f32
  //   %17 = arith.select %15, %arg10, %arg12 : i32
  //   tt.reduce.return %16, %17 : f32, i32
  // }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  //
  // The above mlir code is lowered from this combinator in triton's
  // standard.py:
  //
  //  def _argmax_combine(value1, index1, value2, index2, tie_break_left):
  //    if tie_break_left:
  //        tie = value1 == value2 and index1 < index2
  //    else:
  //        tie = False
  //    gt = value1 > value2 or tie
  //    v_ret = core.where(gt, value1, value2)
  //    i_ret = core.where(gt, index1, index2)
  //    return v_ret, i_ret

  LogicalResult matchTieBreakResult(Value currValue, Value currIndex,
                                    Value reduceValue, Value reduceIndex,
                                    mlir::Block::iterator &it,
                                    Value &tileBreakValue) const {
    // Match the following (section 1. of the above)
    //
    //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    //   %12 = arith.cmpi slt, %arg10, %arg12 : i32
    //   %13 = arith.andi %11, %12 : i1
    //
    // which is equivalent to the following python code
    //
    //   tie = value1 == value2 and index1 < index2

    // matching: %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto eqCmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (eqCmpOp) {
      if (eqCmpOp.getPredicate() != arith::CmpFPredicate::OEQ) {
        return failure();
      }
      if (currValue != eqCmpOp.getLhs() || reduceValue != eqCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %12 = arith.cmpi slt, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto sltCmpOp = dyn_cast<arith::CmpIOp>(*it++);
    if (sltCmpOp) {
      if (sltCmpOp.getPredicate() != arith::CmpIPredicate::slt) {
        return failure();
      }
      if (currIndex != sltCmpOp.getLhs() || reduceIndex != sltCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %13 = arith.andi %11, %12 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto andOp = dyn_cast<arith::AndIOp>(*it++);
    if (andOp) {
      if (andOp.getLhs() != eqCmpOp || andOp.getRhs() != sltCmpOp) {
        return failure();
      }
    } else {
      return failure();
    }

    tileBreakValue = andOp;
    return success();
  }

  LogicalResult matchShouldUpdateValue(Value currValue, Value currIndex,
                                       Value reduceValue, Value reduceIndex,
                                       mlir::Block::iterator &it,
                                       Value &shouldUpdate) const {
    Value tieResult;
    if (failed(matchTieBreakResult(currValue, currIndex, reduceValue,
                                   reduceIndex, it, tieResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Tie break result match failed\n");
      return failure();
    }

    Value comparisonResult;
    if (failed(T::matchComparisonResult(currValue, currIndex, reduceValue,
                                        reduceIndex, it, comparisonResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Comparison result match failed\n");
      return failure();
    }

    // matching: %15 = arith.ori %14, %13 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto orOp = dyn_cast<arith::OrIOp>(*it++);
    if (orOp) {
      if (orOp.getLhs() != comparisonResult || orOp.getRhs() != tieResult) {
        return failure();
      }
    } else {
      return failure();
    }

    shouldUpdate = orOp;
    return success();
  }

  // The init carries the result's layout encoding, when the kernel arrived
  // with layouts: a reduce's output is where its layout is stated.
  Value getInitTensor(ConversionPatternRewriter &rewriter,
                      ArrayRef<int64_t> shape, Value fillValue, Location loc,
                      Attribute encoding = nullptr) const {
    Value initTensor = tensor::EmptyOp::create(
        rewriter, loc,
        RankedTensorType::get(shape, fillValue.getType(), encoding),
        ValueRange{});
    return linalg::FillOp::create(rewriter, loc, ValueRange{fillValue},
                                  ValueRange{initTensor})
        .result();
  }

public:
  ArgMinMaxBaseConverter(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override final {
    if (op.getBody()->getNumArguments() != 4) {
      return failure();
    }

    auto block = op.getBody();
    auto ops = block->without_terminator();

    Value currValue = block->getArgument(0);
    Value currIndex = block->getArgument(1);
    Value reduceValue = block->getArgument(2);
    Value reduceIndex = block->getArgument(3);

    auto opsIt = ops.begin();
    Value shouldUpdate;
    if (failed(matchShouldUpdateValue(currValue, currIndex, reduceValue,
                                      reduceIndex, opsIt, shouldUpdate))) {
      return failure();
    }

    // matching: %16 = arith.select %15, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto valueSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (valueSelectOp) {
      if (valueSelectOp.getCondition() != shouldUpdate ||
          currValue != valueSelectOp.getTrueValue() ||
          reduceValue != valueSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching:%17 = arith.select %15, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto indexSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (indexSelectOp) {
      if (indexSelectOp.getCondition() != shouldUpdate ||
          currIndex != indexSelectOp.getTrueValue() ||
          reduceIndex != indexSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: tt.reduce.return %16, %17 : f32, i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto termOp = dyn_cast<triton::ReduceReturnOp>(*opsIt++);
    if (termOp && termOp == block->getTerminator()) {
      auto opnds = termOp.getOperands();
      if (opnds != ArrayRef<Value>{valueSelectOp, indexSelectOp}) {
        return failure();
      }
    } else {
      return failure();
    }

    auto loc = op.getLoc();

    auto elemTypes = op.getElementTypes();

    // Set the initial value of the rank-0 tensor containing
    // the result value to either -inf or +inf depending on
    // whether we're dealing with argmax or argmin
    auto valueType = elemTypes[0];
    auto valuesAccBaseVal = arith::ConstantOp::create(
        rewriter, loc, valueType,
        rewriter.getFloatAttr(valueType, T::getBaseReductionValue()));

    // Set the initial value of the rank-0 tensor containing the index of the
    // min or max value to -1
    auto indexType = elemTypes[1];
    auto indicesAccBaseVal = arith::ConstantOp::create(
        rewriter, loc, indexType, rewriter.getIntegerAttr(indexType, -1));

    // Get the shape of the resulting tensors (both for values and indices). If
    // we are reducing to a single scalar, then the result's type is a tensor of
    // rank-0, otherwise we can reuse the original result shape
    auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
    const auto isScalarReduce = valueResultType == nullptr;
    SmallVector<int64_t> reductionResultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(valueResultType.getShape())};

    auto encodingOf = [&](unsigned k) -> Attribute {
      if (auto t = dyn_cast<RankedTensorType>(op.getType(k)))
        return t.getEncoding();
      return nullptr;
    };
    SmallVector<Value> outputs{
        getInitTensor(rewriter, reductionResultShape, valuesAccBaseVal, loc,
                      encodingOf(0)),
        getInitTensor(rewriter, reductionResultShape, indicesAccBaseVal, loc,
                      encodingOf(1))};

    auto linalgOp = linalg::ReduceOp::create(
        rewriter, loc, adaptor.getOperands(), outputs,
        SmallVector<int64_t>{adaptor.getAxis()},
        [&](OpBuilder &b, Location loc, ValueRange inputs) {
          assert(inputs.size() == 4);

          auto tritonReduceBlock = op.getBody();
          IRMapping mapping;
          mapping.map(tritonReduceBlock->getArguments(), inputs);

          for (auto &op : tritonReduceBlock->without_terminator()) {
            b.clone(op, mapping);
          }

          auto tritonYield = tritonReduceBlock->getTerminator();
          auto results =
              llvm::map_to_vector(tritonYield->getOperands(), [&](Value val) {
                return mapping.lookup(val);
              });
          linalg::YieldOp::create(b, loc, results);
        });

    if (isScalarReduce) {
      SmallVector<Value> reduceResults{
          tensor::ExtractOp::create(rewriter, loc, valueType,
                                    linalgOp.getResults()[0], ValueRange{}),
          tensor::ExtractOp::create(rewriter, loc, indexType,
                                    linalgOp.getResults()[1], ValueRange{})};
      rewriter.replaceOp(op, reduceResults);
    } else {
      rewriter.replaceOp(op, linalgOp);
    }
    return success();
  }
};

struct ArgMaxConverter : public ArgMinMaxBaseConverter<ArgMaxConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf ogt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OGT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return -std::numeric_limits<float>::infinity();
  }

  ArgMaxConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

struct ArgMinConverter : public ArgMinMaxBaseConverter<ArgMinConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf olt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OLT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return std::numeric_limits<float>::infinity();
  }

  ArgMinConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

// The fallback the three converters above leave behind: a `tt.reduce` whose
// body is neither one known combiner nor the argmin/argmax shape.
//
// WHAT THE PATTERN MATCHING WAS EVER FOR. `tt.reduce`'s body is already a
// `linalg.reduce` body -- ArgMinMaxBaseConverter clones it across verbatim.
// What cannot be cloned is the INIT: linalg.reduce folds a starting
// accumulator, and `ReduceConverter::getRedBaseConstOp` is a table of
// identities keyed by the op (`addf` -> 0, `maximumf` -> -inf, ...). An
// arbitrary body has no entry in that table, and an identity may not even be
// usable when one exists: flash attention's (max, sum) combiner has (-inf, 0)
// and folding it evaluates `0 * exp(-inf - -inf)` = nan.
//
// So this takes the answer TritonArithToLinalg.cpp already writes down beside
// the three registrations -- seed with the first element along the reduction
// axis and reduce the remaining ones -- which needs to know nothing about the
// body at all. Two `tensor.extract_slice`s per operand: slice 0 becomes the
// init (rank-reduced to the result's shape) and [1, extent) becomes the input.
//
// IT RUNS AT BENEFIT 0 so that every pattern above keeps the shape it emits
// today. A plain `tl.sum` still becomes fill + reduce over the whole extent
// with no slicing at all, which is what the same comment calls out as the
// reason not to do this unconditionally: the remainder is `extent - 1`, and a
// hardware that wants powers of two would rather have the identity.
struct ReduceGeneralConverter : public OpConversionPattern<triton::ReduceOp> {

  ReduceGeneralConverter(MLIRContext *context)
      : OpConversionPattern(context, /*benefit=*/0) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto operands = adaptor.getOperands();
    if (operands.empty()) {
      return rewriter.notifyMatchFailure(op, "reduce with no operands");
    }

    auto sourceType = dyn_cast<RankedTensorType>(operands.front().getType());
    if (!sourceType || !sourceType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "expected statically shaped ranked tensor operands");
    }

    // EVERY OPERAND IS SLICED THE SAME WAY, so they have to agree on what the
    // axis is. Triton's verifier already says so; checking here keeps the
    // slicing below from inventing an extent for one of them.
    for (auto operand : operands) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || type.getShape() != sourceType.getShape()) {
        return rewriter.notifyMatchFailure(
            op, "reduce operands do not share one shape");
      }
    }

    const int64_t rank = sourceType.getRank();
    const int64_t axis = op.getAxis();
    if (axis < 0 || axis >= rank) {
      return rewriter.notifyMatchFailure(op, "reduction axis is out of range");
    }
    const int64_t extent = sourceType.getDimSize(axis);
    if (extent < 1) {
      return rewriter.notifyMatchFailure(op, "empty reduction axis");
    }

    // The shape a result has: the source's, with the reduced axis gone. The
    // init slices are rank-reduced to it, which is what linalg.reduce wants
    // for its outputs -- not a kept unit axis.
    SmallVector<int64_t> reducedShape;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != axis) {
        reducedShape.push_back(sourceType.getDimSize(i));
      }
    }

    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> firstOffsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> firstSizes;
    for (int64_t i = 0; i < rank; ++i) {
      firstSizes.push_back(
          rewriter.getIndexAttr(i == axis ? 1 : sourceType.getDimSize(i)));
    }

    // Each init takes the op's own result type (the reduced shape, with the
    // layout encoding a kernel arrived with), and every slice of an operand
    // keeps that operand's encoding: the shapes change, the layout does not.
    SmallVector<Value> inits;
    for (auto [k, operand] : llvm::enumerate(operands)) {
      auto operandType = cast<RankedTensorType>(operand.getType());
      Attribute enc = nullptr;
      if (auto resultType = dyn_cast<RankedTensorType>(op->getResult(k).getType()))
        enc = resultType.getEncoding();
      inits.push_back(tensor::ExtractSliceOp::create(
          rewriter, loc,
          RankedTensorType::get(reducedShape, operandType.getElementType(), enc),
          operand, firstOffsets, firstSizes, strides));
    }

    SmallVector<Value> results;
    if (extent == 1) {
      // NOTHING LEFT TO FOLD. The body is never applied -- a one-element
      // reduction is its own element -- and a `linalg.reduce` over a
      // zero-extent axis would be a shape nothing downstream needs to read.
      results.assign(inits.begin(), inits.end());
    } else {
      SmallVector<OpFoldResult> restOffsets(rank, rewriter.getIndexAttr(0));
      restOffsets[axis] = rewriter.getIndexAttr(1);
      SmallVector<OpFoldResult> restSizes;
      SmallVector<int64_t> restShape;
      for (int64_t i = 0; i < rank; ++i) {
        int64_t size = i == axis ? extent - 1 : sourceType.getDimSize(i);
        restSizes.push_back(rewriter.getIndexAttr(size));
        restShape.push_back(size);
      }

      SmallVector<Value> rest;
      for (auto operand : operands) {
        rest.push_back(tensor::ExtractSliceOp::create(
            rewriter, loc,
            cast<RankedTensorType>(operand.getType()).clone(restShape), operand,
            restOffsets, restSizes, strides));
      }

      auto linalgOp = linalg::ReduceOp::create(
          rewriter, loc, rest, inits, SmallVector<int64_t>{axis},
          [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
            // linalg hands over one argument per input followed by one per
            // init, which is the order tt.reduce's own block uses for the two
            // partials it combines. The body is cloned, not matched.
            auto tritonReduceBlock = op.getBody();
            IRMapping mapping;
            mapping.map(tritonReduceBlock->getArguments(), blockArgs);

            for (auto &innerOp : tritonReduceBlock->without_terminator()) {
              b.clone(innerOp, mapping);
            }

            auto tritonYield = tritonReduceBlock->getTerminator();
            auto yielded =
                llvm::map_to_vector(tritonYield->getOperands(),
                                    [&](Value v) { return mapping.lookup(v); });
            linalg::YieldOp::create(b, loc, yielded);
          });
      results.assign(linalgOp.getResults().begin(),
                     linalgOp.getResults().end());
    }

    // A rank-1 source reduces to a SCALAR, and the slices above are rank-0
    // tensors holding it.
    if (!isa<RankedTensorType>(op.getType(0))) {
      auto elemTypes = op.getElementTypes();
      for (auto [i, result] : llvm::enumerate(results)) {
        results[i] = tensor::ExtractOp::create(rewriter, loc, elemTypes[i],
                                               result, ValueRange{});
      }
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

// get_program_id and get_num_programs:
// When launching triton kernels, we pass 6 additional arguments to indicate
// num_programs and program_id. Amongst those six, we have 3 arguments
// correspond to each axis for num_programs followed by 3 additional arguments
// for program_id.
//
// For instance, with triton kernel example_kernel(a, b, c), we have:
//  example_kernel(
//    a, b, c,
//    num_programs_axis_0,
//    num_programs_axis_1,
//    num_programs_axis_2,
//    program_id_axis_0,
//    program_id_axis_1,
//    program_id_axis_2,
//   )
//
struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

struct GetNumProgramsConverter
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  using OpConversionPattern<triton::GetNumProgramsOp>::OpConversionPattern;

private:
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  GetNumProgramsConverter(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK * 2 + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

// Convert a pair of cmpf and select to either min or max.
// Leave the pattern as simple as possible because triton has plans to emit
// min and max directly.
template <typename CmpOp>
struct MinMaxConverter : public OpRewritePattern<CmpOp> {
  using OpRewritePattern<CmpOp>::OpRewritePattern;

  template <typename T> static T getOnlyUserOfType(Value val) {
    if (!val || !val.hasOneUse()) {
      return nullptr;
    }
    return dyn_cast<T>(*val.getUsers().begin());
  }

  // Only handle the Cmp + OrIOp + Select pattern here.
  static arith::SelectOp findSelectThroughOr(Value cond) {
    if (auto ori = getOnlyUserOfType<arith::OrIOp>(cond)) {
      return getOnlyUserOfType<arith::SelectOp>(ori.getResult());
    }
    return nullptr;
  }

  MinMaxConverter(MLIRContext *context)
      : OpRewritePattern<CmpOp>(context, /*benefit=*/10) {}

  /// Helper that maps a floating-point compare predicate to the
  /// corresponding min/max operation. THis is parametrized by
  /// whether we want NaN-aware operations (MaximumFOp/MinimumFOp) or
  /// numeric operations (MaxNumFOp/MinNumFOp).
  FailureOr<Value> foldCmpToMinMax(PatternRewriter &rewriter, Location loc,
                                   Value lhs, Value rhs,
                                   arith::CmpFPredicate pred,
                                   bool useNaNOps) const {
    switch (pred) {
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::OGE:
      if (useNaNOps) {
        return arith::MaximumFOp::create(rewriter, loc, lhs, rhs).getResult();
      } else {
        return arith::MaxNumFOp::create(rewriter, loc, lhs, rhs).getResult();
      }
      return success();
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::OLE:
      if (useNaNOps) {
        return arith::MinimumFOp::create(rewriter, loc, lhs, rhs).getResult();
      } else {
        return arith::MinNumFOp::create(rewriter, loc, lhs, rhs).getResult();
      }
    default:
      return failure();
    }
  }

  LogicalResult matchAndRewrite(CmpOp cmpOp,
                                PatternRewriter &rewriter) const final {
    Value result = cmpOp.getResult();
    if (!result.hasOneUse()) {
      return failure();
    }

    // 1. Simple pattern: cmpf + select.
    if (auto selectOp = dyn_cast<arith::SelectOp>(*result.getUsers().begin())) {
      if (!(result == selectOp.getCondition() &&
            (cmpOp.getLhs() == selectOp.getTrueValue() &&
             cmpOp.getRhs() == selectOp.getFalseValue()))) {
        return failure();
      }

      rewriteOpWithMinMax(rewriter, cmpOp, selectOp, cmpOp.getPredicate());
      rewriter.eraseOp(cmpOp);
      return success();
    }

    // 2. NaN-aware pattern: cmpf + or + select.
    auto selectOp = findSelectThroughOr(result);
    if (!selectOp) {
      return failure();
    }

    if (failed(foldCmpSelectToMinMax(rewriter, selectOp))) {
      return failure();
    }
    return success();
  }

  /// foldCmpSelectToMinMax performs an optimization pattern that matches
  /// 'arith.select' operations based on a floating-point comparison
  /// and rewrites them into equivalent numeric min/max operations.
  ///
  /// This pattern handles the following case:
  ///
  /// ** NaN-Aware Min/Max Reduction **
  ///    - select (cmpf ogt a, b) || cmpf une a, a), a, b --> arith.maximumf(a,
  ///    b)
  ///    - select (cmpf olt a, b) || cmpf une a, a), a, b --> arith.minimumf(a,
  ///    b)
  ///
  /// These transformations not only improve IR canonicalization but also
  /// allow the successful lowering of tt.reduce operations to linalg
  /// operations, which this conversion pipeline already supports.

  LogicalResult foldCmpSelectToMinMax(PatternRewriter &rewriter,
                                      arith::SelectOp sel) const {

    if (!isa<FloatType>(sel.getType())) {
      return failure();
    }

    Operation *condOp = sel.getCondition().getDefiningOp();
    if (!condOp) {
      return failure();
    }

    Value trueVal = sel.getTrueValue();
    Value falseVal = sel.getFalseValue();

    // NaN-Aware Min/Max Reduction.
    auto ori = dyn_cast<arith::OrIOp>(condOp);
    if (!ori)
      return failure();
    // Extract both sides of the OR condition.
    auto cmp1 = ori.getLhs().getDefiningOp<arith::CmpFOp>();
    auto cmp2 = ori.getRhs().getDefiningOp<arith::CmpFOp>();
    if (!cmp1 || !cmp2)
      return failure();

    // Helper lambdas to identify comparison patterns.
    auto isOGT = [&](arith::CmpFOp cmp) {
      return cmp.getPredicate() == arith::CmpFPredicate::OGT &&
             trueVal == cmp.getLhs() && falseVal == cmp.getRhs();
    };
    auto isOLT = [&](arith::CmpFOp cmp) {
      return cmp.getPredicate() == arith::CmpFPredicate::OLT &&
             trueVal == cmp.getLhs() && falseVal == cmp.getRhs();
    };
    auto isNaN = [&](arith::CmpFOp cmp) {
      return cmp.getPredicate() == arith::CmpFPredicate::UNE &&
             trueVal == cmp.getLhs() && trueVal == cmp.getRhs();
    };

    // Match: select ((ogt(a, b) || une(a, a)), a, b) -> arith.maximumf(a, b).
    if ((isOGT(cmp1) && isNaN(cmp2)) || (isOGT(cmp2) && isNaN(cmp1))) {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(sel);
      FailureOr<Value> foldResult = foldCmpToMinMax(
          rewriter, sel.getLoc(), trueVal, falseVal, arith::CmpFPredicate::OGT,
          /*useNaNOps=*/true);
      if (failed(foldResult)) {
        return failure();
      }
      rewriter.replaceOp(sel, *foldResult);
      return success();
    }

    // Match: select ((olt(a, b) || une(a, a)), a, b) -> arith.minimumf(a, b).
    if ((isOLT(cmp1) && isNaN(cmp2)) || (isOLT(cmp2) && isNaN(cmp1))) {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(sel);
      FailureOr<Value> foldResult = foldCmpToMinMax(
          rewriter, sel.getLoc(), trueVal, falseVal, arith::CmpFPredicate::OLT,
          /*useNaNOps=*/true);
      if (failed(foldResult)) {
        return failure();
      }
      rewriter.replaceOp(sel, *foldResult);
      return success();
    }
    return failure();
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpFOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpFPredicate pred) const {
    FailureOr<Value> foldedResult =
        foldCmpToMinMax(rewriter, selectOp.getLoc(), cmpOp.getLhs(),
                        cmpOp.getRhs(), pred, /*useNaNOps=*/true);
    if (failed(foldedResult)) {
      llvm_unreachable("Unhandled predicate");
    }
    rewriter.replaceOp(selectOp, *foldedResult);
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpIOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<arith::MaxUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<arith::MinSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<arith::MinUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    default:
      llvm_unreachable("Unhandled predicate");
    }
  }
};

struct DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto attr = cast<DenseElementsAttr>(op.getValue());
    auto loc = op.getLoc();

    auto splatConst = arith::ConstantOp::materialize(
        rewriter, attr.getSplatValue<Attribute>(), attr.getElementType(), loc);

    auto init = tensor::EmptyOp::create(
        rewriter, loc, cast<RankedTensorType>(op.getResult().getType()),
        ValueRange{});

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{splatConst},
                                                ValueRange{init});

    return success();
  }
};

class CumSumConverter : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;

  // CumSum is a specific instance of Scan that looks like the following:
  //       %1 = "tt.scan"(%0) <{axis = 1 : i32}> ({
  //       ^bb0(%arg0: f32, %arg1: f32):
  //         %2 = arith.addf %arg0, %arg1 : f32
  //         tt.scan.return %2 : f32
  //       }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  bool isCumSum(triton::ScanOp op) const {
    auto scanBlock = op.getBody();
    auto ops = llvm::map_to_vector(scanBlock->without_terminator(),
                                   [](Operation &op) { return &op; });

    if (ops.size() != 1) {
      return false;
    }

    auto addOp = ops.front();
    if (isa<arith::AddFOp, arith::AddIOp>(addOp)) {
      if (addOp->getResult(0) != scanBlock->getTerminator()->getOperand(0)) {
        return false;
      }

      auto blockArgs =
          llvm::map_range(scanBlock->getArguments(), [](BlockArgument arg) {
            return dyn_cast<Value>(arg);
          });

      auto addArgs = addOp->getOperands();

      return DenseSet<Value>(blockArgs.begin(), blockArgs.end()) ==
             DenseSet<Value>(addArgs.begin(), addArgs.end());
    }

    return false;
  }

public:
  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCumSum(op)) {
      return rewriter.notifyMatchFailure(
          op, "Only support cumsum variant of scan op");
    }

    // ttx::CumSumOp HAS NO PLACE TO PUT `reverse`. It carries an input, an
    // axis and a destination, and nothing else -- so converting a reverse
    // scan silently produced a FORWARD one, with no diagnostic anywhere.
    // Leave it as a tt.scan instead: the region and the reverse flag both
    // survive, and a backend that cannot run it can at least say so.
    if (op.getReverse()) {
      return rewriter.notifyMatchFailure(
          op, "reverse scan cannot be represented as ttx.cumsum");
    }

    auto input = op.getOperand(0);
    auto axis = op.getAxis();
    auto type = dyn_cast<RankedTensorType>(input.getType());

    if (type.getRank() != 1 && type.getRank() != 2 &&
        axis != type.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering scan op to cumsum with rank "
              "= {1, 2} and axis = rank - 1");
    }

    Value init = tensor::EmptyOp::create(rewriter, op.getLoc(), type, ValueRange{});

    auto cumsum = rewriter.replaceOpWithNewOp<ttx::CumSumOp>(
        op, input, rewriter.getUI32IntegerAttr(axis), init);
    // What a pass above wrote on the scan (its discardable attributes) goes along.
    for (NamedAttribute a : op->getDiscardableAttrs())
      cumsum->setAttr(a.getName(), a.getValue());

    return success();
  }
};

class AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op.getResult().getType();
    assert(isa<ShapedType>(resType));
    auto rank = cast<RankedTensorType>(resType).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        /*numResult + numOperands*/ 3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {op.getPtr()};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton op X operating on tensors of pointers to a linalg.generic
// wrapping op X to operate on single pointer.
// This pattern rewriter is almost identical to AddPtrConverter above, except
// that the out param for the linalg op is an empty op instead of reusing one
// of the existing operands. This is because depending on the templatized op,
// the type of the operands might be different, so we cannot pick a default
// operand to reuse for all cases.
template <typename OpType>
class TensorOpConverter : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTensorType =
        dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType) {
      return failure();
    }
    auto rank = resultTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {
        tensor::EmptyOp::create(rewriter, op->getLoc(), resultTensorType, ValueRange{})};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton store op operating on tensors of pointers to a linalg.generic
// wrapping op a triton store op on single pointer.
// Note that this linalg.generic op has an empty `out` param.
class StorePtrToLinalgConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeTensorType = dyn_cast<RankedTensorType>(op.getValue().getType());
    if (!storeTensorType) {
      return failure();
    }
    auto rank = storeTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op.getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs;
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

class ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getSrc();
    auto output = op.getResult();

    auto inputType = input.getType();
    auto outputType = output.getType();
    if (!outputType.hasStaticShape()) {
      return failure();
    }

    if (auto maybeReassociationMap =
            getReassociationIndicesForReshape(inputType, outputType)) {
      auto reassociationMap = *maybeReassociationMap;
      if (outputType.getRank() < inputType.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, outputType, input, reassociationMap);
      } else {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, outputType, input, reassociationMap);
      }
      return success();
    }

    ArrayRef<int64_t> outputShape = outputType.getShape();

    auto shape = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64TensorAttr(outputShape));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outputType, input,
                                                   shape);

    return success();
  }
};

class ExternElementwiseBinaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!op.getPure() || op.getSrcs().size() != 2)
      return failure();
#define POPULATE_BINARY_OP(FUNC_NAME, DST_OP)                                  \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getSrcs()[0], op.getSrcs()[1]); \
    return success();                                                          \
  }

    POPULATE_BINARY_OP("__nv_atan2f", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_atan2", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_powf", math::PowFOp);
    POPULATE_BINARY_OP("__nv_pow", math::PowFOp);
    POPULATE_BINARY_OP("__nv_powif", math::FPowIOp);
    POPULATE_BINARY_OP("__nv_powi", math::FPowIOp);
    POPULATE_BINARY_OP("__nv_copysignf", math::CopySignOp);
    POPULATE_BINARY_OP("__nv_copysign", math::CopySignOp);

#undef POPULATE_BINARY_OP
    return failure();
  }
};

// Three sources: neither converter above matches them, so before this every
// ternary symbol survived as a tt.extern_elementwise with no body.
class ExternElementwiseTernaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getPure() || op.getSrcs().size() != 3)
      return failure();
#define POPULATE_TERNARY_OP(FUNC_NAME, DST_OP)                                 \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getSrcs()[0], op.getSrcs()[1],  \
                                        op.getSrcs()[2]);                      \
    return success();                                                          \
  }

    POPULATE_TERNARY_OP("__nv_fmaf", math::FmaOp);
    POPULATE_TERNARY_OP("__nv_fma", math::FmaOp);

#undef POPULATE_TERNARY_OP
    return failure();
  }
};

class ExternElementwiseUnaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!op.getPure() || op.getSrcs().size() != 1)
      return failure();
#define POPULATE_UNARY_OP(FUNC_NAME, DST_OP)                                   \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getSrcs()[0]);                  \
    return success();                                                          \
  }

    POPULATE_UNARY_OP("__nv_fabsf", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_fabs", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_sinf", math::SinOp);
    POPULATE_UNARY_OP("__nv_sin", math::SinOp);
    POPULATE_UNARY_OP("__nv_cosf", math::CosOp);
    POPULATE_UNARY_OP("__nv_cos", math::CosOp);
    POPULATE_UNARY_OP("__nv_tanf", math::TanOp);
    POPULATE_UNARY_OP("__nv_tan", math::TanOp);
    POPULATE_UNARY_OP("__nv_asinf", math::AsinOp);
    POPULATE_UNARY_OP("__nv_asin", math::AsinOp);
    POPULATE_UNARY_OP("__nv_acosf", math::AcosOp);
    POPULATE_UNARY_OP("__nv_acos", math::AcosOp);
    POPULATE_UNARY_OP("__nv_atanf", math::AtanOp);
    POPULATE_UNARY_OP("__nv_atan", math::AtanOp);
    POPULATE_UNARY_OP("__nv_sinhf", math::SinhOp);
    POPULATE_UNARY_OP("__nv_sinh", math::SinhOp);
    POPULATE_UNARY_OP("__nv_coshf", math::CoshOp);
    POPULATE_UNARY_OP("__nv_cosh", math::CoshOp);
    POPULATE_UNARY_OP("__nv_tanhf", math::TanhOp);
    POPULATE_UNARY_OP("__nv_tanh", math::TanhOp);
    POPULATE_UNARY_OP("__nv_acoshf", math::AcoshOp);
    POPULATE_UNARY_OP("__nv_acosh", math::AcoshOp);
    POPULATE_UNARY_OP("__nv_asinhf", math::AsinhOp);
    POPULATE_UNARY_OP("__nv_asinh", math::AsinhOp);
    POPULATE_UNARY_OP("__nv_atanhf", math::AtanhOp);
    POPULATE_UNARY_OP("__nv_atanh", math::AtanhOp);
    POPULATE_UNARY_OP("__nv_logf", math::LogOp);
    POPULATE_UNARY_OP("__nv_log", math::LogOp);
    POPULATE_UNARY_OP("__nv_log10f", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log10", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log1pf", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_log1p", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_expf", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp2f", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_exp2", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_erff", math::ErfOp);
    POPULATE_UNARY_OP("__nv_erf", math::ErfOp);
    POPULATE_UNARY_OP("__nv_sqrtf", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_sqrt", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrtf", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrt", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_ceilf", math::CeilOp);
    POPULATE_UNARY_OP("__nv_ceil", math::CeilOp);
    POPULATE_UNARY_OP("__nv_floorf", math::FloorOp);
    POPULATE_UNARY_OP("__nv_floor", math::FloorOp);
    POPULATE_UNARY_OP("__nv_truncf", math::TruncOp);
    POPULATE_UNARY_OP("__nv_trunc", math::TruncOp);
    POPULATE_UNARY_OP("__nv_log2f", math::Log2Op);
    POPULATE_UNARY_OP("__nv_log2", math::Log2Op);
    POPULATE_UNARY_OP("__nv_expm1f", math::ExpM1Op);
    POPULATE_UNARY_OP("__nv_expm1", math::ExpM1Op);
    POPULATE_UNARY_OP("__nv_cbrtf", math::CbrtOp);
    POPULATE_UNARY_OP("__nv_cbrt", math::CbrtOp);
    POPULATE_UNARY_OP("__nv_erfcf", math::ErfcOp);
    POPULATE_UNARY_OP("__nv_erfc", math::ErfcOp);
    POPULATE_UNARY_OP("__nv_roundf", math::RoundOp);
    POPULATE_UNARY_OP("__nv_round", math::RoundOp);
    // rint and nearbyint round to nearest under the CURRENT mode, which is
    // ties-to-even unless a program changes it; neither raises inexact here.
    POPULATE_UNARY_OP("__nv_rintf", math::RoundEvenOp);
    POPULATE_UNARY_OP("__nv_rint", math::RoundEvenOp);
    POPULATE_UNARY_OP("__nv_nearbyintf", math::RoundEvenOp);
    POPULATE_UNARY_OP("__nv_nearbyint", math::RoundEvenOp);

#undef POPULATE_UNARY_OP

// The classification ops answer in i1 by op definition (MathOps.td's
// Math_FloatClassificationOp fixes the result to BoolLike), while these symbols
// return int as their C counterparts do, so the answer is widened to the type
// the op being replaced was carrying.
#define POPULATE_UNARY_OP_WIDENED(FUNC_NAME, DST_OP)                           \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    auto bit = DST_OP::create(rewriter, loc, op.getSrcs()[0]);                  \
    rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, op.getType(), bit);         \
    return success();                                                          \
  }

    POPULATE_UNARY_OP_WIDENED("__nv_isnanf", math::IsNaNOp);
    POPULATE_UNARY_OP_WIDENED("__nv_isnand", math::IsNaNOp);
    POPULATE_UNARY_OP_WIDENED("__nv_isinff", math::IsInfOp);
    POPULATE_UNARY_OP_WIDENED("__nv_isinfd", math::IsInfOp);
    // f32 is `finitef` and f64 is `isfinited` -- libdevice drops the `is` on
    // one of the two, and only on this one.
    POPULATE_UNARY_OP_WIDENED("__nv_finitef", math::IsFiniteOp);
    POPULATE_UNARY_OP_WIDENED("__nv_isfinited", math::IsFiniteOp);

#undef POPULATE_UNARY_OP_WIDENED
    return failure();
  }
};

static void populateExternElementwiseOpToMLIROps(RewritePatternSet &patterns) {
  patterns.add<ExternElementwiseBinaryOpConverter,
               ExternElementwiseTernaryOpConverter,
               ExternElementwiseUnaryOpConverter>(patterns.getContext());
}

} // namespace

#endif
