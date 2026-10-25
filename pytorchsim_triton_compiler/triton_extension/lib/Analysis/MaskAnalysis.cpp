//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "TorchsimExt/Analysis/MaskAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include "TorchsimExt/Analysis/OpFoldResultUtils.h"
#include "TorchsimExt/Utils/Encoding.h"
#include "TorchsimExt/Utils/Utils.h"

#include "TorchsimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <limits>

#define DEBUG_TYPE "mask-analysis"

namespace mlir {

namespace triton {

LogicalResult MaskState::parse(Value operand, const Location loc,
                               OpBuilder &builder) {
  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return this->parseConstant(op, loc, builder);
  } else if (isa<IntegerType>(operand.getType())) {
    return this->parseIntScalar(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return this->parseAdd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::SubIOp>()) {
    return this->parseSub(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return this->parseAnd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    return this->parseCmp(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, builder);
  } else if (!operand.getDefiningOp()) {
    return this->parseLoopIterArg(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return this->parseExtSI(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::gpu::ConvertLayoutOp>()) {
    return this->parseConvertLayout(op, loc, builder);
  } else {
    return failure();
  }
}

tensor::ExtractSliceOp MaskState::getExtractSlice(Value source,
                                                  const Location loc,
                                                  OpBuilder &builder) const {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  return tensor::ExtractSliceOp::create(builder, loc, source, offsets, dims,
                                        strides);
}

memref::SubViewOp MaskState::getSubview(Value source, const Location loc,
                                        OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return memref::SubViewOp::create(builder, loc, cast<MemRefType>(dstType),
                                   source, offsets, dims, strides);
}

static memref::SubViewOp createSubview(Value src, Location loc, OpBuilder &b,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return memref::SubViewOp::create(b, loc, cast<MemRefType>(dstType), src,
                                   offsets, sizes, strides);
}

// Assume block1 wraps around and the remainder is block2.
//
// |----------------------|
// |         |            |
// | block2  |  block1    |
// |         |            |
// |----------------------|
//
// Once we copy the chunks in order, the end result is block1 followed by
// block2.
//
//   buffer_tmp:
//
// |----------------------|
// |             |        |
// | block1      | block2 |
// |             |        |
// |----------------------|
//
// Assume we have the following subview:
//
// +++++++++++++++++-------
// +               +      |
// + subview       +      |
// +               +      |
// +++++++++++++++++-------
//
// If we simply take the subview of `buffer_tmp`, this requires an extra
// buffer to just hold the temporary result.
//
// So we can subview into block1 and block2 directly. There are 2 cases:
//   + subview only spans block1
//   + subview spans both block1 and block2, creating sv1 and sv2 (illustrated
//     below for case when we wrap around side-by-side)
//
// |----------------------------------------|
// |                                        |
// |    col2                       col1     |
// |++++++--------|          |+++++++++++++++
// | sv2 + block2 |          | block1 & sv1 +
// |++++++--------|          |+++++++++++++++
// |                                        |
// |----------------------------------------|
//
// For simplicity, assume we only wrap around side-by-side.
//
// Let (row, col1) and (row, col2) be the dimensions of block1 and block2,
// respectively.
//
// Let (rowFull, colFull), (rowView1, colView1) and (rowView2, colView2) be
// the dimensions of the full subview, sv1, and sv2, respectively.
//
// + colView1 = min(colFull, col1)
// + colView2 = colFull - colView1
// + rowView1 = rowView2 = row = rowFull
std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getSideBySideSubviews(Value block1, Value block2, const Location loc,
                                 OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult col1 =
      memref::DimOp::create(builder, loc, block1, 1).getResult();
  OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, builder);
  OpFoldResult subviewCol2 = subOFRs(subviewColFull, subviewCol1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRowFull, subviewCol1}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRowFull, subviewCol2}, strides);

  return {sv1, sv2};
}

std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getStackedSubviews(Value block1, Value block2, const Location loc,
                              OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult row1 =
      memref::DimOp::create(builder, loc, block1, 0).getResult();
  OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, builder);
  OpFoldResult subviewRow2 = subOFRs(subviewRowFull, subviewRow1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRow1, subviewColFull}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRow2, subviewColFull}, strides);
  return {sv1, sv2};
}

LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  start = addOFRs(state.start, scalar, loc, builder);
  end = addOFRs(state.end, scalar, loc, builder);
  dims = state.dims;
  lows = state.lows;
  return success();
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitRemark(loc, "Unexpected case where both lhs and rhs are scalars");
    });
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitRemark(
          loc, "Unsupported scenario where neither lhs nor rhs is a scalar");
    });
    return failure();
  }

  if (lhsState.scalar)
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  else
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
}

LogicalResult MaskState::minStateScalar(const MaskState &lhsState,
                                        const MaskState &rhsState, Location loc,
                                        OpBuilder &builder) {
  // Conjunction where both sides are scalar should not be done after splats. We
  // should ensure that code generation pushes the splat as late as possible.
  if (lhsState.scalar && rhsState.scalar) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitRemark(loc, "Unexpected case where both lhs and rhs are scalars");
    });
    return failure();
  }

  // Caller should ensure that at least one side is scalar.
  if (!lhsState.scalar && !rhsState.scalar) {
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitRemark(
          loc, "Unexpected case where both lhs and rhs are not scalars");
    });
    return failure();
  }

  // If we see a scalar condition in a conjunction with a mask, this means we
  // are either going to take the mask dimension or take nothing at all. To do
  // that we use a select on the scalar value with the mask dimension in the
  // true case and zero in the false case.
  //
  // Example:
  // def kernel(..., index: i32, ...):
  //   ...
  //   offs = tl.arange(0, 8)
  //   mask = offs < 4
  //   scalar = index < 4
  //   ... = tl.load(some_ptr, mask=scalar & mask, other=0)
  auto &scalarState = lhsState.scalar ? lhsState : rhsState;
  auto &nonScalarState = lhsState.scalar ? rhsState : lhsState;
  for (uint32_t i = 0; i < nonScalarState.getRank(); i++) {
    auto nonScalarDim = nonScalarState.dims[i];
    dims.push_back(selectOFRs(scalarState.scalar, nonScalarDim,
                              builder.getZeroAttr(builder.getIndexType()), loc,
                              builder));
  }
  // THE LOW RIDES ALONG UNCHANGED: a false scalar takes the dim to 0, and an
  // empty axis has no first valid lane to be wrong about.
  if (nonScalarState.hasLows())
    lows.assign(nonScalarState.lows.begin(), nonScalarState.lows.end());
  return success();
}

LogicalResult MaskState::minStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.getRank() != rhsState.getRank()) {
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitRemark(
          loc, "Unexpected case where lhs and rhs have different ranks");
    });
    return failure();
  }

  // AN `and` OF TWO MASKS INTERSECTS TWO BOXES, and an extent is only half of
  // one. With no lower bound anywhere the two agree and the cheap form is kept
  // so nothing already working grows an arith op it did not have.
  bool anyLow = lhsState.hasLows() || rhsState.hasLows();
  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    if (!anyLow) {
      dims.push_back(minOFRs(lhsDim, rhsDim, loc, builder));
      continue;
    }
    auto lhsLow = lhsState.lowAt(i, builder);
    auto rhsLow = rhsState.lowAt(i, builder);
    auto newLow = maxOFRs(lhsLow, rhsLow, loc, builder);
    auto newHigh = minOFRs(addOFRs(lhsLow, lhsDim, loc, builder),
                           addOFRs(rhsLow, rhsDim, loc, builder), loc, builder);
    auto newDim = maxOFRs(subOFRs(newHigh, newLow, loc, builder),
                          builder.getIndexAttr(0), loc, builder);
    dims.push_back(newDim);
    lows.push_back(newLow);
  }
  return success();
}

LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location loc, OpBuilder &builder) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && isa<IntegerType>(elementType) &&
           "All elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = builder.getIndexAttr(value.getSExtValue());
    auto op = arith::ConstantOp::materialize(builder, constAttr,
                                             builder.getIndexType(), loc);
    this->scalar = op.getValue();
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    this->scalar = builder.getIndexAttr(value);
  }

  return success();
}

LogicalResult MaskState::parseIntScalar(Value scalar, const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  if (scalar.getType().isInteger(1)) {
    this->scalar = scalar;
  } else {
    auto castOp = arith::IndexCastOp::create(builder, loc,
                                             builder.getIndexType(), scalar);
    this->scalar = castOp.getResult();
  }
  return success();
}

OpFoldResult MaskState::lowAt(unsigned i, OpBuilder &b) const {
  return hasLows() ? lows[i] : OpFoldResult(b.getIndexAttr(0));
}

void MaskState::setLow(unsigned i, OpFoldResult low, OpBuilder &b) {
  if (!hasLows())
    lows.assign(dims.size(), b.getIndexAttr(0));
  lows[i] = low;
}

void MaskState::dump() const {
  llvm::dbgs() << "start: " << start << "\n";
  llvm::dbgs() << "end: " << end << "\n";
  llvm::dbgs() << "scalar: " << scalar << "\n";
  llvm::dbgs() << "useUnsafeMask: " << useUnsafeMask << "\n";
  llvm::dbgs() << "dims: ";
  for (auto dim : dims)
    llvm::dbgs() << "\t" << dim << "\n";
  if (!masks.empty()) {
    llvm::dbgs() << "masks: ";
    for (auto mask : masks)
      llvm::dbgs() << "\t" << mask << "\n";
  }
  llvm::dbgs() << "\n";
}

LogicalResult MaskState::parseSub(arith::SubIOp subOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(subOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(subOp.getRhs(), loc, builder)))
    return failure();

  // ONLY A RANGE MINUS A SCALAR. `scalar - range` runs the lanes backwards and
  // is not a (start, end, dim) at all, and range minus range has no more form
  // than addStates gives it -- both refused, the way addStates refuses them.
  if (lhsState.scalar || !rhsState.scalar || !lhsState.start ||
      !lhsState.end) {
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitRemark(
          loc, "Unsupported scenario where subi is not a range minus a scalar");
    });
    return failure();
  }

  start = subOFRs(lhsState.start, rhsState.scalar, loc, builder);
  end = subOFRs(lhsState.end, rhsState.scalar, loc, builder);
  dims = lhsState.dims;
  lows = lhsState.lows;
  return success();
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder)))
    return failure();

  return this->addStates(lhsState, rhsState, loc, builder);
}

// Whether `state` can say anything about dimension `dim`.
//
// `parseAnd`'s bool arm below deliberately tolerates a FAILED parse of either
// side -- an i1 mask is merged element-wise, so one side being unanalysable is
// not by itself fatal. What it left unsaid is that such a state describes
// nothing: `isMask()` is false for it (no dims) and its `scalar` is null, so
// the ternary that picks one of the two reads a null OpFoldResult and
// `getIntAttr` dereferences it.
//
//     PtrAnalysis::rewriteLoadOp -> MaskState::parse -> parseAnd
//     -> getIntAttr -> OpFoldResultUtils.cpp:23, segfault
//
// Measured on DeepSeek-V4-Flash-Base, whose softmax kernel masks with an `and`
// of a comparison this analysis does not model. Refusing here is the same
// answer the analysis gives for any mask it cannot describe: the load stays on
// the unstructured path.
static bool describesDim(const MaskState &state, unsigned dim) {
  if (state.isMask())
    return dim < state.dims.size() && static_cast<bool>(state.dims[dim]);
  return static_cast<bool>(state.scalar);
}

static LogicalResult andDigits(MaskState &out, const MaskState &lhs,
                               const MaskState &rhs, ArrayRef<int64_t> shape,
                               OpBuilder &builder);

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  bool isBoolOp = false;
  unsigned rank = 1;
  if (auto shapedType = dyn_cast<ShapedType>(andOp.getType())) {
    isBoolOp = shapedType.getElementType().isInteger(1);
    rank = shapedType.getRank();
  }
  MaskState lhsState;
  LogicalResult lResult = lhsState.parse(andOp.getLhs(), loc, builder);
  if (failed(lResult) && !isBoolOp) {
    return failure();
  }

  MaskState rhsState;
  LogicalResult rResult = rhsState.parse(andOp.getRhs(), loc, builder);
  if (failed(rResult) && !isBoolOp) {
    return failure();
  }
  bool anyDigits = lhsState.hasDigits() || rhsState.hasDigits();
  if (anyDigits && (failed(lResult) || failed(rResult)))
    return failure();

  if (isBoolOp) {
    if (lhsState.masks.size() != rank) {
      return failure();
    }

    if (lhsState.masks.size() != rhsState.masks.size()) {
      return failure();
    }

    // merge the masks.
    if (lhsState.masks.size() == rhsState.masks.size()) {
      auto shapedType = cast<ShapedType>(andOp.getType());
      assert(shapedType.hasStaticShape());
      for (size_t i = 0; i < lhsState.masks.size(); i++) {
        Value lhsV = lhsState.masks[i];
        Value rhsV = rhsState.masks[i];
        if (!lhsV && !rhsV) {
          masks.push_back(nullptr);
        } else {
          uint32_t size = shapedType.getShape()[i];
          // The mask rebuilt here takes the layout of the mask it stands in
          // for (the and's own), sliced to the dim it runs along -- the same
          // layout parseCmp gives the mask it is and-ed with.
          auto andType = cast<RankedTensorType>(andOp.getType());
          auto structuredMaskToUnstructuredMask = [&](MaskState state,
                                                      unsigned dim,
                                                      uint32_t size,
                                                      OpBuilder &builder,
                                                      Location loc) {
            OpFoldResult ofr = state.isMask() ? state.dims[dim] : state.scalar;
            if (auto intV = getIntAttr(ofr)) {
              if (intV == size) {
                // Full mask.
                return Value();
              }
            }
            Attribute enc = triton::axisEncoding(andType.getEncoding(),
                                                 andType.getRank(), dim);
            auto targetTensorType =
                RankedTensorType::get({size}, builder.getI32Type(), enc);
            Value range = triton::MakeRangeOp::create(builder, loc,
                                                      targetTensorType, 0, size)
                              .getResult();
            Value v = ofrToIndexValue(ofr, loc, builder);
            v = arith::IndexCastUIOp::create(builder, loc, builder.getI32Type(),
                                             v)
                    .getResult();
            v = triton::SplatOp::create(builder, loc, targetTensorType, v)
                    .getResult();
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::ult, range, v)
                .getResult();
          };
          if (!lhsV) {
            if (!describesDim(lhsState, i)) {
              return failure();
            }
            lhsV = structuredMaskToUnstructuredMask(lhsState, i, size, builder,
                                                    loc);
          } else if (!rhsV) {
            if (!describesDim(rhsState, i)) {
              return failure();
            }
            rhsV = structuredMaskToUnstructuredMask(rhsState, i, size, builder,
                                                    loc);
          }
          if (!lhsV) {
            masks.push_back(rhsV);
            continue;
          } else if (!rhsV) {
            masks.push_back(lhsV);
            continue;
          }
          // And the mask.
          masks.push_back(arith::AndIOp::create(builder, loc, lhsV, rhsV));
        }
      }
      // Only support one unstructured mask.
      if (getUnstructuredMasks().size() > 1) {
        return failure();
      }
    }
  }

  if (anyDigits) {
    auto shape = cast<ShapedType>(andOp.getType()).getShape();
    if (!lhsState.isMask() || !rhsState.isMask() ||
        failed(this->minStates(lhsState, rhsState, loc, builder)))
      return failure();
    return andDigits(*this, lhsState, rhsState, shape, builder);
  }
  if (!lhsState.isMask() || !rhsState.isMask()) {
    return this->minStateScalar(lhsState, rhsState, loc, builder);
  }
  return this->minStates(lhsState, rhsState, loc, builder);
}

// A layout conversion moves a predicate's elements, never which are true: for a mask it is the
// identity, so the analysis looks through it as PtrAnalysis does for an address. What it moves are
// the unstructured masks parsed beyond it, which are in the source's layout: each is converted to
// the conversion's own along its axis, the layout a mask rebuilt here (parseAnd) and a compare
// parsed here (parseCmp) take. A failed parse can still leave masks (parseExpandDims reads them).
LogicalResult MaskState::parseConvertLayout(triton::gpu::ConvertLayoutOp op,
                                            const Location loc,
                                            OpBuilder &builder) {
  LogicalResult result = this->parse(op.getSrc(), loc, builder);
  auto dstType = dyn_cast<RankedTensorType>(op.getType());
  if (!dstType)
    return result;
  for (int64_t dim = 0, e = std::min<int64_t>(masks.size(), dstType.getRank());
       dim < e; dim++) {
    Value &mask = masks[dim];
    if (!mask)
      continue;
    auto maskType = cast<RankedTensorType>(mask.getType());
    Attribute enc =
        triton::axisEncoding(dstType.getEncoding(), dstType.getRank(), dim);
    if (maskType.getEncoding() == enc)
      continue;
    mask = triton::gpu::ConvertLayoutOp::create(
        builder, loc,
        RankedTensorType::get(maskType.getShape(), maskType.getElementType(),
                              enc),
        mask);
  }
  return result;
}

LogicalResult MaskState::parseExtSI(arith::ExtSIOp op, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());
  return parse(op.getIn(), loc, builder);
}

// PAIRS, NOT TWO LISTS: `-3 % 8` read unsigned is 4294967293 and `< 8` is
// false, which is the cell a product of the two would fold to true.
//
// A REMAINDER COMPARED AGAINST ITS MODULUS OR MORE STATES NOTHING -- `x % K`
// is smaller than K by definition, so any bound at K or above excludes no
// element. Left standing it is a `%` inside a mask, which nothing here reads,
// and the access falls to a gather over a bound that could not have excluded
// anything.
//
// AT OR ABOVE, not exactly at: this asked for equality and a kernel masking
// `t % 8` with a BLOCK of 16 fell two rungs for a bound that is always true.
static bool remainderBoundIsAlwaysTrue(arith::CmpIOp cmpOp) {
  Operation *def = cmpOp.getLhs().getDefiningOp();
  if (!def)
    return false;
  auto pred = cmpOp.getPredicate();
  bool sound =
      (isa<arith::RemSIOp>(def) && pred == arith::CmpIPredicate::slt) ||
      (isa<arith::RemUIOp>(def) && (pred == arith::CmpIPredicate::slt ||
                                    pred == arith::CmpIPredicate::ult));
  if (!sound)
    return false;
  auto k = triton::constantIntOrSplat(def->getOperand(1));
  auto bound = triton::constantIntOrSplat(cmpOp.getRhs());
  return k && bound && *k > 0 && *k <= *bound;
}

// `(x // K) < c` IS `x < c*K` for positive c and K -- exact for every x,
// negative x included, because a quotient truncated toward zero is at most 0
// there while `x < c*K` is true anyway. Answers the value the compare should
// have been written on and updates `bound` to the one it should be read
// against; a chained division peels one level per turn.
//
// AND THE PAIRING IS THE SOUNDNESS AGAIN: `-1 divui 8` is 536870911 and not
// less than c, while `-1 < c*K` is true. The division and the compare have to
// read the dividend the same way.
static Value foldBoundThroughDiv(Value lhs, arith::CmpIPredicate pred,
                                 std::optional<int64_t> &bound) {
  for (unsigned depth = 0; depth < 8; depth++) {
    Operation *def = lhs.getDefiningOp();
    if (!def || !bound || *bound <= 0)
      break;
    bool sound =
        (isa<arith::DivSIOp>(def) && pred == arith::CmpIPredicate::slt) ||
        (isa<arith::DivUIOp>(def) && pred == arith::CmpIPredicate::ult);
    if (!sound)
      break;
    auto k = triton::constantIntOrSplat(def->getOperand(1));
    if (!k || *k <= 0 || *bound > std::numeric_limits<int64_t>::max() / *k)
      break;
    bound = *bound * *k;
    lhs = def->getOperand(0);
  }
  return lhs;
}

// A NON-NEGATIVE MULTIPLE OF `k`, structurally: a constant, a program id
// scaled by one, and the sums and casts that keep both. `x % k` over a range
// that starts here and runs whole laps of `k` is the lane's place in its lap.
static bool isNonNegativeMultipleOf(Value v, int64_t k, unsigned depth = 0) {
  if (!v || depth > 8)
    return false;
  if (auto c = triton::constantIntOrSplat(v))
    return *c >= 0 && *c % k == 0;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<arith::ExtSIOp, arith::IndexCastOp, triton::SplatOp>(def))
    return isNonNegativeMultipleOf(def->getOperand(0), k, depth + 1);
  if (isa<arith::AddIOp>(def))
    return isNonNegativeMultipleOf(def->getOperand(0), k, depth + 1) &&
           isNonNegativeMultipleOf(def->getOperand(1), k, depth + 1);
  if (isa<arith::MulIOp>(def))
    for (unsigned i : {0u, 1u}) {
      auto c = triton::constantIntOrSplat(def->getOperand(i));
      Value other = def->getOperand(1 - i);
      if (c && *c >= 0 && *c % k == 0 &&
          (other.getDefiningOp<triton::GetProgramIdOp>() ||
           isNonNegativeMultipleOf(other, 1, depth + 1)))
        return true;
    }
  return false;
}

static bool isNonNegativeMultipleOf(OpFoldResult ofr, int64_t k) {
  if (auto c = getConstantIntValue(ofr))
    return *c >= 0 && *c % k == 0;
  return isNonNegativeMultipleOf(dyn_cast<Value>(ofr), k);
}

// `((x // D) % K) + c` AGAINST A CONSTANT IS A BOX ON ONE DIGIT of whole laps
// of D*K from a lap, when no add of c wraps: [lo, hi) of the digit, the whole
// axis for a bound it never crosses, an empty axis for one it never meets.
LogicalResult MaskState::parsePeriodicCmp(arith::CmpIOp cmpOp, int cmpOpDim,
                                          const Location loc,
                                          OpBuilder &builder) {
  auto ty = dyn_cast<RankedTensorType>(cmpOp.getType());
  auto c = triton::constantIntOrSplat(cmpOp.getRhs());
  if (!ty || !ty.hasStaticShape() || cmpOpDim < 0 || !c)
    return failure();
  Value l = cmpOp.getLhs();
  SmallVector<std::pair<int64_t, unsigned>> shifts;
  for (unsigned depth = 0; depth < 8; depth++) {
    if (auto ext = l.getDefiningOp<arith::ExtSIOp>()) {
      l = ext.getIn();
      continue;
    }
    Operation *def = l.getDefiningOp();
    if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp>(def))
      break;
    auto a = triton::constantIntOrSplat(def->getOperand(0));
    auto b = triton::constantIntOrSplat(def->getOperand(1));
    unsigned bits = getElementTypeOrSelf(def->getResult(0).getType())
                        .getIntOrFloatBitWidth();
    if (b && !a) {
      shifts.push_back({isa<arith::AddIOp>(def) ? *b : -*b, bits});
      l = def->getOperand(0);
    } else if (a && !b && isa<arith::AddIOp>(def)) {
      shifts.push_back({*a, bits});
      l = def->getOperand(1);
    } else {
      break;
    }
  }
  auto rem = l.getDefiningOp<arith::RemSIOp>();
  if (!rem)
    return failure();
  auto k = triton::constantIntOrSplat(rem.getRhs());
  if (!k || *k <= 1)
    return failure();
  int64_t shift = 0;
  for (auto [add, bits] : llvm::reverse(shifts)) {
    shift += add;
    int64_t lim = int64_t(1) << (std::clamp(bits, 2u, 32u) - 2);
    if (std::abs(shift) >= lim || *k >= lim)
      return failure();
  }
  Value x = rem.getLhs();
  int64_t inner = 1;
  for (unsigned depth = 0; depth < 8; depth++) {
    auto div = x.getDefiningOp<arith::DivSIOp>();
    if (!div)
      break;
    auto d = triton::constantIntOrSplat(div.getRhs());
    if (!d || *d <= 0 || inner > std::numeric_limits<int32_t>::max() / *d)
      return failure();
    inner *= *d;
    x = div.getLhs();
  }
  if (inner > std::numeric_limits<int32_t>::max() / *k ||
      ty.getShape()[cmpOpDim] % (inner * *k) != 0)
    return failure();
  int64_t bound = *c - shift, lo = 0, hi = *k;
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::slt:
    hi = std::clamp<int64_t>(bound, 0, *k);
    break;
  case arith::CmpIPredicate::sge:
    lo = std::clamp<int64_t>(bound, 0, *k);
    break;
  case arith::CmpIPredicate::sgt:
    lo = std::clamp<int64_t>(bound + 1, 0, *k);
    break;
  default:
    return failure();
  }
  MaskState range;
  if (failed(range.parse(x, loc, builder)) || range.scalar || !range.start ||
      range.hasDigits() || !isNonNegativeMultipleOf(range.start, inner * *k))
    return failure();
  dims.clear();
  lows.clear();
  for (int64_t i = 0; i < ty.getRank(); i++)
    dims.push_back(
        builder.getIndexAttr(i == cmpOpDim && lo >= hi ? 0 : ty.getShape()[i]));
  masks.assign(ty.getRank(), nullptr);
  digits.assign(ty.getRank(), {});
  if (lo < hi && (lo != 0 || hi != *k))
    digits[cmpOpDim].push_back({inner, *k, lo, hi});
  return success();
}

LogicalResult MaskState::addDigit(unsigned i, Digit d) {
  auto &ds = digits[i];
  for (auto &e : ds) {
    if (e.inner == d.inner && e.radix == d.radix) {
      e.lo = std::max(e.lo, d.lo);
      e.hi = std::min(e.hi, d.hi);
      return success(e.lo < e.hi);
    }
    const Digit &below = e.inner < d.inner ? e : d;
    const Digit &above = e.inner < d.inner ? d : e;
    if (above.inner % (below.inner * below.radix) != 0)
      return failure();
  }
  ds.insert(llvm::upper_bound(ds, d,
                              [](const Digit &a, const Digit &b) {
                                return a.inner < b.inner;
                              }),
            d);
  return success();
}

// AN `and` OF DIGIT BOXES IS A BOX ON THE UNION OF THEIR DIGITS, where every
// axis that has digits keeps its plain bound whole and both sides are boxes; an
// axis bounded to nothing leaves nothing, and no digits.
static LogicalResult andDigits(MaskState &out, const MaskState &lhs,
                               const MaskState &rhs, ArrayRef<int64_t> shape,
                               OpBuilder &builder) {
  if (!lhs.isMask() || !rhs.isMask() ||
      llvm::any_of(lhs.masks, [](Value m) { return bool(m); }) ||
      llvm::any_of(rhs.masks, [](Value m) { return bool(m); }))
    return failure();
  out.digits.assign(shape.size(), {});
  if (llvm::any_of(out.dims, [](OpFoldResult d) {
        return getConstantIntValue(d) == std::optional<int64_t>(0);
      }))
    return success();
  for (const MaskState *side : {&lhs, &rhs})
    for (auto [i, ds] : llvm::enumerate(side->digits))
      for (const auto &d : ds)
        if (i >= shape.size() || failed(out.addDigit(i, d)))
          return failure();
  for (auto [i, ds] : llvm::enumerate(out.digits)) {
    if (ds.empty())
      continue;
    auto dim = getConstantIntValue(out.dims[i]);
    auto low = getConstantIntValue(out.lowAt(i, builder));
    if (!dim || *dim != shape[i] || !low || *low != 0)
      return failure();
  }
  return success();
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  // A TERM THAT CANNOT BE FALSE IS THE WHOLE EXTENT. Answered before the
  // unstructured fallback below is built, so nothing is minted for a mask that
  // turns out to say nothing.
  if (remainderBoundIsAlwaysTrue(cmpOp)) {
    auto ty = dyn_cast<ShapedType>(cmpOp.getType());
    if (!ty || !ty.hasStaticShape())
      return failure();
    for (auto d : ty.getShape()) {
      dims.push_back(builder.getIndexAttr(d));
      masks.push_back(nullptr);
    }
    return success();
  }
  int cmpOpDim = -1;
  if (auto shapedType = dyn_cast<ShapedType>(cmpOp.getType())) {
    for (unsigned r = 0; r < shapedType.getRank(); r++) {
      if (shapedType.getShape()[r] != 1) {
        if (cmpOpDim != -1) {
          // This will happen when the cmp has more than one dimension with size
          // larger than 1.
          // Like a < b while both a and b are tensors with shape 2x2.
          cmpOpDim = -1;
          break;
        }
        cmpOpDim = r;
      }
    }
    masks.clear();
    for (unsigned r = 0; r < shapedType.getRank(); r++) {
      masks.push_back(nullptr);
    }
    // If cmpOpDim == -1, parseCmp must fail later.
    // Here just setup unstructured masks when cmpOpDim != -1.
    if (cmpOpDim != -1) {
      // Save cmpOp as unstructured mask for failure case, will recover it to
      // nullptr later if success.
      Value unstructuredMask = cmpOp;
      if (shapedType.getRank() > 1) {
        // If cmpOp is not 1D, collapse it to 1D.
        // Every other axis is a unit, so the flat mask is the slice of the
        // compare's layout along them.
        Attribute flatEnc;
        if (auto ranked = dyn_cast<RankedTensorType>(shapedType))
          flatEnc = triton::axisEncoding(ranked.getEncoding(), ranked.getRank(),
                                         cmpOpDim);
        auto flatType = RankedTensorType::get({shapedType.getShape()[cmpOpDim]},
                                              shapedType.getElementType(), flatEnc);
        auto maybeReassociationMap =
            getReassociationIndicesForReshape(shapedType, flatType);
        SmallVector<ReassociationIndices> reassociation =
            *maybeReassociationMap;
        // Set masks.
        unstructuredMask = tensor::CollapseShapeOp::create(
            builder, loc, flatType, cmpOp, reassociation);
      }
      masks[cmpOpDim] = unstructuredMask;
    }
  } else {
    cmpOpDim = 0;
    masks.push_back(cmpOp);
  }
  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
      cmpOp.getPredicate() != arith::CmpIPredicate::ult &&
      cmpOp.getPredicate() != arith::CmpIPredicate::sgt &&
      cmpOp.getPredicate() != arith::CmpIPredicate::sge) {
    LLVM_DEBUG(
        { InFlightDiagnostic diag = emitRemark(loc, "Unsupported cmpi"); });
    return failure();
  }

  if (succeeded(parsePeriodicCmp(cmpOp, cmpOpDim, loc, builder)))
    return success();

  // A BOUND ON A QUOTIENT IS A BOUND ON THE DIVIDEND, and only the second form
  // is a range this analysis can place on a lane.
  auto folded = triton::constantIntOrSplat(cmpOp.getRhs());
  Value lhsValue = foldBoundThroughDiv(cmpOp.getLhs(), cmpOp.getPredicate(),
                                       folded);

  MaskState lhsState;
  if (failed(lhsState.parse(lhsValue, loc, builder)))
    return failure();

  MaskState rhsState;
  if (lhsValue != cmpOp.getLhs())
    rhsState.scalar = builder.getIndexAttr(*folded);
  else if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder)))
    return failure();

  int32_t cmpDim = lhsState.scalar && rhsState.scalar ? 0 : -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      if (cmpDim != -1) {
        LLVM_DEBUG({
          InFlightDiagnostic diag =
              emitRemark(loc, "Unsupported cmpi with more than one dimension "
                              "with size larger than 1");
        });
        return failure();
      }
      cmpDim = i;
    }
  }
  // NO AXIS MEANS NOWHERE TO PUT A LOWER BOUND. The assert below reads as if
  // it admitted sge, but its arms bind as (cmpDim != -1) || (slt) || (ult), so
  // a lower bound with no sized axis has always aborted here; refused instead.
  if (cmpDim == -1 && (cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
                       cmpOp.getPredicate() == arith::CmpIPredicate::sgt)) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitRemark(loc, "Unsupported cmpi lower bound with no sized axis");
    });
    return failure();
  }
  assert(
      cmpDim != -1 ||
      (!lhsState.scalar && cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
       cmpOp.getPredicate() == arith::CmpIPredicate::ult) &&
          "Unexpected case where no dimension has size larger than 1");

  OpFoldResult newDim, newLow;
  if (lhsState.scalar) {
    assert(rhsState.scalar && "Unexpected case where rhs is not a scalar");
    // If both lhs and rhs are scalars, we can't just derive the dimension of
    // the mask as the minimum value: lhs/rhs could be 0 and then we don't
    // load/store anything.
    //
    // Instead treat the comparison as a scalar that determines if anything
    // should be loaded/stored by inserting a comparison + select:
    //    dim = lhs < rhs ? lhs.dim : 0
    newDim = compareOFRs(lhsState.scalar, rhsState.scalar, cmpOp.getPredicate(),
                         lhsState.dims[cmpDim], builder.getIndexAttr(0), loc,
                         builder);
  } else if (cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
             cmpOp.getPredicate() == arith::CmpIPredicate::ult) {
    // Important:
    // In the case where the values we are loading are entirely masked off like
    // the following:
    //
    // ---|-------|-----------|
    //    ^       ^           ^
    //   scalar  start       end
    //
    // newEnd = min(end, scalar) = scalar
    // Now scalar < start, so simply doing dim = newEnd - start is incorrect.
    //
    // The correct formula is to optionally move `newDim` back to `start` using
    // max(newEnd, start).
    auto newEnd = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
    newEnd = maxOFRs(newEnd, lhsState.start, loc, builder);
    newDim = subOFRs(newEnd, lhsState.start, loc, builder);
  } else {
    // A LOWER BOUND ON THE VALUE IS A LOWER BOUND ON THE LANE only after the
    // range's own start is taken out: lane l holds start + l, so `value >= c`
    // is `l >= c - start`. Folding it away as always-true -- which is what
    // this arm did, guarded by `c == 0` -- is right only while start >= 0.
    assert(cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
           cmpOp.getPredicate() == arith::CmpIPredicate::sgt);
    // THE SAME ARITHMETIC WHETHER OR NOT THE ENDS ARE NUMBERS. `value >= c` is
    // `lane >= c - start`, so the first live lane is that difference clamped
    // into the axis -- and tts.load carries a low as a value, so a runtime
    // start needs no assumption about its sign. It used to get one: `>= 0`
    // over a runtime start was folded away as the boundary_check shape, whose
    // start is a program id. Inductor's padded conv breaks that -- its start
    // is `i - PADDING + h`, which really does go negative.
    OpFoldResult bound = rhsState.scalar;
    if (cmpOp.getPredicate() == arith::CmpIPredicate::sgt)
      bound = addOFRs(bound, builder.getIndexAttr(1), loc, builder);
    OpFoldResult start = lhsState.start ? lhsState.start
                                        : OpFoldResult(builder.getIndexAttr(0));
    OpFoldResult lo = subOFRs(bound, start, loc, builder);
    lo = maxOFRs(lo, builder.getIndexAttr(0), loc, builder);
    lo = minOFRs(lo, lhsState.dims[cmpDim], loc, builder);
    // A LOW OF ZERO IS NO LOW, and saying so keeps every mask that folded to
    // always-true spelled exactly as it was before this arm could carry one.
    if (auto c = getConstantIntValue(lo); c && *c == 0) {
      newDim = lhsState.dims[cmpDim];
    } else {
      newLow = lo;
      newDim = subOFRs(lhsState.dims[cmpDim], lo, loc, builder);
    }
  }

  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    if (i == cmpDim)
      this->dims.push_back(newDim);
    else
      this->dims.push_back(lhsState.dims[i]);
  }
  if (lhsState.hasLows())
    this->lows.assign(lhsState.lows.begin(), lhsState.lows.end());
  if (newLow)
    setLow(cmpDim, newLow, builder);
  if (cmpOpDim != -1) {
    // Clear masks when success.
    masks[cmpOpDim] = nullptr;
  }
  return success();
}

LogicalResult MaskState::parseLoopIterArg(Value v, const Location loc,
                                          OpBuilder &builder) {
  assert(!v.getDefiningOp());

  auto forOp = llvm::dyn_cast<scf::ForOp>(v.getParentRegion()->getParentOp());

  if (!forOp) {
    return failure();
  }

  auto it = llvm::find(forOp.getRegionIterArgs(), v);
  if (it == forOp.getRegionIterArgs().end()) {
    return failure();
  }

  // This is a bit of a hack!!
  //
  // The offset (MaskState::start) of a mask can now depend on a loop's
  // iter-arg like the following example:
  //
  // idx = offset + tl.arange(0, 4)
  // for it in range(n):
  //   mask = idx < size
  //   x = tl.load(x_ptr + idx, mask=mask)
  //   tl.store(y_ptr + idx, x, mask=mask)
  //   idx += 4
  //
  // See
  // test/Conversion/TritonToStructured/mask_loop_iter_arg.mlir and
  // and
  // python/examples/test_mask_loop_iter_arg.py
  // for IR and full triton code.
  //
  // To support this case, we first make the following assumptions:
  //  - MaskAnalysis is runs after PtrAnalysis's prepass finishes, which means
  //    the offset for the load and store pointers have already been set up
  //    at `argIndex + 1`
  //  - The tensor of indices used by the load / store and the mask are the same
  //    (see above where `idx` appears in both the mask and the pointer
  //    arithmetic). This allows us to use the offset at `argIndex + 1` in the
  //    above assumption. In the future, to make this more robust, we need to
  //    verify that the offsets are indeed the same. Or alternatively, make sure
  //    to generate a separate start and end offset for each mask that is being
  //    updated in loops.
  //
  // Now to generate the mask state in each loop iteration, we first construct
  // the mask state *before* coming into the loop by parsing the init-arg. A
  // mask dimensions stay consistent throughout each loop iteration, but its
  // starting offset (`MaskState::start`) will change. So to construct the mask
  // state for each iteration, we need to make MaskState::state be the offset
  // iter-arg at `argIndex + 1`. Now for `MaskState::end`, we can first compute
  // the distance between `start` and `end` before coming into the loop, then
  // use this distance to compute the actual `end` in each loop.
  auto argIndex = std::distance(forOp.getRegionIterArgs().begin(), it);

  // NESTED LOOPS: THE WIDTH IS MADE OUT THERE, THE POSITION IS READ IN HERE.
  // An inner loop's init-arg is the enclosing loop's iter-arg, which has no
  // defining op, so the state op that says what this value IS sits some levels
  // up. Walk out to it, keeping the loops so an index can be mapped back in.
  SmallVector<scf::ForOp> loops{forOp};   // innermost first
  scf::ForOp outer = forOp;
  size_t outerIndex = argIndex;
  Value initArg = outer.getInitArgs()[outerIndex];
  while (!initArg.getDefiningOp()) {
    auto parent =
        llvm::dyn_cast<scf::ForOp>(initArg.getParentRegion()->getParentOp());
    if (!parent) {
      return failure();
    }
    auto pit = llvm::find(parent.getRegionIterArgs(), initArg);
    if (pit == parent.getRegionIterArgs().end()) {
      return failure();
    }
    outerIndex = std::distance(parent.getRegionIterArgs().begin(), pit);
    outer = parent;
    loops.push_back(outer);
    initArg = outer.getInitArgs()[outerIndex];
  }

  // An iter-arg index at `outer` as this loop numbers it. The prepass threads
  // the same values through every level, but which POSITION they land in is
  // the loop's to say, so it is followed rather than assumed.
  auto mapInward = [&](size_t idx) -> std::optional<size_t> {
    for (int i = static_cast<int>(loops.size()) - 2; i >= 0; --i) {
      Value carried = loops[i + 1].getRegionIterArg(idx);
      auto bit = llvm::find(loops[i].getInitArgs(), carried);
      if (bit == loops[i].getInitArgs().end()) {
        return std::nullopt;
      }
      idx = std::distance(loops[i].getInitArgs().begin(), bit);
    }
    return idx;
  };

  if (auto getStateOp = initArg.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto tritonValue = getStateOp->getOperand(0);
    MaskState lhsState;

    {
      OpBuilder::InsertionGuard guard(builder);
      // Make sure all ops generated for the mask state are inserted before
      // the OUTERMOST loop, so they are loop-invariant and dominate every use.
      builder.setInsertionPoint(outer);
      if (failed(lhsState.parse(tritonValue, loc, builder))) {
        return failure();
      }
    }

    if (!lhsState.start && !lhsState.end) {
      assert(lhsState.scalar && "MaskState must have a scalar");
      lhsState.start = builder.getIndexAttr(0);
      lhsState.end = lhsState.scalar;
    }

    auto dist = subOFRs(lhsState.end, lhsState.start, loc, builder);

    // `start` is a scalar, and the line below calls the FIRST offset iter-arg
    // the start -- true only when there is exactly one.
    if (getStateOp.getOffsets().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "MaskAnalysis: a loop index with more than one "
                                 "offset has no single start to shift\n");
      return failure();
    }
    auto offsetIndex = mapInward(outerIndex + 1);
    if (!offsetIndex) {
      LLVM_DEBUG(llvm::dbgs() << "MaskAnalysis: the offset is not carried all "
                                 "the way in to this loop\n");
      return failure();
    }
    this->start = forOp.getRegionIterArg(*offsetIndex);

    // The base offset is part of start: it rides its own slot now, so the
    // offset iter-arg alone is no longer where the index sits. Its position
    // comes from the op, not from adding up the other segments' sizes.
    if (!getStateOp.getBaseOffsets().empty()) {
      auto baseArg = getStateOp.getBaseOffsets().front();
      auto baseIt = llvm::find(outer.getInitArgs(), baseArg);
      if (baseIt == outer.getInitArgs().end()) {
        LLVM_DEBUG(llvm::dbgs() << "MaskAnalysis: the base offset is not "
                                   "carried by this loop\n");
        return failure();
      }
      auto baseIndex =
          mapInward(std::distance(outer.getInitArgs().begin(), baseIt));
      if (!baseIndex) {
        LLVM_DEBUG(llvm::dbgs() << "MaskAnalysis: the base offset is not "
                                   "carried all the way in to this loop\n");
        return failure();
      }
      this->start = addOFRs(
          this->start, forOp.getRegionIterArg(*baseIndex), loc, builder);
    }
    this->end = addOFRs(this->start, dist, loc, builder);
    this->dims = lhsState.dims;

    return success();
  }

  return failure();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitRemark(
          loc, "stride must be 1 for make_range whose result is used "
               "as load or store masks");
    });
    return failure();
  }

  this->start = builder.getIndexAttr(start);
  this->end = builder.getIndexAttr(end);
  this->dims.push_back(builder.getIndexAttr(shape[0]));

  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, builder)))
    return failure();

  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i]) {
      this->dims[i] = builder.getIndexAttr(dstShape[i]);
      // The axis is now every lane, so whatever the low said about the one
      // lane it was broadcast from no longer names anything.
      if (hasLows())
        this->lows[i] = builder.getIndexAttr(0);
    } else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }

  return success();
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (!isa<IntegerType>(src.getType())) {
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitRemark(
          loc, "splat source must be an integer scalar for load/store masks");
    });
    return failure();
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  for (auto s : dstShape)
    this->dims.push_back(builder.getIndexAttr(s));
  bool isBool = src.getType().isInteger(1);
  if (isBool) {
    // If src is a 1D boolean tensor and parse success.
    // Create masks.
    masks.clear();
    for (unsigned i = 0; i < dstShape.size(); i++) {
      masks.push_back(nullptr);
    }
  }
  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  Value src = expandDimsOp.getSrc();
  auto srcType = cast<ShapedType>(src.getType());
  bool isBoolOp = srcType.getElementType().isInteger(1);
  LogicalResult result = parse(src, loc, builder);
  if (hasDigits() && failed(result))
    return failure();
  if (failed(result)) {
    if (isBoolOp) {
      if (srcType.getRank() > 1 && masks.size() != srcType.getRank()) {
        return failure();
      }
    } else {
      return failure();
    }
  }

  if (isBoolOp) {
    // Save mask for 1D boolean tensor
    if (srcType.getRank() == 1) {
      assert(dstShape.size() == 2);
      masks.resize(dstShape.size());
      masks[axis] = nullptr;
      if (failed(result)) {
        // Recover dims to allow other dim to be processed.
        dims.clear();
        lows.clear();
        dims.push_back(builder.getIndexAttr(srcType.getShape()[0]));
        // Save src as unstructured mask.
        masks[1 - axis] = src;
      } else {
        // save nullptr when parse success.
        masks[1 - axis] = nullptr;
      }
    } else {
      if (failed(result)) {
        auto unstructuredMasks = getUnstructuredMasks();
        if (unstructuredMasks.empty()) {
          return failure();
        }
        if (unstructuredMasks.size() > 1) {
          return failure();
        }
        auto [dim, mask] = unstructuredMasks.front();
        // Recover dims for unstructured mask dim to allow other dim to be
        // processed.
        dims[dim] = builder.getIndexAttr(srcType.getShape()[dim]);
        if (hasLows())
          lows[dim] = builder.getIndexAttr(0);
      }
      masks.insert(masks.begin() + axis, nullptr);
    }
  }

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");
  this->dims.insert(this->dims.begin() + axis, builder.getIndexAttr(1));
  if (hasLows())
    this->lows.insert(this->lows.begin() + axis, builder.getIndexAttr(0));
  if (!digits.empty())
    digits.insert(digits.begin() + axis, SmallVector<Digit>{});

  return success();
}

// Return all non-nullptr masks along with their dimensions.
SmallVector<std::pair<unsigned, Value>> MaskState::getUnstructuredMasks() {
  SmallVector<std::pair<unsigned, Value>> result;

  for (auto [i, m] : llvm::enumerate(masks)) {
    if (m) {
      result.push_back({i, m});
    }
  }

  return result;
}

} // namespace triton
} // namespace mlir
