//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "PyTorchSimExt/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <utility>

#define GET_OP_CLASSES
#include "PyTorchSimExt/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"

using namespace mlir;
using namespace mlir::tts;

namespace mlir {
namespace tts {

namespace utils {
// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return arith::SIToFPOp::create(builder, loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return arith::TruncFOp::create(builder, loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            builder, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

} // namespace utils

void MakeTensorPtrOp::build(OpBuilder &b, OperationState &state, Value base,
                            ArrayRef<int64_t> sizes,
                            ArrayRef<OpFoldResult> strides,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> shape,
                            ArrayRef<int32_t> order, Value baseOffset,
                            Value flatShape, Attribute encoding) {
  SmallVector<int64_t> staticStrides, staticOffsets, staticShape;
  SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicShape;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();
  // non-block pointer: a tensor of pointers, in the layout of the one it replaces
  if (order.empty()) {
    resType = RankedTensorType::get(sizes, basePtr, encoding);
  }
  // block pointer
  else {
    resType = RankedTensorType::get(
        sizes, triton::PointerType::get(elemType, basePtr.getAddressSpace()));
  }

  // No step: this builder's callers describe an axis that walks, and a stepped
  // one is set on the op afterwards by the one place that knows a period.
  build(b, state, resType, base, sizes, dynamicStrides, dynamicOffsets,
        dynamicShape, b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticShape), /*static_step=*/nullptr, order,
        baseOffset, flatShape);
}

void MakeGatherScatterTensorPtrOp::build(OpBuilder &b, OperationState &state,
                                         Value base, Value gatherScatterOffset,
                                         int gatherScatterDim,
                                         ArrayRef<int64_t> sizes,
                                         ArrayRef<OpFoldResult> strides,
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> shape,
                                         Value baseOffset) {
  SmallVector<int64_t> staticStrides, staticOffsets, staticShape;
  SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicShape;
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    if (i != gatherScatterDim)
      dispatchIndexOpFoldResult(offset, dynamicOffsets, staticOffsets);
    else
      staticOffsets.push_back(0);
  }
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();

  resType = RankedTensorType::get(
      sizes, triton::PointerType::get(elemType, basePtr.getAddressSpace()));

  build(b, state, resType, base, gatherScatterOffset,
        b.getI32IntegerAttr(gatherScatterDim), b.getDenseI64ArrayAttr(sizes),
        dynamicStrides, dynamicOffsets, dynamicShape,
        b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticShape), Value(), baseOffset);
}

void MakeGatherScatterTensorPtrOp::build(
    OpBuilder &b, OperationState &state, Value base, Value gatherScatterOffset,
    Value gatherScatterMask, int gatherScatterDim, ArrayRef<int64_t> sizes,
    ArrayRef<OpFoldResult> strides, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> shape, Value baseOffset) {
  SmallVector<int64_t> staticStrides, staticOffsets, staticShape;
  SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicShape;
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    if (i != gatherScatterDim)
      dispatchIndexOpFoldResult(offset, dynamicOffsets, staticOffsets);
    else
      staticOffsets.push_back(0);
  }
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();

  if (gatherScatterOffset.getType().isIntOrIndex()) {
    assert(sizes.size() == 1 && sizes[0] == 1 &&
           "gatherScatterOffset should be a scalar for 1D gather/scatter");
    resType = triton::PointerType::get(elemType, basePtr.getAddressSpace());

  } else {
    resType = RankedTensorType::get(
        sizes, triton::PointerType::get(elemType, basePtr.getAddressSpace()));
  }

  build(b, state, resType, base, gatherScatterOffset,
        b.getI32IntegerAttr(gatherScatterDim), b.getDenseI64ArrayAttr(sizes),
        dynamicStrides, dynamicOffsets, dynamicShape,
        b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticShape), gatherScatterMask, baseOffset);
}

// EVERY AXIS THAT SAYS IT WRAPS HAS TO BE ABLE TO. A wrap whose walk never
// reaches it is a fact that is not one, and the slot exists to carry facts.
static LogicalResult verifyNoDeadWrap(Operation *op, ArrayRef<int64_t> sizes,
                                      ArrayRef<OpFoldResult> shape,
                                      ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> strides) {
  for (auto [i, w] : llvm::enumerate(shape)) {
    if (auto c = getConstantIntValue(w); c && *c == 0)
      continue;
    if (triton::wrapNeverFires(w, offsets[i], strides[i], sizes[i]))
      return op->emitError("shape[")
             << i << "] says the axis wraps, but its walk never reaches the "
                     "boundary -- a wrap that cannot fire is not a wrap";
  }
  return success();
}

// THE BOX, CHECKED WHERE IT IS STATED. `static_mask_dims` is one clamp per
// tile axis with ShapedType::kDynamic standing for "the number is an operand",
// so three things have to agree: its length, `static_mask_lows`'s length, and
// how many `mask_dims` operands arrived. Every consumer re-derived these.
static LogicalResult verifyMaskBox(Operation *op, ArrayRef<int64_t> dims,
                                   std::optional<ArrayRef<int64_t>> lows,
                                   size_t nOperands, size_t nLowOperands,
                                   int64_t rank) {
  if (dims.empty())
    return nOperands == 0 && nLowOperands == 0
               ? success()
               : op->emitError("mask operands but no static_mask_dims to "
                               "place them on");
  if (static_cast<int64_t>(dims.size()) != rank)
    return op->emitError("static_mask_dims has ")
           << dims.size() << " entries for a rank-" << rank << " tile";
  if (lows && lows->size() != dims.size())
    return op->emitError("static_mask_lows has ")
           << lows->size() << " entries for " << dims.size() << " extents";
  if (!lows && nLowOperands)
    return op->emitError("mask_lows operands but no static_mask_lows to "
                         "place them on");
  size_t wanted = llvm::count(dims, ShapedType::kDynamic);
  if (wanted != nOperands)
    return op->emitError() << nOperands << " mask_dims operands but "
                           << "static_mask_dims asks for " << wanted;
  size_t wantedLows = lows ? llvm::count(*lows, ShapedType::kDynamic) : 0;
  if (wantedLows != nLowOperands)
    return op->emitError() << nLowOperands << " mask_lows operands but "
                           << "static_mask_lows asks for " << wantedLows;
  return success();
}

LogicalResult LoadOp::verify() {
  return verifyMaskBox(getOperation(), getStaticMaskDims(),
                       getStaticMaskLows(), getMaskDims().size(),
                       getMaskLows().size(),
                       cast<ShapedType>(getResult().getType()).getRank());
}

LogicalResult StoreOp::verify() {
  return verifyMaskBox(getOperation(), getStaticMaskDims(),
                       getStaticMaskLows(), getMaskDims().size(),
                       getMaskLows().size(),
                       cast<ShapedType>(getValue().getType()).getRank());
}

LogicalResult MakeTensorPtrOp::verify() {
  // A BLOCK POINTER SPELLS SOMETHING ELSE IN THIS SLOT: `shape` is then the
  // underlying tensor's extent for the boundary check, not a modulus, and
  // `order` is what tells the two apart -- PtrState::dimHasModulo asserts on
  // the same condition.
  if (!getOrder().empty())
    return success();
  return verifyNoDeadWrap(getOperation(), getSizes(), getMixedShape(),
                          getMixedOffsets(), getMixedStrides());
}

LogicalResult MakeGatherScatterTensorPtrOp::verify() {
  // Verify that the gatherScatterDim is within the valid range.
  if (getGatherScatterDim() < 0 || getGatherScatterDim() >= getSizes().size()) {
    return emitError("gatherScatterDim is out of bounds");
  }

  // Verify that the sizes, strides, and offsets have compatible dimensions.
  if (getMixedSizes().size() != getMixedStrides().size() ||
      getMixedSizes().size() != getMixedOffsets().size() ||
      getMixedSizes().size() != getMixedShape().size()) {
    return emitError("sizes, strides, offsets and shape must have the same "
                     "number of dimensions");
  }

  // A wrap on the GATHERED axis is arithmetic inside the index tensor, so it
  // never reaches this slot; a non-zero here would be a second spelling of it.
  if (getStaticShape()[getGatherScatterDim()] != 0) {
    return emitError("the gathered axis must not wrap, but shape[")
           << getGatherScatterDim() << "] is "
           << getStaticShape()[getGatherScatterDim()];
  }

  // No `order` on this op, so `shape` has only the one meaning here.
  if (failed(verifyNoDeadWrap(getOperation(), getSizes(), getMixedShape(),
                              getMixedOffsets(), getMixedStrides())))
    return failure();

  // ONE CLAIM, WRITTEN ONCE, AND ONE SENTENCE PER FAULT. The offset and the
  // mask make the SAME shape of claim -- a rank-1 tensor of a stated element
  // kind -- and each was spelled out on its own. They had already drifted in
  // the way a copy drifts: the offset's two rank faults shared one sentence,
  // and the mask's two contradicted each other ("must be a 1D tensor" against
  // "must be a 1D tensor of boolean type"), so no message said which of the
  // three things the op got wrong.
  auto require1DTensor = [&](Type t, StringRef what,
                             RankedTensorType &out) -> LogicalResult {
    out = dyn_cast<RankedTensorType>(t);
    if (!out)
      return emitError(what) << " must be a tensor, but is " << t;
    if (out.getRank() != 1)
      return emitError(what) << " must be rank 1, but is rank "
                             << out.getRank();
    return success();
  };

  RankedTensorType offsetTy;
  if (failed(require1DTensor(getGatherScatterOffset().getType(),
                             "gatherScatterOffset", offsetTy)))
    return failure();
  if (!offsetTy.getElementType().isIntOrIndex())
    return emitError("gatherScatterOffset must hold int or index, but holds ")
           << offsetTy.getElementType();

  if (getGatherScatterMask()) {
    RankedTensorType maskTy;
    if (failed(require1DTensor(getGatherScatterMask().getType(),
                               "gatherScatterMask", maskTy)))
      return failure();
    if (!maskTy.getElementType().isInteger(1))
      return emitError("gatherScatterMask must hold i1, but holds ")
             << maskTy.getElementType();
    // ONE BIT PER ROW: the mask is sized to the offset because it says which
    // of the gathered rows are read, not which elements of them are.
    if (maskTy.getShape()[0] != offsetTy.getShape()[0])
      return emitError("gatherScatterMask must have the same size as "
                       "gatherScatterOffset, but is ")
             << maskTy.getShape()[0] << " against " << offsetTy.getShape()[0];

    // AND THE GATHERED DIM IS THEN EMPTY ON EVERY ACCESS, because a mask on
    // that axis is carried HERE and nowhere else. This was two arms of
    // fourteen identical lines with "load"/"store" swapped, so the rule had to
    // be changed twice and the two could come to disagree about it.
    auto gatheredDimIsEmpty = [&](auto access) -> LogicalResult {
      // THE NAME IS READ, NOT WRITTEN AGAIN. It stood twice -- once as the
      // TypeSwitch's type below, once as a string beside it -- and only the
      // string would go stale if a mnemonic moved.
      StringRef what = access->getName().getStringRef();
      if (!access.hasMask())
        return emitError(what) << " user of tts.make_gather_scatter_tptr with "
                                  "gather_scatter_mask must have mask provided";
      OpFoldResult maskedSize =
          access.getMixedMaskDims()[getGatherScatterDim()];
      auto intAttr =
          dyn_cast_if_present<IntegerAttr>(dyn_cast<Attribute>(maskedSize));
      if (!intAttr || intAttr.getInt() != 0)
        return emitError(what)
               << " user of tts.make_gather_scatter_tptr with "
                  "gather_scatter_mask must have mask size of 0 for "
                  "gather_scatter_dim";
      return success();
    };

    for (auto user : (*this)->getUsers()) {
      auto res =
          TypeSwitch<Operation *, LogicalResult>(user)
              .Case<LoadOp, StoreOp>(
                  [&](auto op) { return gatheredDimIsEmpty(op); })
              .Default([&](Operation *) {
                // THE SAME TWO NAMES AGAIN, so they come off the same types the
                // Case above names. A type reference cannot silently drift; a
                // string beside it can.
                return emitError("tts.make_gather_scatter_tptr can only be "
                                 "used in ")
                       << LoadOp::getOperationName() << " or "
                       << StoreOp::getOperationName() << " operations";
              });
      if (failed(res))
        return res;
    }
  }

  return success();
}

// A low of all zeros is left OFF the op, so an access with no lower bound is
// spelled exactly as it was before the slot existed -- and a low that is a
// VALUE splits into the static/dynamic pair the dims already use.
static DenseI64ArrayAttr splitLows(OpBuilder &b, ArrayRef<OpFoldResult> lows,
                                   SmallVector<Value> &dynamicLows) {
  SmallVector<int64_t> staticLows;
  dispatchIndexOpFoldResults(lows, dynamicLows, staticLows);
  if (dynamicLows.empty() &&
      llvm::none_of(staticLows, [](int64_t v) { return v != 0; }))
    return DenseI64ArrayAttr();
  return b.getDenseI64ArrayAttr(staticLows);
}

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   ArrayRef<OpFoldResult> dims, ArrayRef<OpFoldResult> lows,
                   Value other, Value elemMask, Attribute encoding) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims, dynamicLows;

  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);
  auto staticLows = splitLows(b, lows, dynamicLows);

  // non-block pointer type
  auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType());
  // block pointer type
  auto tensorPtrType = dyn_cast<triton::PointerType>(ptr.getType());

  // The loaded tile takes `encoding`, the layout of the load it replaces; without
  // one it sits where its addresses do (a tensor of pointers' own layout).
  Type resType;
  if (ptrTensorType) {
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    resType = RankedTensorType::get(
        ptrTensorType.getShape(), elemType,
        encoding ? encoding : ptrTensorType.getEncoding());

  } else if (tensorPtrType) {
    auto tensorType = cast<ShapedType>(tensorPtrType.getPointeeType());
    resType = RankedTensorType::get(tensorType.getShape(),
                                    tensorType.getElementType(), encoding);
  }
  build(b, state, resType, ptr, dynamicDims, dynamicLows,
        b.getDenseI64ArrayAttr(staticDims), staticLows, other, elemMask);
}

void StoreOp::build(OpBuilder &b, OperationState &state, Value ptr, Value value,
                    ArrayRef<OpFoldResult> dims, ArrayRef<OpFoldResult> lows,
                    Value elemMask) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims, dynamicLows;

  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);
  auto staticLows = splitLows(b, lows, dynamicLows);

  build(b, state, ptr, value, dynamicDims, dynamicLows,
        b.getDenseI64ArrayAttr(staticDims), staticLows, elemMask);
}

LogicalResult GetStructuredStateOp::verify() {
  auto expectedOffsetAndStrideTypes =
      getOffsetAndStrideTypes(getContext(), getInput().getType());

  if (!expectedOffsetAndStrideTypes.has_value()) {
    return failure();
  }

  auto [expectedOffsetTypes, expectedStrideTypes] =
      *expectedOffsetAndStrideTypes;

  return success(
      expectedOffsetTypes.size() == getOffsets().size() &&
      llvm::equal(expectedOffsetTypes, getOffsets().getTypes()) &&
      expectedStrideTypes.size() == getStrides().size() &&
      llvm::equal(expectedStrideTypes, getStrides().getTypes()) &&
      static_cast<size_t>(getBaseOffsetSegmentSize(getInput().getType())) ==
          getBaseOffsets().size());
}

void GetStructuredStateOp::build(OpBuilder &b, OperationState &state,
                                 Value val) {
  auto type = val.getType();

  // Builder cannot fail, so we default to empty offset and stride types.
  // The invalid op will be rejected by the verifier later.
  auto [offsetTypes, strideTypes] =
      getOffsetAndStrideTypes(b.getContext(), type)
          .value_or(std::make_pair(SmallVector<Type>{}, SmallVector<Type>{}));
  SmallVector<Type> baseOffsetTypes(getBaseOffsetSegmentSize(type),
                                    IndexType::get(b.getContext()));

  build(b, state, val.getType(), offsetTypes, strideTypes, baseOffsetTypes,
        val);
}

std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
GetStructuredStateOp::getOffsetAndStrideTypes(MLIRContext *context, Type type) {
  auto sizes = getOffsetAndStrideSegmentSizes(type);
  if (!sizes.has_value()) {
    return std::nullopt;
  }
  return std::make_pair(
      SmallVector<Type>(sizes->first, IndexType::get(context)),
      SmallVector<Type>(sizes->second, IndexType::get(context)));
}

// A BASE OFFSET ONLY MATTERS WHERE A STRIDE DOES: a scalar pointer's whole
// state is its offset, and a wrap needs an axis to sit on. So this follows the
// stride count -- one where there are strides, none where there are not.
int32_t GetStructuredStateOp::getBaseOffsetSegmentSize(Type type) {
  auto sizes = getOffsetAndStrideSegmentSizes(type);
  return (sizes.has_value() && sizes->second > 0) ? 1 : 0;
}

std::optional<std::pair<int32_t, int32_t>>
GetStructuredStateOp::getOffsetAndStrideSegmentSizes(Type type) {
  int32_t offsetSegmentSize = 0;
  int32_t strideSegmentSize = 0;

  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    if (tensorType.getElementType().isIntOrIndex()) {
      // Tensors of offsets
      // Important note:
      // We only care about tensor of index / int (in addition to pointer type)
      // because only values of int and index type can potentially be part of a
      // pointer arithmetic sequence.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else if (auto ptrType =
                   dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      // Unstructured pointers (tensor<!tt.ptr<type>>)
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    }
  }
  // Block pointers (!tt.ptr<tensor<type>> or !tt.ptr<type>)
  else if (auto ptrType = llvm::dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            llvm::dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else {
      // The only relevant state that can be updated in loops for scalar
      // pointers are offset. No need to include stride here.
      offsetSegmentSize = 1;
    }
  } else {
    return std::nullopt;
  }

  return std::make_pair(offsetSegmentSize, strideSegmentSize);
}

} // namespace tts
} // namespace mlir
