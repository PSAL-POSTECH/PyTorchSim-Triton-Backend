//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace triton {
bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

unsigned storageBytes(Type t) {
  if (auto shaped = dyn_cast<ShapedType>(t))
    t = shaped.getElementType();
  if (isa<triton::PointerType>(t))
    return 8;
  return llvm::divideCeil(std::max(8u, t.getIntOrFloatBitWidth()), 8);
}

// THE WALK'S TWO ENDS, NOT ITS LENGTH. A negative stride runs the axis
// backwards -- a lattice this ladder does spell -- so the span is min..max of
// the two ends, and a negative start is refused because `remsi` keeps the sign.
bool wrapNeverFires(OpFoldResult wrap, OpFoldResult offset,
                    OpFoldResult stride, int64_t size) {
  auto w = getConstantIntValue(wrap);
  auto off = getConstantIntValue(offset);
  auto st = getConstantIntValue(stride);
  if (!w || !off || !st || *w <= 0 || size <= 0)
    return false;

  int64_t end = *off + (size - 1) * *st;
  int64_t lo = std::min(*off, end), hi = std::max(*off, end);
  return lo >= 0 && hi < *w;
}

unsigned ptrCastElementRatio(BitcastOp op) {
  auto pointee = [](Type t) -> Type {
    if (auto shaped = dyn_cast<ShapedType>(t))
      t = shaped.getElementType();
    auto ptr = dyn_cast<triton::PointerType>(t);
    return ptr ? ptr.getPointeeType() : Type();
  };
  Type src = pointee(op.getSrc().getType());
  Type dst = pointee(op.getResult().getType());
  if (!src || !dst)
    return 0;
  unsigned s = storageBytes(src), d = storageBytes(dst);
  // ONE TEST ANSWERS BOTH REFUSALS. A cast that WIDENS has s < d, so `s % d` is
  // s and never 0 -- and it must be refused, because going the other way is a
  // division and nothing here proves the offset is a multiple.
  if (s % d != 0)
    return 0;
  return s / d;
}

std::optional<int64_t> constantIntOrSplat(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = getConstantIntValue(v))
    return c;
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!attr || !attr.isSplat() || !isa<IntegerType>(attr.getElementType()))
    return std::nullopt;
  return attr.getSplatValue<APInt>().getSExtValue();
}
} // namespace triton

} // namespace mlir
