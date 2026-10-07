//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h" // Required for IR/TPtrOps.h.inc

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.h.inc"

using namespace mlir;
using namespace mlir::tptr;

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAddrMutable(),
                       SideEffects::DefaultResource::get());
}

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAddrMutable(),
                       SideEffects::DefaultResource::get());
}

// THE RESULT IS AN INTEGER, SO THE FOLD HAS TO BE ONE. This returned the
// baseType attribute -- `f32` where an i32 was asked for -- which no
// materializeConstant can turn back into a value, so the op never folded and
// every reader had to know an element's size for itself. The size comes from
// the DataLayout, which is where mlir::ptr::TypeOffsetOp asks for it too.
OpFoldResult TypeOffsetOp::fold(FoldAdaptor adaptor) {
  Type base = adaptor.getBaseTypeAttr().getValue();
  if (!base)
    return {};
  DataLayout layout = DataLayout::closest(getOperation());
  llvm::TypeSize size = layout.getTypeSize(base);
  if (size.isScalable())
    return {};
  Type resultTy = getResult().getType();
  if (!isa<IntegerType, IndexType>(resultTy))
    return {};
  return IntegerAttr::get(resultTy, static_cast<int64_t>(size));
}
