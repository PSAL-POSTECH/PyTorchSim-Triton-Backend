//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrTypes.cpp.inc"

using namespace mlir;

namespace {
ParseResult parseIntType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getIndexType();
  return success();
}
void printIntType(OpAsmPrinter &p, Operation *op, Type ty) {
  if (!ty.isIndex())
    p << " : " << ty;
}
} // namespace

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//
void mlir::tptr::TPtrDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrTypes.cpp.inc"
      >();
}

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void mlir::tptr::TPtrDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrAttributes.cpp.inc"
      >();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrOps.cpp.inc"
      >();
}





// THE FOLD'S OTHER HALF. `type_offset` folds to the element's size, and the
// folder needs this dialect to turn that attribute back into a value -- an
// arith.constant, because the size is an integer and nothing here is.
mlir::Operation *mlir::tptr::TPtrDialect::materializeConstant(OpBuilder &builder,
                                                              Attribute value,
                                                              Type type,
                                                              Location loc) {
  if (auto intAttr = dyn_cast<IntegerAttr>(value))
    if (isa<IntegerType, IndexType>(type))
      return arith::ConstantOp::create(builder, loc, type, intAttr);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrAttributes.cpp.inc"

#include "PyTorchSimExt/Dialect/TPtr/IR/TPtrDialect.cpp.inc"
