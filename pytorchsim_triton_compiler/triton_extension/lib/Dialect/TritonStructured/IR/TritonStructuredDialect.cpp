//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "PyTorchSimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;
using namespace mlir::tts;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonStructuredDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PyTorchSimExt/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"
      >();
}

// TWO OPTIONAL GROUPS THAT BOTH OPEN WITH `,` ARE ONE GROUP TO THE PARSER: it
// commits on the comma and then demands the first keyword, so make_tptr printed
// a flat_shape without a base_offset and could not read it back. The keyword
// after the comma is what decides. THE PRINTED FORM IS UNCHANGED.
static ParseResult
parseTailSlots(OpAsmParser &parser,
               std::optional<OpAsmParser::UnresolvedOperand> &baseOffset,
               std::optional<OpAsmParser::UnresolvedOperand> &flatShape) {
  while (succeeded(parser.parseOptionalComma())) {
    llvm::SMLoc loc = parser.getCurrentLocation();
    std::optional<OpAsmParser::UnresolvedOperand> *slot = nullptr;
    if (succeeded(parser.parseOptionalKeyword("base_offset")))
      slot = &baseOffset;
    else if (succeeded(parser.parseOptionalKeyword("flat_shape")))
      slot = &flatShape;
    else
      return parser.emitError(loc, "expected 'base_offset' or 'flat_shape'");
    if (slot->has_value())
      return parser.emitError(loc, "that slot is already given");
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseColon() || parser.parseOperand(operand))
      return failure();
    *slot = operand;
  }
  return success();
}

static void printTailSlots(OpAsmPrinter &printer, Operation *, Value baseOffset,
                           Value flatShape) {
  if (baseOffset)
    printer << ", base_offset: " << baseOffset;
  if (flatShape)
    printer << ", flat_shape: " << flatShape;
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PyTorchSimExt/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"

#include "PyTorchSimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp.inc"
