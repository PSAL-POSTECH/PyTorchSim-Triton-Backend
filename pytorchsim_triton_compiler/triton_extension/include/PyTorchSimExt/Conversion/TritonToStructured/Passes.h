//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_STRUCTURED_CONVERSION_PASSES_H
#define TRITON_TO_STRUCTURED_CONVERSION_PASSES_H

#include "PyTorchSimExt/Conversion/TritonToStructured/TritonToStructured.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "PyTorchSimExt/Conversion/TritonToStructured/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
