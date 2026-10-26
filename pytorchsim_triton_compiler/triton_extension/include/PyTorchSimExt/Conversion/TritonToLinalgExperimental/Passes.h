//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES_H

#include "PyTorchSimExt/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "PyTorchSimExt/Conversion/TritonToLinalgExperimental/TritonToPtr.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "PyTorchSimExt/Conversion/TritonToLinalgExperimental/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
