//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES_H

#include "TorchsimExt/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "TorchsimExt/Conversion/TritonToLinalgExperimental/TritonToPtr.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "TorchsimExt/Conversion/TritonToLinalgExperimental/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
