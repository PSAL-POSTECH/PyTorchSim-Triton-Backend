//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_STRUCTURED_CONVERSION_PASSES
#define TRITON_TO_STRUCTURED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToStructured : Pass<"triton-to-structured", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let constructor = "triton::createTritonToStructuredPass()";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", /*default*/"true",
             "Enable make_gather_scatter_tptr support">,
      Option<"splitSteppedAxes", "split-stepped-axes", "bool", /*default*/"true",
             "Split a tile axis carrying `x // K` into (x // K, x % K) instead of "
             "demoting the access. A pointer that crosses an scf boundary does "
             "not split -- tts.get_structured_state counts its results from the "
             "TYPE -- and declines to the same route this off says. Pass false "
             "for a backend that reads a tile by the TENSOR's rank.">,
      Option<"runPrepassOnly", "run-prepass-only", "bool", /*default*/"false",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">,
      Option<"skipPrepass", "skip-prepass", "bool", /*default*/"false",
             "Skip the prepass">,
      Option<"useUnsafeMask", "use-unsafe-mask", "bool", /*default*/"false",
             "Assume that the mask bounds are never less than starting offsets. May produce incorrect results.">,
      Option<"reportDemotions", "report-demotions", "bool", /*default*/"false",
             "Print, per arithmetic that would not fold into a stride, how many values PtrAnalysis rebuilt as gather/scatter pointers and how many it declined entirely">
  ];
}

#endif
