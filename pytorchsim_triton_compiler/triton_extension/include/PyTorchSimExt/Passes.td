// The pytorchsim TTGIR passes as a Triton plugin: the same two definitions the fork kept in
// TritonGPU's Passes.td, generated here into the plugin's own Passes.h.inc.
#ifndef PYTORCHSIM_PLUGIN_PASSES
#define PYTORCHSIM_PLUGIN_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPyTorchSimAnchorLayouts : Pass<"tritongpu-pytorchsim-anchor-layouts", "mlir::ModuleOp"> {
  let summary = "Every layout the pytorchsim lane machine fixes, written in before layout propagation";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "::pytorchsim::PyTorchSimDialect",
                           "mlir::tensor::TensorDialect", "mlir::linalg::LinalgDialect"];
  let options = [
    Option<"groupBits", "group-bits", "unsigned", "0",
           "bits of the widest vector register group (VLEN x the largest LMUL); 0 leaves reduce orders">,
    Option<"targetDialect", "target", "std::string", "\"\"",
           "the target's dialect, whose cross-lane folds (dialect/src/targets.cmake FOLDS) say which "
           "reduces stay on the lanes">,
  ];
}

def TritonGPUPyTorchSimStripPins : Pass<"tritongpu-pytorchsim-strip-pins", "mlir::ModuleOp"> {
  let summary = "Every layout_constraint pin, taken out once the layouts are propagated";
}

def TritonGPUPyTorchSimRemoveLayoutConversions : Pass<"tritongpu-pytorchsim-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "Upstream's remove-layout-conversions, for a target whose accesses take any layout";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def TritonGPUPyTorchSimRegularizeLayouts : Pass<"tritongpu-pytorchsim-regularize-layouts", "mlir::ModuleOp"> {
  let summary = "Below TritonGPU: every encoding as its #ttg.linear, unit extents folded";
  let dependentDialects = ["mlir::tensor::TensorDialect", "mlir::linalg::LinalgDialect"];
}

def PyTorchSimConvertTritonToTritonGPU : Pass<"pytorchsim-convert-triton-to-tritongpu", "mlir::ModuleOp"> {
  let summary = "convert-triton-to-tritongpu with the pytorchsim lane machine's default layout";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::math::MathDialect",
                           "mlir::scf::SCFDialect", "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"target", "target", "std::string", "\"\"", "the target, e.g. pytorchsim:0">,
    Option<"numWarps", "num-warps", "int32_t", "4", "number of warps">,
    Option<"threadsPerWarp", "threads-per-warp", "int32_t", "32", "number of threads per warp">,
    Option<"numCTAs", "num-ctas", "int32_t", "1", "number of ctas in a cga">,
    Option<"enableSourceRemat", "enable-source-remat", "bool", "false",
           "enable trivial source rematerialization">,
  ];
}

#endif
