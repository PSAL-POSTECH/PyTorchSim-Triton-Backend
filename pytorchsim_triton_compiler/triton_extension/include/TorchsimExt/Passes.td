// The torchsim TTGIR passes as a Triton plugin: the same two definitions the fork kept in
// TritonGPU's Passes.td, generated here into the plugin's own Passes.h.inc.
#ifndef TORCHSIM_PLUGIN_PASSES
#define TORCHSIM_PLUGIN_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUTorchsimAnchorLayouts : Pass<"tritongpu-torchsim-anchor-layouts", "mlir::ModuleOp"> {
  let summary = "Every layout the torchsim lane machine fixes, written in before layout propagation";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "::pytorchsim::PyTorchSimDialect",
                           "mlir::tensor::TensorDialect", "mlir::linalg::LinalgDialect"];
  let options = [
    Option<"groupBits", "group-bits", "unsigned", "0",
           "bits of the widest vector register group (VLEN x the largest LMUL); 0 leaves reduce orders">,
  ];
}

def TritonGPUTorchsimStripPins : Pass<"tritongpu-torchsim-strip-pins", "mlir::ModuleOp"> {
  let summary = "The identity-gather pins torchsim-anchor-layouts wrote, taken out after propagation";
}

def TritonGPUTorchsimRemoveLayoutConversions : Pass<"tritongpu-torchsim-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "Upstream's remove-layout-conversions, for a target whose accesses take any layout";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def TritonGPUTorchsimRegularizeLayouts : Pass<"tritongpu-torchsim-regularize-layouts", "mlir::ModuleOp"> {
  let summary = "Below TritonGPU: every encoding as its #ttg.linear, unit extents folded";
  let dependentDialects = ["mlir::tensor::TensorDialect", "mlir::linalg::LinalgDialect"];
}

def TorchsimConvertTritonToTritonGPU : Pass<"torchsim-convert-triton-to-tritongpu", "mlir::ModuleOp"> {
  let summary = "convert-triton-to-tritongpu with the torchsim lane machine's default layout";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::math::MathDialect",
                           "mlir::scf::SCFDialect", "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"target", "target", "std::string", "\"\"", "the target, e.g. torchsim:0">,
    Option<"numWarps", "num-warps", "int32_t", "4", "number of warps">,
    Option<"threadsPerWarp", "threads-per-warp", "int32_t", "32", "number of threads per warp">,
    Option<"numCTAs", "num-ctas", "int32_t", "1", "number of ctas in a cga">,
    Option<"enableSourceRemat", "enable-source-remat", "bool", "false",
           "enable trivial source rematerialization">,
  ];
}

#endif
