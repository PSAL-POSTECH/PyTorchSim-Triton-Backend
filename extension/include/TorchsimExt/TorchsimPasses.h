// The torchsim TTGIR passes' factories, for the plugin entry point (a separate TU: the
// plugin header pulls in pybind11, the passes are built without RTTI or exceptions).
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir::triton::gpu {
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimAnchorLayouts();
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimStripPins();
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimRemoveLayoutConversions();
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimRegularizeLayouts();
} // namespace mlir::triton::gpu

namespace mlir::triton {
std::unique_ptr<::mlir::Pass> createTorchsimConvertTritonToTritonGPU();
} // namespace mlir::triton
