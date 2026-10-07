// The torchsim TTGIR passes' factories, for the plugin entry point (a separate TU).
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir::triton::gpu {
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimAnchorLayouts();
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimStripPins();
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimRemoveLayoutConversions();
std::unique_ptr<::mlir::Pass> createTritonGPUTorchsimRegularizeLayouts();
}

namespace mlir::triton {
std::unique_ptr<::mlir::Pass> createTorchsimConvertTritonToTritonGPU();
}
