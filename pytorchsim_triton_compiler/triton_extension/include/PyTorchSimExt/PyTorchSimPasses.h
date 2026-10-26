// The pytorchsim TTGIR passes' factories, for the plugin entry point (a separate TU).
#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir::triton::gpu {
std::unique_ptr<::mlir::Pass> createTritonGPUPyTorchSimAnchorLayouts();
std::unique_ptr<::mlir::Pass> createTritonGPUPyTorchSimStripPins();
std::unique_ptr<::mlir::Pass> createTritonGPUPyTorchSimRemoveLayoutConversions();
std::unique_ptr<::mlir::Pass> createTritonGPUPyTorchSimRegularizeLayouts();
}

namespace mlir::triton {
std::unique_ptr<::mlir::Pass> createPyTorchSimConvertTritonToTritonGPU();
}
