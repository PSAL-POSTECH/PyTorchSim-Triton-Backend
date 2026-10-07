// The plugin's entry point: its dialects, and its passes by name -- one each for the TTGIR passes,
// and `pipeline`, which runs a textual pass pipeline over every pass registered here, so a caller
// states triton_shared's stage as the same pass list an opt tool would read.
#include "TorchSim/TorchSimDialect.h"
#include "TorchsimExt/TorchsimPasses.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Tools/PluginUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::triton;

namespace {

struct WriteGeneric : public mlir::PassWrapper<WriteGeneric, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WriteGeneric)
  explicit WriteGeneric(std::string path) : path(std::move(path)) {}
  void runOnOperation() override {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      getOperation().emitError("torchsim write_generic: ") << ec.message();
      return signalPassFailure();
    }
    getOperation()->print(os, mlir::OpPrintingFlags().printGenericOpForm().useLocalScope());
    os << "\n";
  }
  std::string path;
};

void addWriteGeneric(mlir::PassManager *pm, const std::vector<std::string> &args) {
  if (args.size() != 1)
    llvm::report_fatal_error("torchsim write_generic: takes one path");
  pm->addPass(std::make_unique<WriteGeneric>(args[0]));
}

void registerNothing() {}

template <typename F> void registerNamed(F create) {
  ::mlir::registerPass([create]() -> std::unique_ptr<::mlir::Pass> { return create(); });
}

void registerAll() {
  static bool done = false;
  if (done)
    return;
  done = true;
  registerNamed([] { return gpu::createTritonGPUTorchsimAnchorLayouts(); });
  registerNamed([] { return gpu::createTritonGPUTorchsimStripPins(); });
  registerNamed([] { return gpu::createTritonGPUTorchsimRemoveLayoutConversions(); });
  registerNamed([] { return gpu::createTritonGPUTorchsimRegularizeLayouts(); });
  registerNamed([] { return createTorchsimConvertTritonToTritonGPU(); });
  registerNamed([] { return createTritonToStructuredPass(); });
  registerNamed([] { return createTritonToUnstructuredPass(); });
  registerNamed([] { return createTritonArithToLinalgPass(); });
  registerNamed([] { return createTritonPtrToMemrefPass(); });
  registerNamed([] { return createTritonToPtrPass(); });
  registerNamed([] { return createReconcilePtrCastsPass(); });
  ::mlir::registerCSEPass();
  ::mlir::registerCanonicalizerPass();
  ::mlir::registerRemoveDeadValuesPass();
}

void addPipeline(mlir::PassManager *pm, const std::vector<std::string> &args) {
  registerAll();
  std::string text;
  llvm::raw_string_ostream err(text);
  if (args.size() != 1 || mlir::failed(mlir::parsePassPipeline(args[0], *pm, err)))
    llvm::report_fatal_error(llvm::Twine("torchsim pipeline: ") +
                             (args.size() == 1 ? text : "takes one pipeline string"));
}

template <std::unique_ptr<mlir::Pass> (*Create)()>
void addOne(mlir::PassManager *pm, const std::vector<std::string> &) {
  pm->addPass(Create());
}

void registerDialects(mlir::DialectRegistry *registry) {
  registry->insert<::torchsim::TorchSimDialect, mlir::tptr::TPtrDialect, mlir::ptr::PtrDialect,
                   mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect>();
}

}

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo passes[] = {
      {"anchor_layouts", "0.1.0", addOne<gpu::createTritonGPUTorchsimAnchorLayouts>, registerAll},
      {"strip_pins", "0.1.0", addOne<gpu::createTritonGPUTorchsimStripPins>, registerAll},
      {"remove_layout_conversions", "0.1.0",
       addOne<gpu::createTritonGPUTorchsimRemoveLayoutConversions>, registerAll},
      {"pipeline", "0.1.0", addPipeline, registerAll},
      {"write_generic", "0.1.0", addWriteGeneric, registerNothing},
  };
  static plugin::DialectInfo dialects[] = {{"torchsim", "0.1.0", registerDialects}};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION, "TorchsimTritonExt", "0.1.0",
                                    passes, 5, dialects, 1, nullptr, 0, TRITON_VERSION};
  return &info;
}
