// The plugin's entry point: its dialects, its op `torchsim_vcix`, and its passes by name -- one each
// for the TTGIR passes, and `pipeline`, which runs a textual pass pipeline over every pass
// registered here, so a caller states triton_shared's stage as an opt tool's pass list.
#include "TorchSim/TorchSimDialect.h"
#include "TorchSim/TorchSimOps.h"
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
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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

// tl.extra.torchsim.vcix: [result slot, tensors..., then i32 constants opcode, imm, rank and rank
// each of size_per_thread, threads_per_warp, warps_per_cta, order] -> one torchsim.vcix whose
// result takes the first tensor's type, its layout the #ttg.blocked those numbers spell.
void addVcix(TritonOpBuilder &b, std::vector<mlir::Value> &operands) {
  size_t n = 1;
  while (n < operands.size() && mlir::isa<mlir::RankedTensorType>(operands[n].getType()))
    ++n;
  auto ints = [&](size_t from, size_t count) {
    llvm::SmallVector<unsigned> out;
    for (size_t i = from; i < from + count; ++i) {
      llvm::APInt v;
      if (i >= operands.size() || !mlir::matchPattern(operands[i], mlir::m_ConstantInt(&v)))
        llvm::report_fatal_error("torchsim_vcix: operand " + llvm::Twine(i) +
                                 " is not the i32 constant the builtin passes");
      out.push_back(v.getZExtValue());
    }
    return out;
  };
  if (n == 1)
    llvm::report_fatal_error("torchsim_vcix: names no tensor operand");
  auto head = ints(n, 3);
  unsigned rank = head[2];
  if (operands.size() != n + 3 + 4 * rank)
    llvm::report_fatal_error("torchsim_vcix: a rank-" + llvm::Twine(rank) + " layout takes " +
                             llvm::Twine(4 * rank) + " numbers after opcode, imm and rank");
  mlir::MLIRContext *ctx = b.getBuilder().getContext();
  ctx->getOrLoadDialect<gpu::TritonGPUDialect>();
  ctx->getOrLoadDialect<::torchsim::TorchSimDialect>();
  size_t at = n + 3;
  auto layout = gpu::BlockedEncodingAttr::get(
      ctx, ints(at, rank), ints(at + rank, rank), ints(at + 2 * rank, rank),
      ints(at + 3 * rank, rank), gpu::CGAEncodingAttr::get1CTALayout(ctx, rank));
  llvm::SmallVector<mlir::Value> args(operands.begin() + 1, operands.begin() + n);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  operands[0] = b.create<::torchsim::VcixOp>(
      args[0].getType(), args, mlir::IntegerAttr::get(i64, head[0]),
      mlir::IntegerAttr::get(i64, head[1]), layout);
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
  static plugin::OpInfo ops[] = {{"torchsim_vcix", addVcix}};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION, "TorchsimTritonExt", "0.1.0",
                                    passes, 5, dialects, 1, ops, 1, TRITON_VERSION};
  return &info;
}
