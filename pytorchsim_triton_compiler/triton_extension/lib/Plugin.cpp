// The plugin's entry point: its dialects, its op `torchsim_vcix`, and its passes by name -- one each
// for the TTGIR passes, and `pipeline`, which runs a textual pass pipeline over every pass
// registered here, so a caller states the linalg lowering as an opt tool's pass list.
#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/PyTorchSimOps.h"
#include "PyTorchSim/PyTorchSimExternDialect.h"
#include "PyTorchSim/PyTorchSimExternOps.h"
#include "TorchsimExt/TorchsimPasses.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "TorchsimExt/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "TorchsimExt/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "TorchsimExt/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "TorchsimExt/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "TorchsimExt/Conversion/TritonToStructured/TritonToStructured.h"
#include "TorchsimExt/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "TorchsimExt/Dialect/TPtr/IR/TPtrDialect.h"
#include "TorchsimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "TorchsimExt/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
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

// Erases every op named in `names`; one that still has a use is a failure, not a deletion.
struct DropOps : public mlir::PassWrapper<DropOps, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropOps)
  explicit DropOps(std::vector<std::string> names) : names(std::move(names)) {}
  void runOnOperation() override {
    llvm::SmallVector<mlir::Operation *> found;
    getOperation()->walk([&](mlir::Operation *op) {
      if (llvm::is_contained(names, op->getName().getStringRef()))
        found.push_back(op);
    });
    for (mlir::Operation *op : found) {
      if (!op->use_empty()) {
        op->emitError("torchsim drop_ops: a result of this op is still used");
        return signalPassFailure();
      }
      op->erase();
    }
  }
  std::vector<std::string> names;
};

void addDropOps(mlir::PassManager *pm, const std::vector<std::string> &args) {
  pm->addPass(std::make_unique<DropOps>(args));
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

// tl.extra.torchsim.vcix: [result slot, tensors..., then i32 constants opcode, imm, the form's
// length and characters, 0 or 1 result, a layout per tensor then the result's (each its rank and
// rank each of size_per_thread, threads_per_warp, warps_per_cta, order), the result's shape, and
// a scalar of its element type] -> one pytorchsim_extern.vcix.
void addVcix(TritonOpBuilder &b, std::vector<mlir::Value> &operands) {
  size_t n = 1;
  while (n < operands.size() && mlir::isa<mlir::RankedTensorType>(operands[n].getType()))
    ++n;
  size_t at = n;
  auto ints = [&](size_t count) {
    llvm::SmallVector<unsigned> out;
    for (size_t i = at; i < at + count; ++i) {
      llvm::APInt v;
      if (i >= operands.size() || !mlir::matchPattern(operands[i], mlir::m_ConstantInt(&v)))
        llvm::report_fatal_error("torchsim_vcix: operand " + llvm::Twine(i) +
                                 " is not the i32 constant the builtin passes");
      out.push_back(v.getZExtValue());
    }
    at += count;
    return out;
  };
  auto head = ints(3);
  std::string form;
  for (unsigned c : ints(head[2]))
    form.push_back(char(c));
  bool hasResult = ints(1)[0];
  mlir::MLIRContext *ctx = b.getBuilder().getContext();
  ctx->getOrLoadDialect<gpu::TritonGPUDialect>();
  ctx->getOrLoadDialect<::pytorchsim::external::PyTorchSimExternDialect>();
  auto layout = [&]() -> mlir::Attribute {
    unsigned rank = ints(1)[0];
    auto spt = ints(rank), tpw = ints(rank), wpc = ints(rank), order = ints(rank);
    return gpu::BlockedEncodingAttr::get(ctx, spt, tpw, wpc, order,
                                         gpu::CGAEncodingAttr::get1CTALayout(ctx, rank));
  };
  llvm::SmallVector<mlir::Attribute> layouts;
  for (size_t i = 1; i < n; ++i)
    layouts.push_back(layout());
  mlir::Attribute outLayout;
  llvm::SmallVector<mlir::Type> results;
  if (hasResult) {
    outLayout = layout();
    auto dims = ints(mlir::cast<gpu::BlockedEncodingAttr>(outLayout).getRank());
    if (at + 1 != operands.size())
      llvm::report_fatal_error("torchsim_vcix: no scalar of the result's element type at the end");
    mlir::Value carrier = operands[at];
    results.push_back(mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t>(dims.begin(), dims.end()), carrier.getType()));
    if (auto *def = carrier.getDefiningOp(); def && def->use_empty())
      def->erase();
  } else if (at != operands.size()) {
    llvm::report_fatal_error("torchsim_vcix: operands left over after the layouts");
  }
  llvm::SmallVector<mlir::Value> args(operands.begin() + 1, operands.begin() + n);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  auto signedImm = int64_t(int32_t(head[1]));
  auto op = b.create<::pytorchsim::external::VcixOp>(
      mlir::TypeRange(results), args, mlir::Value(), mlir::IntegerAttr::get(i64, head[0]),
      mlir::IntegerAttr::get(i64, signedImm), mlir::StringAttr::get(ctx, form),
      mlir::ArrayAttr::get(ctx, layouts), outLayout);
  operands[0] = results.empty() ? mlir::Value() : op->getResult(0);
}

void registerDialects(mlir::DialectRegistry *registry) {
  registry->insert<::pytorchsim::PyTorchSimDialect, ::pytorchsim::external::PyTorchSimExternDialect,
                   mlir::tptr::TPtrDialect, mlir::ptr::PtrDialect,
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
      {"drop_ops", "0.1.0", addDropOps, registerNothing},
  };
  static plugin::DialectInfo dialects[] = {{"pytorchsim", "0.1.0", registerDialects}};
  static plugin::OpInfo ops[] = {{"torchsim_vcix", addVcix}};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION, "TorchsimTritonExt", "0.1.0",
                                    passes, 6, dialects, 1, ops, 1, TRITON_VERSION};
  return &info;
}
