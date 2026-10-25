// The plugin's entry point: its dialects, its ops `torchsim_vcix` and `torchsim_layout_constraint`,
// and its passes by name -- one each for the TTGIR passes, and `pipeline`, which runs a textual pass
// pipeline over every pass registered here, so a caller states the linalg lowering as an opt tool's
// pass list.
#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/PyTorchSimOps.h"
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
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
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

// The i32 constants a tl.extra.torchsim builtin passes after its tensors, read in order.
struct Ints {
  std::vector<mlir::Value> &operands;
  size_t at;
  const char *who;
  llvm::SmallVector<unsigned> take(size_t count) {
    llvm::SmallVector<unsigned> out;
    for (size_t i = at; i < at + count; ++i) {
      llvm::APInt v;
      if (i >= operands.size() || !mlir::matchPattern(operands[i], mlir::m_ConstantInt(&v)))
        llvm::report_fatal_error(llvm::Twine(who) + ": operand " + llvm::Twine(i) +
                                 " is not the i32 constant the builtin passes");
      out.push_back(v.getZExtValue());
    }
    at += count;
    return out;
  }
};

// A layout as tl.extra.torchsim spells it (_spelled): 0 and a blocked layout's rank, size_per_thread,
// threads_per_warp, warps_per_cta, order; or 1 and a linear layout's rank, its register, lane, warp
// and block basis counts, the bases and the shape. Whether the machine takes it is the anchor's to
// say; here only a spelling no layout has is refused. `rank` is set to the layout's.
mlir::Attribute readLayout(Ints &in, mlir::MLIRContext *ctx, unsigned *rankOut = nullptr) {
  unsigned kind = in.take(1)[0];
  unsigned rank = in.take(1)[0];
  if (rankOut)
    *rankOut = rank;
  if (kind == 0) {
    auto spt = in.take(rank), tpw = in.take(rank), wpc = in.take(rank), order = in.take(rank);
    return gpu::BlockedEncodingAttr::get(ctx, spt, tpw, wpc, order,
                                         gpu::CGAEncodingAttr::get1CTALayout(ctx, rank));
  }
  if (kind != 1)
    llvm::report_fatal_error(llvm::Twine(in.who) + ": layout kind " + llvm::Twine(kind) +
                             " is neither blocked (0) nor linear (1)");
  auto counts = in.take(4);
  LinearLayout::BasesT bases;
  const char *dims[] = {"register", "lane", "warp", "block"};
  for (auto [dim, count] : llvm::zip(dims, counts)) {
    std::vector<std::vector<int32_t>> each;
    for (unsigned i = 0; i < count; ++i) {
      auto b = in.take(rank);
      each.emplace_back(b.begin(), b.end());
    }
    bases[mlir::StringAttr::get(ctx, dim)] = std::move(each);
  }
  auto shape = in.take(rank);
  auto pow2 = [](unsigned n) { return n && llvm::isPowerOf2_32(n); };
  for (auto &[dim, each] : bases)
    for (auto &basis : each)
      for (auto [x, n] : llvm::zip(basis, shape))
        if (!pow2(n) || unsigned(x) >= n)
          llvm::report_fatal_error(llvm::Twine(in.who) + ": a linear layout's " + dim.str() +
                                   " basis past its shape, or a shape not a power of two");
  llvm::SmallVector<std::pair<mlir::StringAttr, int32_t>> outDims;
  for (auto [name, n] : llvm::zip(standardOutDimNames(ctx, rank), shape))
    outDims.push_back({name, int32_t(n)});
  return gpu::LinearEncodingAttr::get(
      ctx, LinearLayout(std::move(bases), outDims, /*requireSurjective=*/false));
}

// tl.extra.torchsim.vcix: [result slot, tensors..., then i32 constants opcode, imm, the form's
// length and characters, 0 or 1 result, a layout per tensor then the result's (readLayout), the
// result's shape, and a scalar of its element type] -> one pytorchsim.vcix.
void addVcix(TritonOpBuilder &b, std::vector<mlir::Value> &operands) {
  size_t n = 1;
  while (n < operands.size() && mlir::isa<mlir::RankedTensorType>(operands[n].getType()))
    ++n;
  Ints in{operands, n, "torchsim_vcix"};
  auto head = in.take(3);
  std::string form;
  for (unsigned c : in.take(head[2]))
    form.push_back(char(c));
  bool hasResult = in.take(1)[0];
  mlir::MLIRContext *ctx = b.getBuilder().getContext();
  ctx->getOrLoadDialect<gpu::TritonGPUDialect>();
  ctx->getOrLoadDialect<::pytorchsim::PyTorchSimDialect>();
  llvm::SmallVector<mlir::Attribute> layouts;
  for (size_t i = 1; i < n; ++i)
    layouts.push_back(readLayout(in, ctx));
  mlir::Attribute outLayout;
  llvm::SmallVector<mlir::Type> results;
  if (hasResult) {
    unsigned rank;
    outLayout = readLayout(in, ctx, &rank);
    auto dims = in.take(rank);
    if (in.at + 1 != operands.size())
      llvm::report_fatal_error("torchsim_vcix: no scalar of the result's element type at the end");
    mlir::Value carrier = operands[in.at];
    results.push_back(mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t>(dims.begin(), dims.end()), carrier.getType()));
    if (auto *def = carrier.getDefiningOp(); def && def->use_empty())
      def->erase();
  } else if (in.at != operands.size()) {
    llvm::report_fatal_error("torchsim_vcix: operands left over after the layouts");
  }
  llvm::SmallVector<mlir::Value> args(operands.begin() + 1, operands.begin() + n);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  auto signedImm = int64_t(int32_t(head[1]));
  auto op = b.create<::pytorchsim::VcixOp>(
      mlir::TypeRange(results), args, mlir::Value(), mlir::IntegerAttr::get(i64, head[0]),
      mlir::IntegerAttr::get(i64, signedImm), mlir::StringAttr::get(ctx, form),
      mlir::ArrayAttr::get(ctx, layouts), outLayout);
  operands[0] = results.empty() ? mlir::Value() : op->getResult(0);
}

// tl.extra.torchsim.constrain: [result slot, the tile, then its layout as i32 constants
// (readLayout)] -> one pytorchsim.layout_constraint.
void addLayoutConstraint(TritonOpBuilder &b, std::vector<mlir::Value> &operands) {
  mlir::MLIRContext *ctx = b.getBuilder().getContext();
  ctx->getOrLoadDialect<gpu::TritonGPUDialect>();
  ctx->getOrLoadDialect<::pytorchsim::PyTorchSimDialect>();
  Ints in{operands, 2, "torchsim_layout_constraint"};
  mlir::Attribute layout = readLayout(in, ctx);
  if (in.at != operands.size())
    llvm::report_fatal_error("torchsim_layout_constraint: operands left over after the layout");
  mlir::Value tile = operands[1];
  operands[0] = b.create<::pytorchsim::LayoutConstraintOp>(tile.getType(), tile, layout);
}

void registerDialects(mlir::DialectRegistry *registry) {
  registry->insert<::pytorchsim::PyTorchSimDialect, mlir::tptr::TPtrDialect, mlir::ptr::PtrDialect,
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
  static plugin::OpInfo ops[] = {{"torchsim_vcix", addVcix},
                                 {"torchsim_layout_constraint", addLayoutConstraint}};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION, "TorchsimTritonExt", "0.1.0",
                                    passes, 6, dialects, 1, ops, 2, TRITON_VERSION};
  return &info;
}
