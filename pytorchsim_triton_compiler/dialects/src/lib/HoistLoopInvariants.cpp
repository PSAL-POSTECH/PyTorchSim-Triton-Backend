// pytorchsim-hoist-loop-invariants: what an affine loop computes the same on every trip is
// computed once before it. A read leaves only when nothing in the loop may write what it reads, as
// mlir::AliasAnalysis answers; nothing that writes ever leaves.
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

using namespace mlir;

namespace {

/// The memref.get_global `v` is a view of, through every ViewLikeOpInterface; null if none.
memref::GetGlobalOp globalUnder(Value v) {
  while (auto view = v.getDefiningOp<ViewLikeOpInterface>()) {
    if (view.getViewDest() != v)
      break;
    v = view.getViewSource();
  }
  return v.getDefiningOp<memref::GetGlobalOp>();
}

/// Two memref.globals are two buffers. LocalAliasAnalysis tells only allocations apart, and every
/// scratchpad buffer is a global once bufferize has placed it.
struct GlobalAlias {
  AliasResult alias(Value a, Value b) {
    memref::GetGlobalOp ga = globalUnder(a), gb = globalUnder(b);
    if (ga && gb && ga.getNameAttr() != gb.getNameAttr())
      return AliasResult::NoAlias;
    return AliasResult::MayAlias;
  }
  ModRefResult getModRef(Operation *, Value) { return ModRefResult::getModAndRef(); }
};

/// What the loop may write: the buffers its writes name, and whether something writes what no
/// value names -- an op of unknown effects, or a write to all of an addressable resource.
struct Writes {
  SmallVector<Value> buffers;
  bool unknown = false;
};

Writes writesIn(affine::AffineForOp loop) {
  Writes w;
  loop.getBody()->walk([&](Operation *op) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface) {
      w.unknown |= !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
      return;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (MemoryEffects::EffectInstance &e : effects) {
      if (!isa<MemoryEffects::Write, MemoryEffects::Free>(e.getEffect()))
        continue;
      if (Value v = e.getValue())
        w.buffers.push_back(v);
      else
        w.unknown |= e.getResource()->isAddressable();
    }
  });
  return w;
}

/// The buffers `op` reads, when reading named buffers is all it does.
std::optional<SmallVector<Value>> onlyReads(Operation *op) {
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return std::nullopt;
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  SmallVector<Value> read;
  for (MemoryEffects::EffectInstance &e : effects) {
    if (!isa<MemoryEffects::Read>(e.getEffect()) || !e.getValue())
      return std::nullopt;
    read.push_back(e.getValue());
  }
  return read;
}

/// Whether `op` may run once before the loop instead of on every trip: an op without effects that
/// cannot fault, or, when the loop surely runs, one that faults or reads what the loop never writes.
bool mayLeave(Operation *op, const Writes &w, AliasAnalysis &aa, bool runs) {
  if (op->getNumRegions())
    return false;
  if (isMemoryEffectFree(op))
    return runs || isSpeculatable(op);
  if (!runs || w.unknown)
    return false;
  std::optional<SmallVector<Value>> read = onlyReads(op);
  return read && llvm::all_of(*read, [&](Value r) {
           return llvm::all_of(w.buffers, [&](Value b) { return aa.alias(r, b).isNo(); });
         });
}

struct HoistLoopInvariants
    : public PassWrapper<HoistLoopInvariants, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistLoopInvariants)
  StringRef getArgument() const final { return "pytorchsim-hoist-loop-invariants"; }
  StringRef getDescription() const final {
    return "Compute once before an affine loop what it computes the same on every trip";
  }

  /// Innermost loop first, so what leaves an inner loop is asked again of the one around it.
  void runOnOperation() override {
    AliasAnalysis aa(getOperation());
    aa.addAnalysisImplementation(GlobalAlias());
    SmallVector<affine::AffineForOp> loops;
    getOperation().walk([&](affine::AffineForOp loop) { loops.push_back(loop); });
    for (affine::AffineForOp loop : loops) {
      Writes w = writesIn(loop);
      std::optional<uint64_t> trips = affine::getConstantTripCount(loop);
      bool runs = trips && *trips > 0;
      moveLoopInvariantCode(
          {&loop.getRegion()},
          [&](Value v, Region *) { return loop.isDefinedOutsideOfLoop(v); },
          [&](Operation *op, Region *) { return mayLeave(op, w, aa, runs); },
          [&](Operation *op, Region *) { op->moveBefore(loop); });
    }
  }
};

} // namespace

namespace torchsim {
void registerHoistLoopInvariantsPass() {
  mlir::registerPass([] { return std::make_unique<HoistLoopInvariants>(); });
}
} // namespace torchsim
