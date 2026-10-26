// What LoopLikeOpInterface and RegionBranchOpInterface say of one op -- its ivs, bounds, iter_args, and
// the operands that choose a region rather than flow into one -- asked by the passes in Python through
// _pytorchsim_ext, so that no pass tells a loop or a branch by its name.
#include "PyTorchSim-c/Dialects.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

/// The loop's ivs into `out` (at most `cap`): their count, or -1 when `op` is no loop that states them.
extern "C" int pytorchsimLoopIvs(MlirOperation c_op, MlirValue *out, int cap) {
  auto loop = dyn_cast<LoopLikeOpInterface>(unwrap(c_op));
  if (!loop)
    return -1;
  std::optional<SmallVector<Value>> ivs = loop.getLoopInductionVars();
  if (!ivs)
    return -1;
  for (int i = 0; i < (int)ivs->size() && i < cap; ++i)
    out[i] = wrap((*ivs)[i]);
  return ivs->size();
}

/// Bound `k` of kind `which` (0 lower, 1 upper, 2 step): 0 with `*value`, 1 with `*constant`, -1 none.
extern "C" int pytorchsimLoopBound(MlirOperation c_op, int which, int k, MlirValue *value, int64_t *constant) {
  auto loop = dyn_cast<LoopLikeOpInterface>(unwrap(c_op));
  if (!loop)
    return -1;
  std::optional<SmallVector<OpFoldResult>> got = which == 0   ? loop.getLoopLowerBounds()
                                                 : which == 1 ? loop.getLoopUpperBounds()
                                                              : loop.getLoopSteps();
  if (!got || k < 0 || k >= (int)got->size())
    return -1;
  OpFoldResult b = (*got)[k];
  if (auto v = dyn_cast<Value>(b)) {
    *value = wrap(v);
    return 0;
  }
  auto attr = dyn_cast<IntegerAttr>(cast<Attribute>(b));
  if (!attr)
    return -1;
  *constant = attr.getInt();
  return 1;
}

/// The loop's iter_args, their inits and the values yielded into them (null when it states none), at
/// most `cap`: their count, or -1 when `op` is no loop.
extern "C" int pytorchsimLoopIterArgs(MlirOperation c_op, MlirValue *args, MlirValue *inits, MlirValue *yielded,
                                    int cap) {
  auto loop = dyn_cast<LoopLikeOpInterface>(unwrap(c_op));
  if (!loop)
    return -1;
  Block::BlockArgListType iter = loop.getRegionIterArgs();
  MutableArrayRef<OpOperand> in = loop.getInitsMutable();
  std::optional<MutableArrayRef<OpOperand>> ys = loop.getYieldedValuesMutable();
  for (int i = 0; i < (int)iter.size() && i < cap; ++i) {
    args[i] = wrap(Value(iter[i]));
    inits[i] = wrap(in[i].get());
    yielded[i] = ys ? wrap((*ys)[i].get()) : MlirValue{nullptr};
  }
  return iter.size();
}

/// Whether `op` is a RegionBranchOpInterface op that is no loop: a branch.
extern "C" bool pytorchsimIsBranch(MlirOperation c_op) {
  Operation *op = unwrap(c_op);
  return isa<RegionBranchOpInterface>(op) && !isa<LoopLikeOpInterface>(op);
}

/// The operands of a RegionBranchOpInterface op that flow into none of its entry regions -- what picks
/// the region -- into `out` (at most `cap`): their count, or -1 when `op` is no such op. An entry that
/// forwards nothing is an empty range at the operands' end, compared by address, never dereferenced.
extern "C" int pytorchsimBranchControls(MlirOperation c_op, MlirValue *out, int cap) {
  auto br = dyn_cast<RegionBranchOpInterface>(unwrap(c_op));
  if (!br)
    return -1;
  SmallVector<Attribute> none(br->getNumOperands(), Attribute());
  SmallVector<RegionSuccessor> entries;
  br.getEntrySuccessorRegions(none, entries);
  llvm::SmallPtrSet<OpOperand *, 8> forwarded;
  for (RegionSuccessor &s : entries) {
    OperandRange fwd = br.getEntrySuccessorOperands(s);
    const OpOperand *base = fwd.getBase();
    for (OpOperand &o : br->getOpOperands())
      if (&o >= base && &o < base + fwd.size())
        forwarded.insert(&o);
  }
  int n = 0;
  for (OpOperand &o : br->getOpOperands())
    if (!forwarded.contains(&o)) {
      if (n < cap)
        out[n] = wrap(o.get());
      ++n;
    }
  return n;
}

/// Whether `op` is a LoopLikeOpInterface loop, whether or not it states ivs.
extern "C" bool pytorchsimIsLoop(MlirOperation c_op) { return isa<LoopLikeOpInterface>(unwrap(c_op)); }

/// The values that flow into block argument `c_arg` of a region entry block of a RegionBranchOpInterface
/// op -- from the op's entry and from every terminator of its regions that branches there -- into `out`
/// (at most `cap`): their count, or -1 when `c_arg` is no such argument.
extern "C" int pytorchsimRegionInputs(MlirValue c_arg, MlirValue *out, int cap) {
  auto arg = dyn_cast<BlockArgument>(unwrap(c_arg));
  if (!arg)
    return -1;
  Region *region = arg.getOwner()->getParent();
  auto br = dyn_cast_or_null<RegionBranchOpInterface>(region->getParentOp());
  if (!br || arg.getOwner() != &region->front())
    return -1;
  SmallVector<Value> got;
  br.getPredecessorValues(RegionSuccessor(region), arg.getArgNumber(), got);
  for (int i = 0; i < (int)got.size() && i < cap; ++i)
    out[i] = wrap(got[i]);
  return got.size();
}

/// The operands of a RegionBranchTerminatorOpInterface op forwarded to none of its successors -- what
/// decides where it branches, a while's condition -- into `out` (at most `cap`): their count, or -1.
extern "C" int pytorchsimTerminatorControls(MlirOperation c_op, MlirValue *out, int cap) {
  auto t = dyn_cast<RegionBranchTerminatorOpInterface>(unwrap(c_op));
  if (!t)
    return -1;
  SmallVector<Attribute> none(t->getNumOperands(), Attribute());
  SmallVector<RegionSuccessor> succ;
  t.getSuccessorRegions(none, succ);
  llvm::SmallPtrSet<OpOperand *, 8> forwarded;
  for (RegionSuccessor &s : succ) {
    MutableOperandRange fwd = t.getMutableSuccessorOperands(s);
    for (OpOperand &o : fwd)
      forwarded.insert(&o);
  }
  int n = 0;
  for (OpOperand &o : t->getOpOperands())
    if (!forwarded.contains(&o)) {
      if (n < cap)
        out[n] = wrap(o.get());
      ++n;
    }
  return n;
}
