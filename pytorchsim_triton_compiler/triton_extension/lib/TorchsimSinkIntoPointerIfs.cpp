// triton-torchsim-sink-into-pointer-ifs: the code after an scf.if that yields a pointer, moved
// into each of its branches. Inductor's shared-body foreach kernel picks a sub-kernel's pointers
// by program id (`if pid < n: arg = in_ptr0 elif ...`) and runs one body on the pick; this
// machine's DMA names its DRAM argument statically, so the body is run inside the branch that
// knows which argument it is. Exactly one branch runs, so the copies are not a second execution.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DECL_TRITONTORCHSIMSINKINTOPOINTERIFS
#define GEN_PASS_DEF_TRITONTORCHSIMSINKINTOPOINTERIFS
#include "TorchsimExt/Passes.h.inc"

namespace {

bool isPointerLike(Type t) {
  if (auto shaped = dyn_cast<ShapedType>(t))
    t = shaped.getElementType();
  return isa<PointerType>(t);
}

bool yieldsPointer(scf::IfOp ifOp) {
  return llvm::any_of(ifOp.getResultTypes(), isPointerLike);
}

// The ops between `ifOp` and its block's terminator, if every use of the if's results is among
// them and the terminator reads none of what they define; else nothing.
std::optional<SmallVector<Operation *>> tailOf(scf::IfOp ifOp) {
  Block *block = ifOp->getBlock();
  SmallVector<Operation *> tail;
  for (Operation *op = ifOp->getNextNode(); op && op != block->getTerminator();
       op = op->getNextNode())
    tail.push_back(op);
  llvm::SmallPtrSet<Operation *, 16> inTail(tail.begin(), tail.end());
  for (Value r : ifOp.getResults())
    for (Operation *user : r.getUsers())
      if (!inTail.count(block->findAncestorOpInBlock(*user)))
        return std::nullopt;
  for (Value v : block->getTerminator()->getOperands()) {
    Operation *def = v.getDefiningOp();
    if (def == ifOp.getOperation() || (def && inTail.count(block->findAncestorOpInBlock(*def))))
      return std::nullopt;
  }
  return tail;
}

LogicalResult sink(scf::IfOp ifOp) {
  if (!ifOp.getElseRegion().hasOneBlock())
    return failure();
  auto tail = tailOf(ifOp);
  if (!tail)
    return failure();
  OpBuilder b(ifOp);
  for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    auto yield = cast<scf::YieldOp>(region->front().getTerminator());
    IRMapping map;
    for (auto [res, val] : llvm::zip(ifOp.getResults(), yield.getOperands()))
      map.map(res, val);
    b.setInsertionPoint(yield);
    for (Operation *op : *tail)
      b.clone(*op, map);
    scf::YieldOp::create(b, yield.getLoc());
    yield.erase();
  }
  for (Operation *op : llvm::reverse(*tail))
    op->erase();
  b.setInsertionPoint(ifOp);
  auto bare = scf::IfOp::create(b, ifOp.getLoc(), TypeRange{}, ifOp.getCondition(),
                                /*withElseRegion=*/true);
  bare.getThenRegion().takeBody(ifOp.getThenRegion());
  bare.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp.erase();
  return success();
}

// The canonicalizer folds an if whose branches only pick values into arith.selects; a pointer
// picked that way goes back into one scf.if yielding every select on the same condition, so
// sink() has a branch to move the body into. One if per condition, or the copies would multiply.
LogicalResult liftSelects(Block &block) {
  arith::SelectOp first;
  for (Operation &op : block)
    if (auto sel = dyn_cast<arith::SelectOp>(&op))
      if (isPointerLike(sel.getType()) && !isa<ShapedType>(sel.getCondition().getType())) {
        first = sel;
        break;
      }
  if (!first)
    return failure();
  Value cond = first.getCondition();
  SmallVector<arith::SelectOp> group;
  for (Operation &op : block)
    if (auto sel = dyn_cast<arith::SelectOp>(&op))
      if (sel.getCondition() == cond && !isa<ShapedType>(sel.getType()))
        group.push_back(sel);
  Operation *last = group.back();
  for (arith::SelectOp sel : group)
    for (Operation *user : sel->getUsers()) {
      Operation *anc = block.findAncestorOpInBlock(*user);
      if (!anc || !last->isBeforeInBlock(anc))
        return failure();
    }
  OpBuilder b(last->getContext());
  b.setInsertionPointAfter(last);
  SmallVector<Type> types;
  SmallVector<Value> yes, no;
  for (arith::SelectOp sel : group) {
    types.push_back(sel.getType());
    yes.push_back(sel.getTrueValue());
    no.push_back(sel.getFalseValue());
  }
  auto ifOp = scf::IfOp::create(b, last->getLoc(), types, cond, /*withElseRegion=*/true);
  OpBuilder tb = OpBuilder::atBlockEnd(ifOp.thenBlock());
  scf::YieldOp::create(tb, last->getLoc(), yes);
  OpBuilder eb = OpBuilder::atBlockEnd(ifOp.elseBlock());
  scf::YieldOp::create(eb, last->getLoc(), no);
  for (auto [sel, res] : llvm::zip(group, ifOp.getResults())) {
    sel.getResult().replaceAllUsesWith(res);
    sel.erase();
  }
  return success();
}

struct SinkIntoPointerIfs : impl::TritonTorchsimSinkIntoPointerIfsBase<SinkIntoPointerIfs> {
  void runOnOperation() override {
    bool lifted = true;
    while (lifted) {
      lifted = false;
      getOperation().walk([&](Block *block) {
        if (!lifted && succeeded(liftSelects(*block)))
          lifted = true;
      });
    }
    // Outermost first: an inner if's tail is the enclosing branch's yield until the enclosing
    // if has been sunk, so the walk restarts after every change.
    bool changed = true;
    while (changed) {
      changed = false;
      getOperation().walk<WalkOrder::PreOrder>([&](scf::IfOp ifOp) {
        if (ifOp.getNumResults() == 0 || !yieldsPointer(ifOp) || failed(sink(ifOp)))
          return WalkResult::advance();
        changed = true;
        return WalkResult::interrupt();
      });
    }
  }
};

}  // namespace

}  // namespace gpu
}  // namespace triton
}  // namespace mlir
