// The TPU's answers to the anchor's layout questions: its MXU takes A with K on the lanes, B and the
// product with N, and turns a dot around where a reduce wants the rows on the lanes; its XLU folds
// one f32 max, min or sum across the lanes.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "TorchsimExt/LayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

namespace {

// What an elementwise chain joins `v` to, a tt.trans turning the chain's axes around on the way:
// the reduces over `v`'s last axis and over its first, each of which keeps the lanes off the axis
// it reduces (anchorReduce), and the dots whose operand or product the chain is.
struct Meets {
  bool rowReduce = false, colReduce = false;
  SetVector<Operation *> dots;
};

Meets meetsOf(Value v) {
  auto shape = cast<RankedTensorType>(v.getType()).getShape();
  int last = shape.size() - 1;
  auto chains = [](Operation *op) {
    auto trans = dyn_cast_or_null<TransOp>(op);
    return op && (op->hasTrait<OpTrait::Elementwise>() || isa<ConvertLayoutOp>(op) ||
                  (trans && trans.getType().getRank() == 2));
  };
  Meets met;
  SmallVector<std::pair<Value, bool>> todo{{v, false}};
  DenseSet<Value> seen{v};
  auto visit = [&](Operation *op, ValueRange xs, bool turned) {
    turned ^= isa<TransOp>(op);
    for (Value x : xs) {
      auto t = dyn_cast<RankedTensorType>(x.getType());
      SmallVector<int64_t> want(shape);
      if (turned)
        std::reverse(want.begin(), want.end());
      if (t && t.getShape() == ArrayRef<int64_t>(want) && seen.insert(x).second)
        todo.push_back({x, turned});
    }
  };
  while (!todo.empty()) {
    auto [x, turned] = todo.pop_back_val();
    for (Operation *user : x.getUsers()) {
      if (auto red = dyn_cast<ReduceOp>(user)) {
        int axis = turned ? last - red.getAxis() : red.getAxis();
        met.rowReduce |= axis == last;
        met.colReduce |= axis == 0;
      }
      if (isa<DotOp>(user))
        met.dots.insert(user);
      if (chains(user))
        visit(user, user->getResults(), turned);
    }
    Operation *def = x.getDefiningOp();
    if (isa_and_nonnull<DotOp>(def))
      met.dots.insert(def);
    if (chains(def))
      visit(def, def->getOperands(), turned);
  }
  return met;
}

// The dots turned around: each whose product or operand a row reduce meets, when its M rows fill
// as many lanes as its N columns would and turning breaks no other layout the dot's values meet --
// no reduce over a first axis (it wants that axis off the lanes turning puts there) and no dot
// left as it is (it keeps the other axis on the lanes). Then the turn only removes the reduce's
// crossing; where it would trade it for another, the dot stays as it is.
DenseSet<Operation *> turnedDots(ArrayRef<DotOp> dots, unsigned lanes) {
  auto filled = [&](int64_t n) { return std::min<int64_t>(n, lanes); };
  DenseMap<Operation *, SetVector<Operation *>> peers;
  DenseSet<Operation *> turned;
  for (DotOp dot : dots) {
    auto ct = cast<RankedTensorType>(dot.getC().getType());
    if (ct.getRank() != 2 || filled(ct.getShape()[0]) < filled(ct.getShape()[1]))
      continue;
    Meets all;
    for (Value v : {dot.getA(), dot.getB(), dot.getD()}) {
      Meets m = meetsOf(v);
      all.rowReduce |= m.rowReduce;
      all.colReduce |= m.colReduce;
      all.dots.insert(m.dots.begin(), m.dots.end());
    }
    all.dots.remove(dot);
    if (all.rowReduce && !all.colReduce) {
      turned.insert(dot);
      peers[dot] = std::move(all.dots);
    }
  }
  for (bool changed = true; changed;) {
    changed = false;
    for (auto &[dot, met] : peers)
      if (turned.contains(dot) &&
          llvm::any_of(met, [&](Operation *o) { return !turned.contains(o); })) {
        turned.erase(dot);
        changed = true;
      }
  }
  return turned;
}

class TpuLayoutRules : public LayoutRules {
public:
  // The array takes A with K on the lanes, B and the product with N: every operand's last axis.
  // A turned dot has every operand's first axis on the lanes, the rows the reduce keeps there.
  DenseMap<Operation *, DotForm> dotForms(ArrayRef<DotOp> dots, unsigned lanes) const override {
    DenseSet<Operation *> turned = turnedDots(dots, lanes);
    DenseMap<Operation *, DotForm> forms;
    for (DotOp dot : dots) {
      bool t = turned.contains(dot);
      int axis = t ? 0 : cast<RankedTensorType>(dot.getC().getType()).getRank() - 1;
      forms[dot] = {axis, axis, axis, t};
    }
    return forms;
  }

  // The reduces the lanes fold: one operand of f32, combined by one max, min or add of the element
  // and the accumulator.
  bool foldsAcross(ReduceOp red) const override {
    if (red.getNumOperands() != 1 ||
        !cast<RankedTensorType>(red.getOperand(0).getType()).getElementType().isF32())
      return false;
    Block &body = red.getCombineOp().front();
    if (body.getOperations().size() != 2)
      return false;
    Operation &comb = body.front();
    if (!isa<arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp, arith::AddFOp>(comb))
      return false;
    return llvm::is_contained(comb.getOperands(), body.getArgument(0)) &&
           llvm::is_contained(comb.getOperands(), body.getArgument(1));
  }
};

}

const LayoutRules &tpuLayoutRules() {
  static const TpuLayoutRules rules;
  return rules;
}

}
