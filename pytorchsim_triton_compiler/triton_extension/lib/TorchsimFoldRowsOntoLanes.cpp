// torchsim-fold-rows-onto-lanes: before any layout is assigned, a kernel whose rows are fewer
// than the lanes is written so every lane works, as the machine spreads a row over its lanes.
//
// One row over [1, n] (n a multiple of the lanes): the reduce is viewed as [n / lanes, lanes] and
// transposed to [lanes, n / lanes] -- element j at (j % lanes, j / lanes), the default layout's own
// lanes, so no element moves -- then each lane folds its n / lanes elements and only the lanes'
// partials are folded across.
//
// R rows (1 < R < lanes, k = lanes / R lanes a row, n a multiple of k): the whole kernel runs in
// the folded space, every [R, n] tile computed as [R * k, n / k] -- element (r, b * n / k + a) at
// lane r * k + b, depth a, so each lane holds a contiguous chunk of its row and the default
// layout's one lane dim holds k lanes of every row. Every value is made again there (Remake)
// rather than moved; loads, stores and elementwise ops run on the folded tiles as they are (the
// same elements in another order); a reduce over the row folds in each lane, then the k lanes of
// each row. A kernel with any other use of an [R, n] tile is left as it is (TORCHSIM_FOLD_DEBUG
// names the op).
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <cstdlib>

namespace mlir::triton {
#define GEN_PASS_DECL_TORCHSIMFOLDROWSONTOLANES
#define GEN_PASS_DEF_TORCHSIMFOLDROWSONTOLANES
#include "TorchsimExt/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace mlir::triton;

ReduceOp cloneReduce(OpBuilder &b, Location loc, ReduceOp like, ValueRange srcs, int axis) {
  auto red = ReduceOp::create(b, loc, srcs, axis);
  IRMapping m;
  like.getCombineOp().cloneInto(&red.getCombineOp(), m);
  return red;
}

bool foldOneRow(ReduceOp red, int64_t lanes) {
  if (red->getNumOperands() != 1)
    return false;
  auto t = dyn_cast<RankedTensorType>(red.getOperand(0).getType());
  if (!t)
    return false;
  int rank = t.getRank(), axis = red.getAxis();
  auto shape = t.getShape();
  int64_t n = shape[axis];
  if (n <= lanes || n % lanes)
    return false;
  for (int d = 0; d < rank; ++d)
    if (d != axis && shape[d] != 1)
      return false;
  // A reshape's result (a tile flattened to fold it whole) keeps the order it was given.
  if (red.getOperand(0).getDefiningOp<ReshapeOp>())
    return false;
  OpBuilder b(red);
  Location loc = red.getLoc();
  Value view = ReshapeOp::create(b, loc, ArrayRef<int64_t>{n / lanes, lanes}, red.getOperand(0));
  Value byLane = TransOp::create(b, loc, view, ArrayRef<int32_t>{1, 0});
  auto inLane = cloneReduce(b, loc, red, ValueRange{byLane}, 1);
  auto across = cloneReduce(b, loc, red, inLane.getResults(), 0);
  Value r = across->getResult(0);
  Type want = red->getResult(0).getType();
  if (r.getType() != want)
    r = SplatOp::create(b, loc, want, r);
  red->getResult(0).replaceAllUsesWith(r);
  red.erase();
  return true;
}

// R rows folded onto the lanes: [R, n] computed as [R * k, n / k], element (r, b * n / k + a) at
// lane p = r * k + b, depth a -- each lane a contiguous chunk of its row, so an access to a
// row-major [R, n] is p * (n / k) + a, one stride a lane: [R, n] reshaped, nothing permuted.
struct RowFold {
  int64_t rows, n, k;
  int64_t lanes() const { return rows * k; }
  int64_t depth() const { return n / k; }
  bool isTile(Type t) const { return hasShape(t, rows, n); }
  static bool hasShape(Type t, int64_t a, int64_t b) {
    auto rt = dyn_cast<RankedTensorType>(t);
    return rt && rt.getRank() == 2 && rt.getShape()[0] == a && rt.getShape()[1] == b;
  }
  RankedTensorType tileOf(Type elem) const {
    return RankedTensorType::get({lanes(), depth()}, elem);
  }
  RankedTensorType rowOf(Type elem) const { return RankedTensorType::get({lanes(), 1}, elem); }
};

Type elemOf(Value v) { return cast<RankedTensorType>(v.getType()).getElementType(); }

Value splatI32(OpBuilder &b, Location loc, ArrayRef<int64_t> shape, int64_t v) {
  return arith::ConstantOp::create(b, loc, DenseElementsAttr::get(
      RankedTensorType::get(shape, b.getI32Type()), b.getI32IntegerAttr(v).getValue()));
}

// The values of a kernel, made again in the folded space where they are used -- nothing moved
// but a row's statistic. A row's value [R, 1] becomes [R * k, 1] (lane p holds row p / k's), a
// column's [1, n] becomes the whole [R * k, n / k] (lane p, depth a holds column
// (p % k) * n / k + a). Sources: the row and column indices (make_range, in 1-D or expanded),
// splats, splat constants; on them, in 1-D or 2-D, arithmetic, addptr and loads are redone -- so a
// pointer is always a splat base and folded offsets, never a reshaped tile of pointers. A row's
// value made from a reduce's result (a statistic, after any arithmetic on it) is the one thing
// moved: broadcast to [R, k] and viewed as [R * k] by a reshape the anchor pass lays on the lanes
// (torchsim.lanes_of_rows), a move of R * k elements.
class Remake {
public:
  Remake(const RowFold &rf) : rf(rf) {}

  // Whether `v`, of shape [R, 1] (row) or [1, n] (column), can be made folded.
  bool can(Value v, bool row) {
    Operation *def = v.getDefiningOp();
    if (auto ex = dyn_cast_or_null<ExpandDimsOp>(def)) {
      if (ex.getAxis() != (row ? 1 : 0))
        return false;
      // A row's statistic: made from a reduce's result, expanded. Moved, unless a pointer.
      if (fromReduce(ex.getSrc()))
        return row && !isa<PointerType>(elemOf(v));
      return can(ex.getSrc(), row);     // the 1-D value it widens, made the same way
    }
    if (auto mr = dyn_cast_or_null<MakeRangeOp>(def))
      return mr.getStart() == 0 && mr.getEnd() == (row ? rf.rows : rf.n);
    if (isa_and_nonnull<SplatOp>(def))
      return true;
    if (auto c = dyn_cast_or_null<arith::ConstantOp>(def)) {
      auto d = dyn_cast<DenseElementsAttr>(c.getValue());
      return d && d.isSplat();
    }
    if (!def || def->getNumRegions() || def->getNumResults() != 1 ||
        !(isa<arith::ArithDialect, math::MathDialect>(def->getDialect()) ||
          isa<AddPtrOp, LoadOp, ExternElementwiseOp>(def)))
      return false;
    for (Value o : def->getOperands())
      if (isa<RankedTensorType>(o.getType()) &&
          (cast<RankedTensorType>(o.getType()).getShape() !=
               cast<RankedTensorType>(v.getType()).getShape() ||
           !can(o, row)))
        return false;
    return true;
  }

  Value make(OpBuilder &b, Location loc, Value v, bool row) {
    auto &memo = row ? rows : cols;
    if (Value got = memo.lookup(v))
      return got;
    Type elem = elemOf(v);
    RankedTensorType want = row ? rf.rowOf(elem) : rf.tileOf(elem);
    Operation *def = v.getDefiningOp();
    Value out;
    if (auto ex = dyn_cast<ExpandDimsOp>(def)) {
      out = fromReduce(ex.getSrc()) ? statistic(b, loc, v) : make(b, loc, ex.getSrc(), row);
    } else if (isa<MakeRangeOp>(def)) {
      out = row ? rowIndex(b, loc) : columnIndex(b, loc);
    } else if (auto sp = dyn_cast<SplatOp>(def)) {
      out = SplatOp::create(b, loc, want, sp.getSrc());
    } else if (auto c = dyn_cast<arith::ConstantOp>(def)) {
      out = arith::ConstantOp::create(b, loc, want,
                                      cast<DenseElementsAttr>(c.getValue()).reshape(want));
    } else {
      IRMapping m;
      for (Value o : def->getOperands())
        if (isa<RankedTensorType>(o.getType()))
          m.map(o, make(b, loc, o, row));
      Operation *cl = b.clone(*def, m);
      cl->getResult(0).setType(want);
      out = cl->getResult(0);
    }
    memo[v] = out;
    return out;
  }

  // A [R, 1] or [1, n] value broadcast to the [R, n] tile, folded.
  Value tile(OpBuilder &b, Location loc, Value src) {
    auto st = cast<RankedTensorType>(src.getType());
    auto want = rf.tileOf(st.getElementType());
    if (st.getShape()[0] == 1 && st.getShape()[1] == 1)
      return BroadcastOp::create(b, loc, want, src);
    if (st.getShape()[0] == 1)
      return make(b, loc, src, /*row=*/false);
    return BroadcastOp::create(b, loc, want, make(b, loc, src, /*row=*/true));
  }

private:
  // Whether a 1-D value is a reduce's result, or computed from one.
  static bool fromReduce(Value v) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    if (isa<ReduceOp>(def))
      return true;
    if (def->getNumRegions())
      return false;
    return llvm::any_of(def->getOperands(), [](Value o) {
      return isa<RankedTensorType>(o.getType()) && fromReduce(o);
    });
  }

  Value rowIndex(OpBuilder &b, Location loc) {
    Value p = MakeRangeOp::create(b, loc, RankedTensorType::get({rf.lanes()}, b.getI32Type()), 0,
                                  rf.lanes());
    p = ExpandDimsOp::create(b, loc, p, 1);
    return arith::DivSIOp::create(b, loc, p, splatI32(b, loc, {rf.lanes(), 1}, rf.k));
  }

  Value columnIndex(OpBuilder &b, Location loc) {
    auto full = rf.tileOf(b.getI32Type());
    Value p = MakeRangeOp::create(b, loc, RankedTensorType::get({rf.lanes()}, b.getI32Type()), 0,
                                  rf.lanes());
    p = ExpandDimsOp::create(b, loc, p, 1);
    Value inRow = arith::RemSIOp::create(b, loc, p, splatI32(b, loc, {rf.lanes(), 1}, rf.k));
    Value chunk = arith::MulIOp::create(b, loc, inRow,
                                        splatI32(b, loc, {rf.lanes(), 1}, rf.depth()));
    Value a = MakeRangeOp::create(b, loc, RankedTensorType::get({rf.depth()}, b.getI32Type()), 0,
                                  rf.depth());
    a = ExpandDimsOp::create(b, loc, a, 0);
    return arith::AddIOp::create(b, loc, BroadcastOp::create(b, loc, full, a),
                                 BroadcastOp::create(b, loc, full, chunk));
  }

  Value statistic(OpBuilder &b, Location loc, Value v) {
    Type elem = elemOf(v);
    Value wide = BroadcastOp::create(b, loc, RankedTensorType::get({rf.rows, rf.k}, elem), v);
    auto view = ReshapeOp::create(b, loc, ArrayRef<int64_t>{rf.lanes()}, wide);
    view->setAttr("torchsim.lanes_of_rows", b.getUnitAttr());
    return ExpandDimsOp::create(b, loc, view, 1);
  }

  const RowFold &rf;
  DenseMap<Value, Value> rows, cols;
};

// Whether every op touching an [R, n] tile is one the fold rewrites, and every value broadcast
// into a tile can be made folded.
bool foldable(triton::FuncOp f, const RowFold &rf, Remake &remake) {
  Operation *refused = nullptr;
  auto admits = [&](Operation *op) -> bool {
    if (isa<triton::FuncOp, ReturnOp>(op) || op->getParentOfType<ReduceOp>())
      return true;
    bool touches = llvm::any_of(op->getOperandTypes(), [&](Type t) { return rf.isTile(t); }) ||
                   llvm::any_of(op->getResultTypes(), [&](Type t) { return rf.isTile(t); });
    if (!touches)
      return true;
    if (auto r = dyn_cast<ReduceOp>(op))
      return r.getAxis() == 1 &&
             llvm::all_of(r->getOperandTypes(), [&](Type t) { return rf.isTile(t); });
    if (auto bc = dyn_cast<BroadcastOp>(op)) {
      auto st = cast<RankedTensorType>(bc.getSrc().getType());
      if (st.getShape()[0] == 1 && st.getShape()[1] == 1)
        return true;
      if (RowFold::hasShape(st, 1, rf.n))
        return remake.can(bc.getSrc(), /*row=*/false);
      if (RowFold::hasShape(st, rf.rows, 1))
        return remake.can(bc.getSrc(), /*row=*/true);
      return false;
    }
    if (isa<SplatOp>(op))
      return true;
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto d = dyn_cast<DenseElementsAttr>(c.getValue());
      return d && d.isSplat();
    }
    if (op->getNumRegions())
      return false;
    // Elementwise: every tensor operand and result an [R, n] tile.
    for (Type t : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()))
      if (isa<RankedTensorType>(t) && !rf.isTile(t))
        return false;
    return isa<LoadOp, StoreOp, AddPtrOp, ExternElementwiseOp>(op) ||
           isa<arith::ArithDialect, math::MathDialect>(op->getDialect());
  };
  f.walk([&](Operation *op) {
    if (!refused && !admits(op))
      refused = op;
  });
  if (refused && std::getenv("TORCHSIM_FOLD_DEBUG"))
    llvm::errs() << "torchsim-fold-rows-onto-lanes: not folded at " << *refused << "\n";
  return !refused;
}

bool foldRows(triton::FuncOp f, int64_t lanes) {
  if (f.getBody().getBlocks().size() != 1)
    return false;
  RowFold rf{0, 0, 0};
  f.walk([&](ReduceOp r) {
    auto t = dyn_cast<RankedTensorType>(r.getOperand(0).getType());
    if (rf.rows || !t || t.getRank() != 2 || r.getAxis() != 1)
      return;
    int64_t R = t.getShape()[0], n = t.getShape()[1];
    if (R > 1 && R < lanes && lanes % R == 0 && n % (lanes / R) == 0 && n > lanes / R)
      rf = {R, n, lanes / R};
  });
  if (!rf.rows)
    return false;
  Remake remake(rf);
  if (!foldable(f, rf, remake))
    return false;
  IRMapping folded;
  SmallVector<Operation *> old;
  for (Operation &opRef : llvm::make_early_inc_range(f.getBody().front())) {
    Operation *op = &opRef;
    if (isa<ReturnOp>(op))
      continue;
    bool touches = llvm::any_of(op->getOperandTypes(), [&](Type t) { return rf.isTile(t); }) ||
                   llvm::any_of(op->getResultTypes(), [&](Type t) { return rf.isTile(t); });
    if (!touches)
      continue;
    OpBuilder b(op);
    Location loc = op->getLoc();
    old.push_back(op);
    if (auto bc = dyn_cast<BroadcastOp>(op)) {
      folded.map(bc.getResult(), remake.tile(b, loc, bc.getSrc()));
      continue;
    }
    if (auto s = dyn_cast<SplatOp>(op)) {
      folded.map(s.getResult(), SplatOp::create(b, loc, rf.tileOf(elemOf(s)), s.getSrc()));
      continue;
    }
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto ft = rf.tileOf(elemOf(c));
      folded.map(c.getResult(), arith::ConstantOp::create(
          b, loc, ft, cast<DenseElementsAttr>(c.getValue()).reshape(ft)));
      continue;
    }
    if (auto r = dyn_cast<ReduceOp>(op)) {
      SmallVector<Value> srcs;
      for (Value v : r.getOperands())
        srcs.push_back(folded.lookup(v));
      auto inLane = cloneReduce(b, loc, r, srcs, 1);
      SmallVector<Value> byRow;
      for (Value v : inLane->getResults())
        byRow.push_back(ReshapeOp::create(b, loc, ArrayRef<int64_t>{rf.rows, rf.k}, v));
      auto across = cloneReduce(b, loc, r, byRow, 1);
      for (unsigned i = 0; i < r->getNumResults(); ++i)
        r->getResult(i).replaceAllUsesWith(across->getResult(i));
      continue;
    }
    Operation *clone = b.clone(*op, folded);
    for (auto [res, nres] : llvm::zip(op->getResults(), clone->getResults()))
      if (rf.isTile(res.getType())) {
        nres.setType(rf.tileOf(cast<RankedTensorType>(res.getType()).getElementType()));
        folded.map(res, nres);
      }
  }
  for (Operation *op : llvm::reverse(old))
    if (op->use_empty())
      op->erase();
  return true;
}

class FoldRowsOntoLanes
    : public triton::impl::TorchsimFoldRowsOntoLanesBase<FoldRowsOntoLanes> {
public:
  using TorchsimFoldRowsOntoLanesBase::TorchsimFoldRowsOntoLanesBase;

  void runOnOperation() override {
    SmallVector<ReduceOp> reduces;
    getOperation().walk([&](ReduceOp r) { reduces.push_back(r); });
    for (ReduceOp r : reduces)
      foldOneRow(r, lanes);
    getOperation().walk([&](triton::FuncOp f) { foldRows(f, lanes); });
  }
};

} // namespace
