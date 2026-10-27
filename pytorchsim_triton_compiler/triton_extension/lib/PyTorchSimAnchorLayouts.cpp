// tritongpu-pytorchsim-anchor-layouts: every layout this lane machine fixes, in one place before the one
// remove-layout-conversions: a dot's and a vcix's operands and result (turned where a reduce wants rows on
// lanes), a reduce's input off its axis, a long scan's input in lane rows, a reshape's or cat's result.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "PyTorchSim/CrossLaneFolds.h"
#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/PyTorchSimOps.h"
#include "PyTorchSim/PyTorchSimExternDialect.h"
#include "PyTorchSim/PyTorchSimExternOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DECL_TRITONGPUPYTORCHSIMANCHORLAYOUTS
#define GEN_PASS_DECL_TRITONGPUPYTORCHSIMSTRIPPINS
#define GEN_PASS_DEF_TRITONGPUPYTORCHSIMANCHORLAYOUTS
#define GEN_PASS_DEF_TRITONGPUPYTORCHSIMSTRIPPINS
#include "PyTorchSimExt/Passes.h.inc"

namespace {

Attribute blocked(MLIRContext *ctx, ArrayRef<unsigned> spt, unsigned axis, unsigned lanes) {
  unsigned rank = spt.size();
  SmallVector<unsigned> tpw(rank, 1), wpc(rank, 1), order;
  tpw[axis] = lanes;
  order.push_back(axis);
  for (int d = rank - 1; d >= 0; --d)
    if (unsigned(d) != axis)
      order.push_back(d);
  return BlockedEncodingAttr::get(ctx, spt, tpw, wpc, order,
                                  CGAEncodingAttr::get1CTALayout(ctx, rank));
}

Attribute laneLayout(MLIRContext *ctx, unsigned rank, unsigned axis, unsigned lanes) {
  return blocked(ctx, SmallVector<unsigned>(rank, 1), axis, lanes);
}

Attribute bankLayout(MLIRContext *ctx, ArrayRef<int64_t> shape, unsigned axis, unsigned lanes) {
  SmallVector<unsigned> spt(shape.begin(), shape.end());
  return blocked(ctx, spt, axis, lanes);
}

SmallVector<int> laneDims(RankedTensorType t) {
  SmallVector<int> out;
  LinearLayout ll = toLinearLayout(t);
  auto kLane = StringAttr::get(t.getContext(), "lane");
  if (!llvm::is_contained(ll.getInDimNames(), kLane))
    return out;
  for (int p = 0; p < ll.getInDimSizeLog2(kLane); ++p)
    for (auto [d, v] : llvm::enumerate(ll.getBasis(kLane, p)))
      if (v && t.getShape()[d] != 1 && !llvm::is_contained(out, int(d)))
        out.push_back(d);
  return out;
}

Value convertTo(OpBuilder &b, Location loc, Value v, Attribute enc) {
  auto t = cast<RankedTensorType>(v.getType());
  if (t.getEncoding() == enc)
    return v;
  return ConvertLayoutOp::create(b, loc, t.cloneWithEncoding(enc), v);
}

// `v` kept in `enc`: a pytorchsim_extern.layout_constraint, which remove-layout-conversions propagates from
// (it is a LayoutAnchor) and strip-pins removes.
Value pin(OpBuilder &b, Location loc, Value v, Attribute enc) {
  Value in = convertTo(b, loc, v, enc);
  return ::pytorchsim::external::LayoutConstraintOp::create(b, loc, in.getType(), in, enc).getResult();
}

void retypeResult(Operation *op, unsigned i, Type want) {
  Value r = op->getResult(i);
  Type old = r.getType();
  if (old == want)
    return;
  r.setType(want);
  if (r.use_empty())
    return;
  OpBuilder b(op->getContext());
  b.setInsertionPointAfter(op);
  auto back = ConvertLayoutOp::create(b, op->getLoc(), old, r);
  r.replaceAllUsesExcept(back, back);
}

Attribute laneLayoutOf(RankedTensorType t, unsigned axis, unsigned lanes) {
  MLIRContext *ctx = t.getContext();
  Attribute enc = laneLayout(ctx, t.getRank(), axis, lanes);
  LinearLayout ll = toLinearLayout(t.getShape(), enc);
  auto bases = ll.getBases();
  auto &regs = bases[StringAttr::get(ctx, "register")];
  std::stable_partition(regs.begin(), regs.end(),
                        [&](const std::vector<int32_t> &b) { return b[axis] < int32_t(lanes); });
  return LinearEncodingAttr::get(ctx, LinearLayout(std::move(bases), ll.getOutDims(),
                                                   ll.isSurjective()));
}

bool isZeroSplat(Value v) {
  while (auto cvt = v.getDefiningOp<ConvertLayoutOp>())
    v = cvt.getSrc();
  if (auto splat = v.getDefiningOp<SplatOp>())
    v = splat.getSrc();
  Attribute value;
  if (!matchPattern(v, m_Constant(&value)))
    return false;
  if (auto dense = dyn_cast<SplatElementsAttr>(value))
    value = dense.getSplatValue<Attribute>();
  if (auto f = dyn_cast<FloatAttr>(value))
    return f.getValue().isZero();
  if (auto i = dyn_cast<IntegerAttr>(value))
    return i.getValue().isZero();
  return false;
}

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

// The array takes A with K on the lanes, B and the product with N. A turned dot is computed as
// C^T = B^T A^T: B is the array's A, A its B, and every operand has its first axis on the lanes,
// the rows the reduce keeps there.
void anchorDot(DotOp dot, unsigned lanes, bool turned) {
  OpBuilder b(dot);
  Location loc = dot.getLoc();
  auto ct = cast<RankedTensorType>(dot.getC().getType());
  unsigned axis = turned ? 0 : ct.getRank() - 1;
  auto laid = [&](Value v) {
    return convertTo(b, loc, v, laneLayoutOf(cast<RankedTensorType>(v.getType()), axis, lanes));
  };
  Value a = laid(dot.getA()), bv = laid(dot.getB());

  Value c;
  if (isZeroSplat(dot.getC())) {
    auto t = ct.cloneWithEncoding(laneLayoutOf(ct, axis, lanes));
    Type elt = t.getElementType();
    Value zero = arith::ConstantOp::create(b, loc, elt, cast<TypedAttr>(b.getZeroAttr(elt)));
    Value empty = tensor::EmptyOp::create(b, loc, t, ValueRange{});
    c = linalg::FillOp::create(b, loc, ValueRange{zero}, ValueRange{empty}).result();
  } else {
    c = laid(dot.getC());
  }
  ArrayAttr maps;
  if (turned) {
    // The array's m walks the dot's N and its n the dot's M: B is [k, m], A [n, k], C [n, m].
    AffineExpr m, n, k;
    bindDims(b.getContext(), m, n, k);
    maps = b.getAffineMapArrayAttr({AffineMap::get(3, 0, {k, m}, b.getContext()),
                                    AffineMap::get(3, 0, {n, k}, b.getContext()),
                                    AffineMap::get(3, 0, {n, m}, b.getContext())});
    std::swap(a, bv);
  }
  auto product = ::pytorchsim::DotOp::create(b, loc, c.getType(), a, bv, c, maps);
  auto back = ConvertLayoutOp::create(b, loc, dot.getType(), product.getResult());
  dot.replaceAllUsesWith(back.getResult());
  dot.erase();
}

// The layout an anchor already fixed for v: the product of a pytorchsim.dot that reaches v through
// elementwise ops of v's shape, when exactly one such layout does.
Attribute anchoredLayout(Value v) {
  auto shape = cast<RankedTensorType>(v.getType()).getShape();
  Attribute found;
  bool clash = false;
  SmallVector<Value> work{v};
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (!work.empty() && !clash) {
    Operation *def = work.pop_back_val().getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    if (auto cvt = dyn_cast<ConvertLayoutOp>(def)) {
      if (cvt.getSrc().getDefiningOp<::pytorchsim::DotOp>()) {
        Attribute enc = cvt.getSrc().getType().getEncoding();
        clash = found && found != enc;
        found = enc;
      } else {
        work.push_back(cvt.getSrc());
      }
      continue;
    }
    if (!def->hasTrait<OpTrait::Elementwise>())
      continue;
    for (Value o : def->getOperands())
      if (auto ot = dyn_cast<RankedTensorType>(o.getType()); ot && ot.getShape() == shape)
        work.push_back(o);
  }
  return clash ? Attribute() : found;
}

// A reduce whose input an anchor laid along the reduced axis, on lanes, over more elements than
// there are lanes, reduces in two steps: first the elements each lane holds of the axis, in that
// anchored layout, then only the [.., lanes, ..] partial in the layout anchorReduce gives it. Both
// the first step's input and its split view are pinned, so the second step's layout does not move
// back over them. `onLanes`, when given, is the layout to split in instead of an anchor's, and
// splits a reduce whose other dims are all 1 too.
void splitLaneReduce(ReduceOp red, unsigned lanes, Attribute onLanes = {}) {
  auto t = cast<RankedTensorType>(red.getOperand(0).getType());
  int rank = t.getRank(), axis = red.getAxis();
  auto shape = t.getShape();
  int64_t n = shape[axis];
  if (rank < 2 || n <= lanes || n % lanes ||
      (!onLanes &&
       llvm::none_of(llvm::seq(rank), [&](int d) { return d != axis && shape[d] != 1; })))
    return;
  Attribute inLane = onLanes ? onLanes : anchoredLayout(red.getOperand(0));
  if (!inLane || !llvm::is_contained(laneDims(t.cloneWithEncoding(inLane)), axis))
    return;
  Location loc = red.getLoc();
  SmallVector<int64_t> split(shape.begin(), shape.end());
  split[axis] = n / lanes;
  split.insert(split.begin() + axis + 1, lanes);
  auto *infer = dyn_cast<DialectInferLayoutInterface>(&inLane.getDialect());
  Attribute splitEnc;
  if (!infer || failed(infer->inferReshapeOpEncoding(shape, inLane, split, splitEnc,
                                                     /*allowReorder=*/false, loc)))
    return;
  OpBuilder b(red);
  SmallVector<Value> parts;
  for (Value v : red.getOperands()) {
    auto vt = cast<RankedTensorType>(v.getType());
    Value whole = pin(b, loc, v, inLane);
    Value r = ReshapeOp::create(b, loc, RankedTensorType::get(split, vt.getElementType(), splitEnc),
                                whole, /*allowReorder=*/false);
    parts.push_back(pin(b, loc, r, splitEnc));
  }
  auto first = ReduceOp::create(b, loc, parts, axis);
  IRMapping map;
  red.getCombineOp().cloneInto(&first.getCombineOp(), map);
  for (unsigned i = 0; i < red->getNumOperands(); ++i)
    red->setOperand(i, first->getResult(i));
}

// The layout v is made in, under the conversions between its maker and its reader.
Attribute madeLayout(Value v) {
  while (auto c = v.getDefiningOp<ConvertLayoutOp>())
    v = c.getSrc();
  return cast<RankedTensorType>(v.getType()).getEncoding();
}

// The reduces the lanes fold: one operand, combined by one combiner of the element and the accumulator that
// the target's cross-lane unit folds over that element type (pytorchsim::foldOf).
bool foldsAcrossLanes(ReduceOp red, const ::pytorchsim::CrossLaneFolds &unit) {
  if (red.getNumOperands() != 1)
    return false;
  Block &body = red.getCombineOp().front();
  if (body.getOperations().size() != 2)
    return false;
  Operation &comb = body.front();
  Type elem = cast<RankedTensorType>(red.getOperand(0).getType()).getElementType();
  if (!::pytorchsim::foldOf(unit, &comb, elem))
    return false;
  return llvm::is_contained(comb.getOperands(), body.getArgument(0)) &&
         llvm::is_contained(comb.getOperands(), body.getArgument(1));
}

// The layout that lays a reduce's axis across the lanes: the axis's lanes first, then each lane's
// rows, its wraps of the axis outermost -- a lane holds its rows of every lap.
Attribute axisOnLanes(MLIRContext *ctx, int rank, int axis, unsigned lanes) {
  SmallVector<unsigned> tpw(rank, 1), wpc(rank, 1), order;
  tpw[axis] = lanes;
  for (int d = rank - 1; d >= 0; --d)
    if (d != axis)
      order.push_back(d);
  order.push_back(axis);
  return BlockedEncodingAttr::get(ctx, SmallVector<unsigned>(rank, 1), tpw, wpc, order,
                                  CGAEncodingAttr::get1CTALayout(ctx, rank));
}

// THE ROWS ONE LANE HOLDS, when an address in the kernel reads a row through `row // K`: K
// consecutive rows read one row of that operand, so holding them in one lane keeps the
// operand's broadcast in that lane's registers -- held by rows across the lanes, the operand
// has to be spread over the lanes after it is loaded. All of a lane's rows, `extent / lanes`,
// when that is a multiple of every such K; 1, the rows across the lanes, otherwise.
unsigned rowsPerLane(Operation *root, int dim, int64_t extent, unsigned lanes) {
  if (extent % lanes != 0)
    return 1;
  int64_t held = extent / lanes;
  bool found = false;
  bool fits = true;
  root->walk([&](arith::DivSIOp div) {
    auto t = dyn_cast<RankedTensorType>(div.getType());
    if (!t || t.getRank() <= dim || t.getShape()[dim] != extent)
      return;
    for (auto [d, n] : llvm::enumerate(t.getShape()))
      if (int(d) != dim && n != 1)
        return;
    APInt k;
    if (!matchPattern(div.getRhs(), m_ConstantInt(&k)) || k.getSExtValue() <= 1)
      return;
    // ONLY AN ADDRESS: the quotient reaches a load's pointer.
    SmallVector<Value> todo{div.getResult()};
    DenseSet<Value> seen{div.getResult()};
    bool reaches = false;
    while (!todo.empty() && !reaches) {
      Value v = todo.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (auto load = dyn_cast<LoadOp>(user); load && load.getPtr() == v) {
          reaches = true;
          break;
        }
        if (isa<LoadOp, StoreOp>(user))
          continue;
        for (Value r : user->getResults())
          if (isa<RankedTensorType>(r.getType()) && seen.insert(r).second)
            todo.push_back(r);
      }
    }
    if (!reaches)
      return;
    found = true;
    fits &= held % k.getSExtValue() == 0;
  });
  return found && fits ? unsigned(held) : 1u;
}

// A short axis over many rows, held in the registers, is held outermost there: the reduce then adds
// whole register groups of rows, where held innermost each row is a group of its own.
Attribute reduceAcrossRows(Attribute enc, RankedTensorType t, int axis, unsigned lanes, unsigned groupBits) {
  auto bl = dyn_cast<BlockedEncodingAttr>(enc);
  if (!bl || !groupBits || bl.getThreadsPerWarp()[axis] != 1)
    return enc;
  int64_t group = groupBits / t.getElementTypeBitWidth();
  int64_t rows = t.getNumElements() / t.getShape()[axis] / lanes;
  if (t.getShape()[axis] >= group || rows < group)
    return enc;
  SmallVector<unsigned> order;
  for (unsigned d : bl.getOrder())
    if (d != unsigned(axis))
      order.push_back(d);
  order.push_back(unsigned(axis));
  return BlockedEncodingAttr::get(t.getContext(), bl.getSizePerThread(), bl.getThreadsPerWarp(),
                                  bl.getWarpsPerCTA(), order, bl.getCGALayout());
}

// A reduce the lanes fold (foldsAcrossLanes) keeps its axis on the lanes when every lane rides it, in its
// input's layout (an axis longer than the lanes is split by splitLaneReduce first); any other reduce takes
// the axis off the lanes.
void anchorReduce(ReduceOp red, unsigned lanes, unsigned groupBits,
                  const ::pytorchsim::CrossLaneFolds &unit) {
  // Rows that leave lanes idle, for a reduce the lanes do not fold: its axis goes onto the lanes all
  // the same, each lane reduces what it holds of the axis first, and only the lanes' partials leave
  // them -- a few rows of the whole axis would otherwise sit in a few lanes, or one.
  {
    auto t = cast<RankedTensorType>(red.getOperand(0).getType());
    int rank = t.getRank(), axis = red.getAxis();
    int64_t rows = 1;
    for (int d = 0; d < rank; ++d)
      if (d != axis)
        rows *= t.getShape()[d];
    if (!foldsAcrossLanes(red, unit) && rows < int64_t(lanes) && t.getShape()[axis] > int64_t(lanes))
      splitLaneReduce(red, lanes, axisOnLanes(red.getContext(), rank, axis, lanes));
    else
      splitLaneReduce(red, lanes);
  }
  auto t = cast<RankedTensorType>(red.getOperand(0).getType());
  int rank = t.getRank(), axis = red.getAxis();
  auto shape = t.getShape();
  Attribute enc = t.getEncoding();
  SmallVector<int> on = laneDims(t);
  bool keep = false;
  if (foldsAcrossLanes(red, unit)) {
    Attribute made = madeLayout(red.getOperand(0));
    for (Attribute a : {made, enc}) {
      if (!a || keep)
        continue;
      SmallVector<int> at = laneDims(t.cloneWithEncoding(a));
      bool alone = llvm::none_of(llvm::seq(rank), [&](int d) { return d != axis && shape[d] != 1; });
      if (at.size() == 1 && at[0] == axis && (shape[axis] <= int64_t(lanes) || alone)) {
        enc = a;
        on = at;
        keep = true;
      }
    }
  }
  // Rows that leave lanes idle: a reduce the lanes can fold whose other dims fill fewer lanes than
  // the machine has, and whose axis fills them, takes its axis onto the lanes -- as the v6e lays a
  // few rows' columns across its lanes and folds each row there.
  if (!keep && foldsAcrossLanes(red, unit) && !llvm::is_contained(on, axis) &&
      shape[axis] >= int64_t(lanes)) {
    int64_t rows = 1;
    for (int d : on)
      rows *= shape[d];
    Attribute made = madeLayout(red.getOperand(0));
    SmallVector<int> madeOn;
    if (made)
      madeOn = laneDims(t.cloneWithEncoding(made));
    if (rows < int64_t(lanes) && madeOn.size() == 1 && madeOn[0] == axis) {
      // made with the axis on the lanes already: read as it is made
      enc = made;
      on = {axis};
      keep = true;
    } else if (rows < int64_t(lanes)) {
      enc = axisOnLanes(red.getContext(), rank, axis, lanes);
      on = {axis};
      keep = true;
    }
  }
  if (on.empty() || (llvm::is_contained(on, axis) && !keep)) {
    int wide = -1;
    for (int d = 0; d < rank && wide < 0; ++d)
      if (d != axis && shape[d] != 1)
        wide = d;
    if (wide >= 0) {
      // Rows of the other axis on lanes, the reduced axis first among the registers so the
      // elements one lane reduces sit in consecutive registers.
      SmallVector<unsigned> tpw(rank, 1), wpc(rank, 1), order{unsigned(axis)};
      tpw[wide] = lanes;
      for (int d = rank - 1; d >= 0; --d)
        if (d != axis)
          order.push_back(d);
      enc = BlockedEncodingAttr::get(red.getContext(), SmallVector<unsigned>(rank, 1), tpw, wpc,
                                     order, CGAEncodingAttr::get1CTALayout(red.getContext(), rank));
    } else {
      int unit = axis;
      for (int d = 0; d < rank; ++d)
        if (d != axis) {
          unit = d;
          break;
        }
      enc = bankLayout(red.getContext(), shape, unit, lanes);
    }
  }
  enc = reduceAcrossRows(enc, t, axis, lanes, groupBits);
  // ROWS ON THE LANES ONE PER LANE, the reduce off them: held as runs instead when an address
  // reads them in runs (rowsPerLane). The reduce is the same in either -- each lane folds the
  // rows it holds.
  if (auto bl = dyn_cast<BlockedEncodingAttr>(enc)) {
    SmallVector<unsigned> spt(bl.getSizePerThread()), tpw(bl.getThreadsPerWarp());
    for (int d = 0; d < rank; ++d)
      if (d != axis && tpw[d] == lanes && spt[d] == 1)
        spt[d] = rowsPerLane(red->getParentOfType<FuncOp>(), d, shape[d], lanes);
    if (spt != SmallVector<unsigned>(bl.getSizePerThread()))
      enc = BlockedEncodingAttr::get(red.getContext(), spt, tpw, bl.getWarpsPerCTA(),
                                     bl.getOrder(), bl.getCGALayout());
  }
  OpBuilder b(red);
  for (OpOperand &o : red->getOpOperands())
    o.set(pin(b, red.getLoc(), o.get(), enc));
  if (rank == 1)
    return;
  for (unsigned i = 0; i < red->getNumResults(); ++i) {
    auto rt = cast<RankedTensorType>(red->getResult(i).getType());
    auto want = rt.cloneWithEncoding(
        SliceEncodingAttr::get(red.getContext(), axis, cast<DistributedEncodingTrait>(enc)));
    retypeResult(red, i, want);
  }
}

void anchorScan(ScanOp scan, unsigned lanes) {
  auto t = cast<RankedTensorType>(scan.getOperand(0).getType());
  int axis = scan.getAxis();
  int64_t n = t.getShape()[axis];
  if (n <= lanes || n % lanes || !llvm::is_contained(laneDims(t), axis))
    return;
  SmallVector<unsigned> spt(t.getRank(), 1);
  spt[axis] = lanes;
  Attribute enc = blocked(scan.getContext(), spt, axis, lanes);
  OpBuilder b(scan);
  for (OpOperand &o : scan->getOpOperands())
    o.set(pin(b, scan.getLoc(), o.get(), enc));
  for (unsigned i = 0; i < scan->getNumResults(); ++i)
    retypeResult(scan, i, cast<RankedTensorType>(scan->getResult(i).getType()).cloneWithEncoding(enc));
}

void anchorReorderReshape(ReshapeOp op) {
  if (!op.getAllowReorder())
    return;
  auto src = op.getSrc().getType();
  auto dst = op.getType();
  auto *infer = dyn_cast<DialectInferLayoutInterface>(&src.getEncoding().getDialect());
  Attribute inPlace;
  if (!infer || failed(infer->inferReshapeOpEncoding(src.getShape(), src.getEncoding(),
                                                     dst.getShape(), inPlace,
                                                     op.getAllowReorder(), op.getLoc())))
    return;
  retypeResult(op, 0, dst.cloneWithEncoding(inPlace));
}

void anchorCat(CatOp op) {
  auto src = cast<RankedTensorType>(op.getLhs().getType());
  if (!isa_and_nonnull<DistributedEncodingTrait>(src.getEncoding()) ||
      llvm::any_of(src.getShape(), [](int64_t n) { return !llvm::isPowerOf2_64(n); }))
    return;
  MLIRContext *ctx = op.getContext();
  LinearLayout ll = toLinearLayout(src);
  auto bases = ll.getBases();
  std::vector<int32_t> half(src.getRank(), 0);
  half[0] = src.getShape()[0];
  bases[StringAttr::get(ctx, "register")].push_back(half);
  auto outDims = ll.getOutDims();
  outDims[0].second *= 2;
  auto enc = LinearEncodingAttr::get(ctx, LinearLayout(std::move(bases), outDims,
                                                       true));

  Value r = op.getResult();
  Type old = r.getType();
  r.setType(op.getType().cloneWithEncoding(enc));
  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  Value pinned = pin(b, op.getLoc(), r, enc);
  Value back = convertTo(b, op.getLoc(), pinned, cast<RankedTensorType>(old).getEncoding());
  r.replaceAllUsesExcept(back, pinned.getDefiningOp());
}

// A layout a kernel named for a tile of `op` (a vcix, a layout_constraint), or a refusal when this
// machine cannot take it: one warp of exactly its lanes, one CTA, the tile's rank.
FailureOr<BlockedEncodingAttr> kernelLayout(Operation *op, Attribute layout, Value tile, unsigned lanes) {
  auto enc = dyn_cast_or_null<BlockedEncodingAttr>(layout);
  if (!enc) {
    op->emitError("pytorchsim-anchor-layouts: a ")
        << op->getName() << " layout is a #ttg.blocked encoding, got " << layout;
    return failure();
  }
  unsigned rank = cast<RankedTensorType>(tile.getType()).getRank();
  unsigned threads = product(enc.getThreadsPerWarp());
  if (enc.getRank() != rank || threads != lanes || product(enc.getWarpsPerCTA()) != 1 ||
      enc.getCGALayout() != CGAEncodingAttr::get1CTALayout(op->getContext(), rank)) {
    op->emitError("pytorchsim-anchor-layouts: the ") << op->getName() << " layout "
        << enc << " is not one this machine can take: it has " << lanes
        << " lanes in one warp and one CTA, and the layout puts " << threads
        << " threads in a warp, " << product(enc.getWarpsPerCTA()) << " warps in a CTA, rank "
        << enc.getRank() << " on a rank-" << rank << " tile";
    return failure();
  }
  return enc;
}

// A kernel's layout_constraint in the encoding it names, the uses after it in the layout they had: the
// converts on both sides keep the types; remove-layout-conversions propagates from it and drops them.
LogicalResult anchorConstraint(::pytorchsim::external::LayoutConstraintOp op, unsigned lanes) {
  FailureOr<BlockedEncodingAttr> enc = kernelLayout(op, op.getLayout(), op.getSrc(), lanes);
  if (failed(enc))
    return failure();
  auto old = cast<RankedTensorType>(op.getType());
  OpBuilder b(op);
  op.getSrcMutable().set(convertTo(b, op.getLoc(), op.getSrc(), *enc));
  op.getResult().setType(old.cloneWithEncoding(*enc));
  b.setInsertionPointAfter(op);
  Value back = convertTo(b, op.getLoc(), op.getResult(), old.getEncoding());
  op.getResult().replaceAllUsesExcept(back, back.getDefiningOp());
  return success();
}

// Each operand of a vcix pinned to the layout the kernel named for it, its result to its own.
LogicalResult anchorVcix(::pytorchsim::external::VcixOp op, unsigned lanes) {
  SmallVector<BlockedEncodingAttr> encs;
  for (auto [layout, a] : llvm::zip(op.getLayouts(), op.getArgs())) {
    FailureOr<BlockedEncodingAttr> enc = kernelLayout(op, layout, a, lanes);
    if (failed(enc))
      return failure();
    encs.push_back(*enc);
  }
  Value r = op.getResult();
  FailureOr<BlockedEncodingAttr> out;
  if (r && failed(out = kernelLayout(op, op.getOutLayoutAttr(), r, lanes)))
    return failure();
  OpBuilder b(op);
  for (auto [o, enc] : llvm::zip(op.getArgsMutable(), encs))
    o.set(pin(b, op.getLoc(), o.get(), enc));
  if (!r)
    return success();
  Type old = r.getType();
  r.setType(cast<RankedTensorType>(old).cloneWithEncoding(*out));
  b.setInsertionPointAfter(op);
  Value pinned = pin(b, op.getLoc(), r, *out);
  Value back = convertTo(b, op.getLoc(), pinned, cast<RankedTensorType>(old).getEncoding());
  r.replaceAllUsesExcept(back, pinned.getDefiningOp());
  return success();
}

LogicalResult refuseWhatPropagationNeverFinishes(ModuleOp mod) {
  WalkResult r = mod.walk([](scf::ForOp loop) {
    bool rank3 = llvm::any_of(loop.getRegionIterArgs(), [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && t.getRank() >= 3;
    });
    if (!rank3)
      return WalkResult::advance();
    WalkResult inner = loop.getBody()->walk([](TransOp t) {
      return t.getType().getRank() >= 3 ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (inner.wasInterrupted()) {
      loop.emitError("pytorchsim-anchor-layouts: a loop carrying a rank-3 tile with a tt.trans in "
                     "its body -- remove-layout-conversions does not terminate on this shape");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(r.wasInterrupted());
}

}

class TritonGPUPyTorchSimAnchorLayoutsPass
    : public impl::TritonGPUPyTorchSimAnchorLayoutsBase<TritonGPUPyTorchSimAnchorLayoutsPass> {
public:
  using TritonGPUPyTorchSimAnchorLayoutsBase::TritonGPUPyTorchSimAnchorLayoutsBase;
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod->setAttr("ttg.global-access-any-layout", UnitAttr::get(mod.getContext()));
    if (failed(refuseWhatPropagationNeverFinishes(mod)))
      return signalPassFailure();
    unsigned lanes = TritonGPUDialect::getThreadsPerWarp(mod);
    FailureOr<::pytorchsim::CrossLaneFolds> unit = ::pytorchsim::crossLaneFoldsOf(targetDialect);
    if (failed(unit)) {
      mod.emitError() << "target " << targetDialect << " is no registered target's dialect";
      return signalPassFailure();
    }
    SmallVector<DotOp> dots;
    SmallVector<ReduceOp> reduces;
    SmallVector<ReshapeOp> reshapes;
    SmallVector<ScanOp> scans;
    SmallVector<CatOp> cats;
    SmallVector<::pytorchsim::external::VcixOp> vcixes;
    SmallVector<::pytorchsim::external::LayoutConstraintOp> constraints;
    mod.walk([&](Operation *op) {
      if (auto v = dyn_cast<::pytorchsim::external::VcixOp>(op))
        vcixes.push_back(v);
      if (auto c = dyn_cast<::pytorchsim::external::LayoutConstraintOp>(op))
        constraints.push_back(c);
      if (auto c = dyn_cast<CatOp>(op); c && isa<RankedTensorType>(c.getType()))
        cats.push_back(c);
      if (auto sc = dyn_cast<ScanOp>(op); sc && isa<RankedTensorType>(sc.getOperand(0).getType()))
        scans.push_back(sc);
      if (auto d = dyn_cast<DotOp>(op))
        dots.push_back(d);
      else if (auto r = dyn_cast<ReduceOp>(op);
               r && isa<RankedTensorType>(r.getOperand(0).getType()))
        reduces.push_back(r);
      else if (auto s = dyn_cast<ReshapeOp>(op))
        reshapes.push_back(s);
    });
    for (::pytorchsim::external::LayoutConstraintOp c : constraints)
      if (failed(anchorConstraint(c, lanes)))
        return signalPassFailure();
    for (::pytorchsim::external::VcixOp v : vcixes)
      if (failed(anchorVcix(v, lanes)))
        return signalPassFailure();
    DenseSet<Operation *> turned = turnedDots(dots, lanes);
    for (DotOp d : dots)
      anchorDot(d, lanes, turned.contains(d));
    for (ReshapeOp s : reshapes)
      anchorReorderReshape(s);
    for (CatOp c : cats)
      anchorCat(c);
    for (ReduceOp r : reduces)
      anchorReduce(r, lanes, groupBits, *unit);
    for (ScanOp sc : scans)
      anchorScan(sc, lanes);
  }
};

class TritonGPUPyTorchSimStripPinsPass
    : public impl::TritonGPUPyTorchSimStripPinsBase<TritonGPUPyTorchSimStripPinsPass> {
public:
  void runOnOperation() override {
    SmallVector<::pytorchsim::external::LayoutConstraintOp> pins;
    getOperation().walk([&](::pytorchsim::external::LayoutConstraintOp p) { pins.push_back(p); });
    for (::pytorchsim::external::LayoutConstraintOp p : pins) {
      p.getResult().replaceAllUsesWith(p.getSrc());
      p->erase();
    }
  }
};

}
}
}
