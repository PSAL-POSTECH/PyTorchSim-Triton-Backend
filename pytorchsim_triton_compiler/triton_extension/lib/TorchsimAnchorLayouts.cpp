// tritongpu-torchsim-anchor-layouts: every layout this lane machine fixes, written into TritonGPU
// IR in one place before the one remove-layout-conversions -- a dot's and a vcix's operands and
// result (a dot turned around where a reduce wants its rows on the lanes), a reduce's input off
// the reduced axis, a long scan's input in rows of lanes, a reorder-allowed reshape's or a cat's
// result where its elements are.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "TorchSim/TorchSimDialect.h"
#include "TorchSim/TorchSimOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DECL_TRITONGPUTORCHSIMANCHORLAYOUTS
#define GEN_PASS_DECL_TRITONGPUTORCHSIMSTRIPPINS
#define GEN_PASS_DEF_TRITONGPUTORCHSIMANCHORLAYOUTS
#define GEN_PASS_DEF_TRITONGPUTORCHSIMSTRIPPINS
#include "TorchsimExt/Passes.h.inc"

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

Value pin(OpBuilder &b, Location loc, Value v, Attribute enc) {
  auto t = cast<RankedTensorType>(v.getType());
  int rank = t.getRank(), n = t.getShape().back();
  auto i32 = b.getI32Type();
  SmallVector<Attribute> encs{enc};
  for (int k = 0; k < rank - 1; ++k)
    encs.push_back(SliceEncodingAttr::get(b.getContext(), 0, cast<DistributedEncodingTrait>(encs.back())));
  Value idx = MakeRangeOp::create(b, loc, RankedTensorType::get({n}, i32, encs.back()), 0, n);
  for (int k = rank - 2; k >= 0; --k) {
    SmallVector<int64_t> shape(rank - 1 - k, 1);
    shape.push_back(n);
    idx = ExpandDimsOp::create(b, loc, RankedTensorType::get(shape, i32, encs[k]), idx, 0);
  }
  if (rank > 1)
    idx = BroadcastOp::create(b, loc, RankedTensorType::get(t.getShape(), i32, enc), idx);
  OperationState st(loc, GatherOp::getOperationName());
  st.addOperands({convertTo(b, loc, v, enc), idx});
  st.addTypes(t.cloneWithEncoding(enc));
  st.addAttribute("axis", b.getI32IntegerAttr(rank - 1));
  st.addAttribute("efficient_layout", b.getUnitAttr());
  st.addAttribute("torchsim.pin", b.getUnitAttr());
  return b.create(st)->getResult(0);
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
  auto product = ::torchsim::DotOp::create(b, loc, c.getType(), a, bv, c, maps);
  auto back = ConvertLayoutOp::create(b, loc, dot.getType(), product.getResult());
  dot.replaceAllUsesWith(back.getResult());
  dot.erase();
}

// The layout an anchor already fixed for v: the product of a torchsim.dot that reaches v through
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
      if (cvt.getSrc().getDefiningOp<::torchsim::DotOp>()) {
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
// back over them.
void splitLaneReduce(ReduceOp red, unsigned lanes) {
  auto t = cast<RankedTensorType>(red.getOperand(0).getType());
  int rank = t.getRank(), axis = red.getAxis();
  auto shape = t.getShape();
  int64_t n = shape[axis];
  if (rank < 2 || n <= lanes || n % lanes ||
      llvm::none_of(llvm::seq(rank), [&](int d) { return d != axis && shape[d] != 1; }))
    return;
  Attribute inLane = anchoredLayout(red.getOperand(0));
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

// The reduces the lanes fold: one operand of f32, combined by one max or add of the element and the
// accumulator.
bool foldsAcrossLanes(ReduceOp red) {
  if (red.getNumOperands() != 1 ||
      !cast<RankedTensorType>(red.getOperand(0).getType()).getElementType().isF32())
    return false;
  Block &body = red.getCombineOp().front();
  if (body.getOperations().size() != 2)
    return false;
  Operation &comb = body.front();
  if (!isa<arith::MaximumFOp, arith::MaxNumFOp, arith::AddFOp>(comb))
    return false;
  return llvm::is_contained(comb.getOperands(), body.getArgument(0)) &&
         llvm::is_contained(comb.getOperands(), body.getArgument(1));
}

void anchorReduce(ReduceOp red, unsigned lanes) {
  splitLaneReduce(red, lanes);
  auto t = cast<RankedTensorType>(red.getOperand(0).getType());
  int rank = t.getRank(), axis = red.getAxis();
  auto shape = t.getShape();
  Attribute enc = t.getEncoding();
  SmallVector<int> on = laneDims(t);
  // A reduce the lanes can fold (one f32 max or sum) keeps its axis on the lanes when every lane
  // rides that axis, read in the layout its input is made in -- the elements a fold combines are
  // the same in any order: each lane folds what it holds, and the lanes' partials fold through the
  // cross-lane unit when the reduce is lowered. An axis longer than the lanes beside another axis
  // is split by splitLaneReduce first. Any other reduce takes the axis off the lanes.
  bool keep = false;
  if (foldsAcrossLanes(red)) {
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
  if (!keep && foldsAcrossLanes(red) && !llvm::is_contained(on, axis) &&
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
      // the axis's lanes first, then each lane's rows, its wraps of the axis outermost: a lane
      // holds its rows of every lap
      SmallVector<unsigned> tpw(rank, 1), wpc(rank, 1), order;
      tpw[axis] = lanes;
      for (int d = rank - 1; d >= 0; --d)
        if (d != axis)
          order.push_back(d);
      order.push_back(axis);
      enc = BlockedEncodingAttr::get(red.getContext(), SmallVector<unsigned>(rank, 1), tpw, wpc,
                                     order, CGAEncodingAttr::get1CTALayout(red.getContext(), rank));
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

// A layout a kernel named for one vcix tile, or a refusal when this machine cannot take it: one
// warp of exactly its lanes, one CTA, the tile's rank.
FailureOr<BlockedEncodingAttr> vcixLayout(::torchsim::VcixOp op, Attribute layout, Value tile,
                                          unsigned lanes) {
  auto enc = dyn_cast_or_null<BlockedEncodingAttr>(layout);
  if (!enc) {
    op.emitError("torchsim-anchor-layouts: a vcix layout is a #ttg.blocked encoding, got ")
        << layout;
    return failure();
  }
  unsigned rank = cast<RankedTensorType>(tile.getType()).getRank();
  unsigned threads = product(enc.getThreadsPerWarp());
  if (enc.getRank() != rank || threads != lanes || product(enc.getWarpsPerCTA()) != 1 ||
      enc.getCGALayout() != CGAEncodingAttr::get1CTALayout(op.getContext(), rank)) {
    op.emitError("torchsim-anchor-layouts: the vcix layout ")
        << enc << " is not one this machine can take: it has " << lanes
        << " lanes in one warp and one CTA, and the layout puts " << threads
        << " threads in a warp, " << product(enc.getWarpsPerCTA()) << " warps in a CTA, rank "
        << enc.getRank() << " on a rank-" << rank << " tile";
    return failure();
  }
  return enc;
}

// Each operand of a vcix pinned to the layout the kernel named for it, its result to its own.
LogicalResult anchorVcix(::torchsim::VcixOp op, unsigned lanes) {
  SmallVector<BlockedEncodingAttr> encs;
  for (auto [layout, a] : llvm::zip(op.getLayouts(), op.getArgs())) {
    FailureOr<BlockedEncodingAttr> enc = vcixLayout(op, layout, a, lanes);
    if (failed(enc))
      return failure();
    encs.push_back(*enc);
  }
  Value r = op.getResult();
  FailureOr<BlockedEncodingAttr> out;
  if (r && failed(out = vcixLayout(op, op.getOutLayoutAttr(), r, lanes)))
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
      loop.emitError("torchsim-anchor-layouts: a loop carrying a rank-3 tile with a tt.trans in "
                     "its body -- remove-layout-conversions does not terminate on this shape");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(r.wasInterrupted());
}

}

class TritonGPUTorchsimAnchorLayoutsPass
    : public impl::TritonGPUTorchsimAnchorLayoutsBase<TritonGPUTorchsimAnchorLayoutsPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod->setAttr("ttg.global-access-any-layout", UnitAttr::get(mod.getContext()));
    if (failed(refuseWhatPropagationNeverFinishes(mod)))
      return signalPassFailure();
    unsigned lanes = TritonGPUDialect::getThreadsPerWarp(mod);
    SmallVector<DotOp> dots;
    SmallVector<ReduceOp> reduces;
    SmallVector<ReshapeOp> reshapes;
    SmallVector<ScanOp> scans;
    SmallVector<CatOp> cats;
    SmallVector<::torchsim::VcixOp> vcixes;
    mod.walk([&](Operation *op) {
      if (auto v = dyn_cast<::torchsim::VcixOp>(op))
        vcixes.push_back(v);
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
    for (::torchsim::VcixOp v : vcixes)
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
      anchorReduce(r, lanes);
    for (ScanOp sc : scans)
      anchorScan(sc, lanes);
  }
};

class TritonGPUTorchsimStripPinsPass
    : public impl::TritonGPUTorchsimStripPinsBase<TritonGPUTorchsimStripPinsPass> {
public:
  void runOnOperation() override {
    SmallVector<GatherOp> pins;
    getOperation().walk([&](GatherOp g) {
      if (g->hasAttr("torchsim.pin"))
        pins.push_back(g);
    });
    for (GatherOp g : pins) {
      if (g.getSrc().getType() != g.getType()) {
        g.emitError("a pin that changes its type is not a pin");
        return signalPassFailure();
      }
      g.getResult().replaceAllUsesWith(g.getSrc());
      g->erase();
    }
  }
};

}
}
}
