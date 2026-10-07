// tritongpu-torchsim-anchor-layouts: every layout this lane machine fixes, written into TritonGPU
// IR in one place before the one remove-layout-conversions -- a dot's operands and result, a
// reduce's input off the reduced axis, a long scan's input in rows of lanes, a reorder-allowed
// reshape's or a cat's result where its elements are.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "TorchSim/TorchSimDialect.h"
#include "TorchSim/TorchSimOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

// A blocked layout with `spt` elements per lane and the lanes all on `axis`.
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

// The lanes on `axis`, one element each.
Attribute laneLayout(MLIRContext *ctx, unsigned rank, unsigned axis, unsigned lanes) {
  return blocked(ctx, SmallVector<unsigned>(rank, 1), axis, lanes);
}

// The whole tile in every lane's registers, the lanes along `axis` holding copies: a reduce
// with no kept axis wider than one runs inside the lane this way.
Attribute bankLayout(MLIRContext *ctx, ArrayRef<int64_t> shape, unsigned axis, unsigned lanes) {
  SmallVector<unsigned> spt(shape.begin(), shape.end());
  return blocked(ctx, spt, axis, lanes);
}

// The dims a tile's lane bases step, units left out (lanes over a unit dim hold copies).
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

// `v` through an identity gather along its last axis in `enc`, flagged efficient_layout: an
// anchor remove-layout-conversions keeps, which torchsim-strip-pins takes out after it.
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

// `op`'s result `i` takes `enc`; every reader keeps the old type through a conversion back.
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

// `t`'s lanes on `axis`, one element each, as the linear layout of its shape; where the axis is
// longer than the lanes, its bits past them are the slowest registers -- the axis wraps onto the
// bank's outermost depth.
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

// All zeros, as tt.dot's C is read: a splat of a zero constant, or a dense splat zero, through
// any layout conversion.
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

// A, B and C each ride their last axis, each in the linear layout of its own shape -- the
// systolic array's operands, which no GPU dot encoding states. So the product is torchsim.dot,
// this machine's op; readers keep the old type through a conversion back.
void anchorDot(DotOp dot, unsigned lanes) {
  OpBuilder b(dot);
  Location loc = dot.getLoc();
  auto laid = [&](Value v) {
    auto t = cast<RankedTensorType>(v.getType());
    return convertTo(b, loc, v, laneLayoutOf(t, t.getRank() - 1, lanes));
  };
  Value a = laid(dot.getA()), bv = laid(dot.getB());
  // A zero C is the product's own zero tile, filled right here in its layout -- not a constant,
  // which canonicalization hoists and readers share: the accumulator stays the dot's alone.
  Value c;
  auto ct = cast<RankedTensorType>(dot.getC().getType());
  if (isZeroSplat(dot.getC())) {
    auto t = ct.cloneWithEncoding(laneLayoutOf(ct, ct.getRank() - 1, lanes));
    Type elt = t.getElementType();
    Value zero = arith::ConstantOp::create(b, loc, elt, cast<TypedAttr>(b.getZeroAttr(elt)));
    Value empty = tensor::EmptyOp::create(b, loc, t, ValueRange{});
    c = linalg::FillOp::create(b, loc, ValueRange{zero}, ValueRange{empty}).result();
  } else {
    c = laid(dot.getC());
  }
  auto product = ::torchsim::DotOp::create(b, loc, c.getType(), a, bv, c, /*indexing_maps=*/nullptr);
  auto back = ConvertLayoutOp::create(b, loc, dot.getType(), product.getResult());
  dot.replaceAllUsesWith(back.getResult());
  dot.erase();
}

// The input where it stands when its lanes avoid the reduced axis, else the first kept axis
// wider than one, else one bank: no reduce below crosses the lanes.
void anchorReduce(ReduceOp red, unsigned lanes) {
  auto t = cast<RankedTensorType>(red.getOperand(0).getType());
  int rank = t.getRank(), axis = red.getAxis();
  auto shape = t.getShape();
  Attribute enc = t.getEncoding();
  SmallVector<int> on = laneDims(t);
  if (on.empty() || llvm::is_contained(on, axis)) {
    int wide = -1;
    for (int d = 0; d < rank && wide < 0; ++d)
      if (d != axis && shape[d] != 1)
        wide = d;
    if (wide >= 0) {
      enc = laneLayout(red.getContext(), rank, wide, lanes);
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

// A scan longer than the lanes along an axis the lanes ride: element i in lane i / lanes,
// register i % lanes -- so the rows of lanes length it is folded into below each sit in one
// lane, scanned in the bank, the fold itself a renaming. Anything else scans where it stands.
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

// A reshape that may reorder takes the layout its elements already have, renamed: below it is
// a row-major reshape, and any other layout would be a move nobody wrote.
void anchorReorderReshape(ReshapeOp op) {
  if (!op.getAllowReorder())
    return;
  auto src = op.getSrc().getType();
  auto dst = op.getType();
  auto *infer = dyn_cast<DialectInferLayoutInterface>(&src.getEncoding().getDialect());
  Attribute inPlace;
  if (!infer || failed(infer->inferReshapeOpEncoding(src.getShape(), src.getEncoding(),
                                                     dst.getShape(), inPlace, op.getLoc())))
    return;
  retypeResult(op, 0, dst.cloneWithEncoding(inPlace));
}

// A cat's result where its elements are as triton_shared writes it -- lhs then rhs along dim 0 --
// each half in its operands' layout, the half one more register, the slowest.
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
                                                       /*requireSurjective=*/true));
  // Pinned: nothing constrains a cat's result, so remove-layout-conversions would move it.
  Value r = op.getResult();
  Type old = r.getType();
  r.setType(op.getType().cloneWithEncoding(enc));
  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  Value pinned = pin(b, op.getLoc(), r, enc);
  Value back = convertTo(b, op.getLoc(), pinned, cast<RankedTensorType>(old).getEncoding());
  r.replaceAllUsesExcept(back, pinned.getDefiningOp());
}

// A loop carrying a rank-3 tile with a tt.trans in its body: remove-layout-conversions does not
// terminate on it.
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

} // namespace

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
    mod.walk([&](Operation *op) {
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
    for (DotOp d : dots)
      anchorDot(d, lanes);
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

// After remove-layout-conversions the pins have done their work: each is its source again.
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

} // namespace gpu
} // namespace triton
} // namespace mlir
