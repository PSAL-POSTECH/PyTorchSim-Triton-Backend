// tritongpu-pytorchsim-regularize-layouts: below TritonGPU, every encoding spelled as the
// LinearLayout it is and unit extents folded with their layouts renamed, so the passes below read
// bases and nothing else.
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "PyTorchSimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DECL_TRITONGPUPYTORCHSIMREGULARIZELAYOUTS
#define GEN_PASS_DEF_TRITONGPUPYTORCHSIMREGULARIZELAYOUTS
#include "PyTorchSimExt/Passes.h.inc"

static Type linearized(Type t) {
  auto ty = dyn_cast<RankedTensorType>(t);
  if (!ty || !isa_and_nonnull<DistributedEncodingTrait>(ty.getEncoding()))
    return t;
  auto hull = llvm::map_to_vector(ty.getShape(), [](int64_t n) {
    return int64_t(llvm::PowerOf2Ceil(uint64_t(std::max<int64_t>(n, 1))));
  });
  LinearLayout ll = toLinearLayout(hull, ty.getEncoding());
  auto bases = ll.getBases();
  llvm::erase_if(bases[StringAttr::get(ty.getContext(), "register")],
                 [](const std::vector<int32_t> &b) { return llvm::all_of(b, [](int32_t x) { return x == 0; }); });
  auto enc = LinearEncodingAttr::get(
      ty.getContext(), LinearLayout(std::move(bases), ll.getOutDims(), ll.isSurjective()));
  return enc == ty.getEncoding() ? t : ty.cloneWithEncoding(enc);
}

static Type collapsedType(RankedTensorType t, ArrayRef<int64_t> shape,
                          ArrayRef<ReassociationIndices> groups) {
  auto enc = dyn_cast_or_null<LinearEncodingAttr>(t.getEncoding());
  if (!enc)
    return RankedTensorType::get(shape, t.getElementType());
  MLIRContext *ctx = t.getContext();
  const LinearLayout &ll = enc.getLinearLayout();
  auto names = standardOutDimNames(ctx, t.getRank());
  SmallVector<int64_t> sizes;
  for (const ReassociationIndices &g : groups) {
    int64_t n = 1;
    for (int64_t d : g)
      n *= ll.getOutDimSize(names[d]);
    sizes.push_back(n);
  }
  return RankedTensorType::get(shape, t.getElementType(),
                               LinearEncodingAttr::get(ctx, reshapeLayout(ctx, ll, sizes)));
}

// Each dim of `dims` not in `gone`, renumbered as if the dims of `gone` were not there.
static SmallVector<int64_t> renumbered(ArrayRef<int64_t> dims, ArrayRef<int64_t> gone) {
  SmallVector<int64_t> out;
  for (int64_t d : dims)
    if (!llvm::is_contained(gone, d))
      out.push_back(d - llvm::count_if(gone, [&](int64_t g) { return g < d; }));
  return out;
}

// A named linalg op rebuilt as itself over the collapsed operands, its dimensions or permutation renumbered:
// a03 reads those attributes, so the op stays named and is not made a linalg.generic.
static IndexingMapOpInterface rebuildNamed(Location loc, OpBuilder &b, IndexingMapOpInterface op,
                                           ArrayRef<Value> operands, ArrayRef<AffineMap>,
                                           const llvm::SmallDenseSet<unsigned> &dropped) {
  auto dps = cast<DestinationStyleOpInterface>(op.getOperation());
  NamedAttrList attrs(op->getAttrs());
  if (auto red = dyn_cast<linalg::ReduceOp>(op.getOperation())) {
    SmallVector<int64_t> gone(dropped.begin(), dropped.end());
    attrs.set("dimensions", b.getDenseI64ArrayAttr(renumbered(red.getDimensions(), gone)));
  }
  if (auto tr = dyn_cast<linalg::TransposeOp>(op.getOperation())) {
    auto gone = llvm::map_to_vector(dropped, [&](unsigned i) { return tr.getPermutation()[i]; });
    attrs.set("permutation", b.getDenseI64ArrayAttr(renumbered(tr.getPermutation(), gone)));
  }
  OperationState state(loc, op->getName(), operands,
                       TypeRange(ValueRange(operands.take_back(dps.getNumDpsInits()))), attrs);
  IRMapping map;
  op->getRegion(0).cloneInto(state.addRegion(), map);
  return cast<IndexingMapOpInterface>(b.create(state));
}

// linalg.reduce, linalg.transpose and linalg.fill drop their unit loops as linalg.generic does.
template <typename OpTy> struct DropNamedUnitDims : OpRewritePattern<OpTy> {
  DropNamedUnitDims(MLIRContext *ctx, const linalg::ControlDropUnitDims &control)
      : OpRewritePattern<OpTy>(ctx), control(control) {}
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rw) const override {
    auto r = linalg::dropUnitDims(rw, cast<IndexingMapOpInterface>(op.getOperation()),
                                  rebuildNamed, control);
    if (failed(r))
      return failure();
    rw.replaceOp(op, r->replacements);
    return success();
  }
  linalg::ControlDropUnitDims control;
};

// Whether `t` carries a linear encoding of its own rank.
static bool isLaid(Type t) {
  auto ty = cast<RankedTensorType>(t);
  auto enc = dyn_cast_or_null<LinearEncodingAttr>(ty.getEncoding());
  return enc && enc.getLinearLayout().getNumOutDims() == ty.getRank();
}

// A tensor whose linear encoding is of another rank: that encoding collapsed onto the tensor's shape, when
// its out dims group into that shape; else none.
static std::optional<Type> reranked(Type t) {
  auto ty = dyn_cast<RankedTensorType>(t);
  auto enc = ty ? dyn_cast_or_null<LinearEncodingAttr>(ty.getEncoding()) : nullptr;
  if (!enc || isLaid(ty))
    return std::nullopt;
  auto sizes = llvm::to_vector(llvm::map_range(enc.getLinearLayout().getOutDimSizes(),
                                               [](int32_t n) { return int64_t(n); }));
  auto hull = llvm::map_to_vector(ty.getShape(), [](int64_t n) {
    return int64_t(llvm::PowerOf2Ceil(uint64_t(std::max<int64_t>(n, 1))));
  });
  auto groups = getReassociationIndicesForCollapse(sizes, hull);
  if (!groups)
    return std::nullopt;
  return collapsedType(RankedTensorType::get(sizes, ty.getElementType(), enc), ty.getShape(), *groups);
}

// The canonical set's rank-reducing rewrites infer a reshape's or cast's new side bare, or keep the old
// rank's encoding on it; that side takes the other side's layout (collapsed). Returns whether any did.
static bool giveRankReducedLayouts(ModuleOp mod) {
  bool given = false;
  auto give = [&](Value bare, Type laid) {
    bare.setType(laid);
    given = true;
  };
  mod.walk([&](Operation *op) {
    for (Value r : op->getResults())
      if (auto t = reranked(r.getType()))
        give(r, *t);
    if (auto e = dyn_cast<tensor::ExpandShapeOp>(op))
      if (!isLaid(e.getSrcType()) && isLaid(e.getResultType()))
        give(e.getSrc(), collapsedType(e.getResultType(), e.getSrcType().getShape(),
                                       e.getReassociationIndices()));
    if (auto c = dyn_cast<tensor::CollapseShapeOp>(op))
      if (!isLaid(c.getResultType()) && isLaid(c.getSrcType()))
        give(c.getResult(), collapsedType(c.getSrcType(), c.getResultType().getShape(),
                                          c.getReassociationIndices()));
    if (auto c = dyn_cast<tensor::CastOp>(op)) {
      if (!isa<RankedTensorType>(c.getSource().getType()) || !isa<RankedTensorType>(c.getType()))
        return;
      if (!isLaid(c.getSource().getType()) && isLaid(c.getType()))
        give(c.getSource(), c.getType());
      else if (!isLaid(c.getType()) && isLaid(c.getSource().getType()))
        give(c.getResult(), c.getSource().getType());
    }
  });
  return given;
}

// A reshape of a static tensor.empty is a tensor.empty of the reshape's type, its layout kept.
template <typename OpTy> struct ReshapedEmpty : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rw) const override {
    auto e = op.getSrc().template getDefiningOp<tensor::EmptyOp>();
    auto t = cast<RankedTensorType>(op.getType());
    if (!e || !t.hasStaticShape())
      return failure();
    rw.replaceOpWithNewOp<tensor::EmptyOp>(op, t.getShape(), t.getElementType(), t.getEncoding());
    return success();
  }
};

// The unit dims a tensor of pointers can drop with its loads and stores: size 1 at a static offset 0 (so
// their one index adds nothing to an address, wrapped or not), on a non-block pointer, inside every user's
// mask box (a static extent of at least 1 from a static low of 0). None when that would leave rank 0.
static SmallVector<int64_t> transferUnits(tts::MakeTensorPtrOp op) {
  ArrayRef<int64_t> sizes = op.getSizes();
  if (!op.getOrder().empty() || !isLaid(op.getType()))
    return {};
  auto inBox = [](SmallVector<OpFoldResult> dims, SmallVector<OpFoldResult> lows, int64_t d) {
    if (dims.empty())
      return true;
    auto n = getConstantIntValue(dims[d]), lo = getConstantIntValue(lows[d]);
    return n && lo && *n >= 1 && *lo == 0;
  };
  SmallVector<int64_t> units;
  for (auto [d, n] : llvm::enumerate(sizes)) {
    if (n != 1 || op.getStaticOffsets()[d] != 0)
      continue;
    bool kept = llvm::all_of(op->getUses(), [&](OpOperand &u) {
      if (auto l = dyn_cast<tts::LoadOp>(u.getOwner()))
        return inBox(l.getMixedMaskDims(), l.getMixedMaskLows(), d);
      if (auto s = dyn_cast<tts::StoreOp>(u.getOwner()))
        return u.getOperandNumber() == 0 && inBox(s.getMixedMaskDims(), s.getMixedMaskLows(), d);
      return false;
    });
    if (kept)
      units.push_back(d);
  }
  if (units.size() == sizes.size())
    return {};
  return units;
}

// `all` without the entries of `units`.
template <typename T> static SmallVector<T> withoutDims(ArrayRef<T> all, ArrayRef<int64_t> units) {
  SmallVector<T> out;
  for (auto [d, x] : llvm::enumerate(all))
    if (!llvm::is_contained(units, int64_t(d)))
      out.push_back(x);
  return out;
}

// The pointer op again over the kept dims: each dropped dim's stride, wrap and step go with its one index.
static Value rebuiltPtr(PatternRewriter &rw, tts::MakeTensorPtrOp op, ArrayRef<int64_t> units, Type t) {
  auto ptr = tts::MakeTensorPtrOp::create(
      rw, op.getLoc(), op.getBase(), withoutDims(op.getSizes(), units),
      withoutDims<OpFoldResult>(op.getMixedStrides(), units),
      withoutDims<OpFoldResult>(op.getMixedOffsets(), units),
      withoutDims<OpFoldResult>(op.getMixedShape(), units), {}, op.getBaseOffset(), op.getFlatShape(),
      cast<RankedTensorType>(t).getEncoding());
  if (op.getStaticStep())
    ptr.setStaticStepAttr(rw.getDenseI64ArrayAttr(withoutDims(*op.getStaticStep(), units)));
  return ptr;
}

// A tensor of pointers, its loads and its stores rebuilt without the unit dims transferUnits gives: the same
// addresses in the same order, the loaded and stored tiles collapsed and expanded around them.
struct DropTransferUnitDims : OpRewritePattern<tts::MakeTensorPtrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tts::MakeTensorPtrOp op, PatternRewriter &rw) const override {
    SmallVector<int64_t> units = transferUnits(op);
    if (units.empty())
      return failure();
    auto ptrTy = cast<RankedTensorType>(op.getType());
    SmallVector<int64_t> shape = withoutDims(op.getSizes(), units);
    auto groups = *getReassociationIndicesForCollapse(ptrTy.getShape(), shape);
    auto narrow = [&](Type t) { return collapsedType(cast<RankedTensorType>(t), shape, groups); };
    Value ptr = rebuiltPtr(rw, op, units, narrow(ptrTy));
    auto box = [&](auto x) {
      SmallVector<OpFoldResult> dims, lows;
      if (x.hasMask()) {
        dims = withoutDims<OpFoldResult>(x.getMixedMaskDims(), units);
        lows = withoutDims<OpFoldResult>(x.getMixedMaskLows(), units);
      }
      return std::make_pair(dims, lows);
    };
    auto collapsed = [&](Value v) -> Value {
      if (!v)
        return v;
      return tensor::CollapseShapeOp::create(rw, op.getLoc(), narrow(v.getType()), v, groups);
    };
    for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      rw.setInsertionPoint(user);
      if (auto l = dyn_cast<tts::LoadOp>(user)) {
        auto [dims, lows] = box(l);
        Type t = narrow(l.getType());
        auto n = tts::LoadOp::create(rw, l.getLoc(), ptr, dims, lows, l.getOther(),
                                     collapsed(l.getElemMask()), cast<RankedTensorType>(t).getEncoding());
        rw.replaceOpWithNewOp<tensor::ExpandShapeOp>(l, l.getType(), n, groups);
      } else {
        auto s = cast<tts::StoreOp>(user);
        auto [dims, lows] = box(s);
        tts::StoreOp::create(rw, s.getLoc(), ptr, collapsed(s.getValue()), dims, lows,
                             collapsed(s.getElemMask()));
        rw.eraseOp(s);
      }
    }
    rw.eraseOp(op);
    return success();
  }
};

static void foldUnitExtents(ModuleOp mod) {
  linalg::ControlDropUnitDims control;
  control.controlFn = [def = control.controlFn](Operation *op) {
    if (isa<linalg::ReduceOp, linalg::TransposeOp, linalg::FillOp>(op))
      return llvm::to_vector(llvm::seq<unsigned>(0, cast<linalg::LinalgOp>(op).getNumLoops()));
    return def(op);
  };
  control.collapseFn = [](RewriterBase &rw, Location loc, Value v, ArrayRef<int64_t> shape,
                          ArrayRef<ReassociationIndices> groups,
                          const linalg::ControlDropUnitDims &) -> FailureOr<Value> {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || shape.empty())
      return failure();
    return tensor::CollapseShapeOp::create(rw, loc, collapsedType(t, shape, groups), v, groups)
        .getResult();
  };
  control.expandFn = [](RewriterBase &rw, Location loc, Value r, Value dest,
                        ArrayRef<ReassociationIndices> groups,
                        const linalg::ControlDropUnitDims &) -> FailureOr<Value> {
    auto t = dyn_cast<RankedTensorType>(dest.getType());
    if (!t)
      return failure();
    return tensor::ExpandShapeOp::create(rw, loc, t, r, groups).getResult();
  };
  RewritePatternSet patterns(mod.getContext());
  linalg::populateFoldUnitExtentDimsPatterns(patterns, control);
  patterns.add<DropNamedUnitDims<linalg::ReduceOp>, DropNamedUnitDims<linalg::TransposeOp>,
               DropNamedUnitDims<linalg::FillOp>>(mod.getContext(), control);
  patterns.add<DropTransferUnitDims>(mod.getContext());
  (void)applyPatternsGreedily(mod, std::move(patterns));
  RewritePatternSet canonical(mod.getContext());
  linalg::populateFoldUnitExtentDimsCanonicalizationPatterns(canonical, control);
  canonical.add<ReshapedEmpty<tensor::CollapseShapeOp>, ReshapedEmpty<tensor::ExpandShapeOp>>(
      mod.getContext());
  FrozenRewritePatternSet frozen(std::move(canonical));
  do
    (void)applyPatternsGreedily(mod, frozen);
  while (giveRankReducedLayouts(mod));
}

class TritonGPUPyTorchSimRegularizeLayoutsPass
    : public impl::TritonGPUPyTorchSimRegularizeLayoutsBase<
          TritonGPUPyTorchSimRegularizeLayoutsPass> {

  void fillComposedLayouts() {
    MLIRContext *ctx = &getContext();
    auto layoutOf = [](RankedTensorType t) -> std::optional<LinearLayout> {
      if (!isa_and_nonnull<DistributedEncodingTrait>(t.getEncoding()) ||
          llvm::any_of(t.getShape(), [](int64_t n) { return !llvm::isPowerOf2_64(n); }))
        return std::nullopt;
      return toLinearLayout(t);
    };
    auto give = [&](Value v, LinearLayout ll) {
      auto t = cast<RankedTensorType>(v.getType());
      v.setType(t.cloneWithEncoding(LinearEncodingAttr::get(ctx, std::move(ll))));
    };
    bool changed = true;
    while (changed) {
      changed = false;
      getOperation().walk([&](Operation *op) {
        StringRef n = op->getName().getStringRef();
        bool reshape = n == "tensor.expand_shape" || n == "tensor.collapse_shape" ||
                       n == "tensor.reshape";
        bool transpose = n == "linalg.transpose";
        if (!reshape && !transpose)
          return;
        Value a = op->getOperand(0);
        Value b = transpose ? op->getOperand(1) : op->getResult(0);
        auto ta = dyn_cast<RankedTensorType>(a.getType());
        auto tb = dyn_cast<RankedTensorType>(b.getType());
        if (!ta || !tb || (ta.getEncoding() != nullptr) == (tb.getEncoding() != nullptr))
          return;
        SmallVector<int> order;
        if (transpose)
          for (int64_t p : op->getAttrOfType<DenseI64ArrayAttr>("permutation").asArrayRef())
            order.push_back(int(p));
        if (ta.getEncoding()) {
          auto ll = layoutOf(ta);
          if (!ll)
            return;
          give(b, transpose ? transposeLinearLayout(*ll, order)
                            : reshapeLayout(ctx, *ll, tb.getShape()));
          if (transpose)
            for (Value r : op->getResults())
              r.setType(b.getType());
        } else {
          auto ll = layoutOf(tb);
          if (!ll)
            return;
          SmallVector<int> back(order.size());
          for (auto [i, p] : llvm::enumerate(order))
            back[p] = int(i);
          give(a, transpose ? transposeLinearLayout(*ll, back)
                            : reshapeLayout(ctx, *ll, ta.getShape()));
        }
        changed = true;
      });
    }
  }

public:
  void runOnOperation() override {
    fillComposedLayouts();
    getOperation().walk([](Operation *op) {
      for (Value r : op->getResults())
        r.setType(linearized(r.getType()));
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument a : block.getArguments())
            a.setType(linearized(a.getType()));
      if (auto f = dyn_cast<FunctionOpInterface>(op)) {
        auto ft = cast<FunctionType>(f.getFunctionType());
        auto in = llvm::map_to_vector(ft.getInputs(), linearized);
        auto out = llvm::map_to_vector(ft.getResults(), linearized);
        f.setFunctionTypeAttr(TypeAttr::get(FunctionType::get(op->getContext(), in, out)));
      }
    });
    foldUnitExtents(getOperation());
  }
};

}
}
}
