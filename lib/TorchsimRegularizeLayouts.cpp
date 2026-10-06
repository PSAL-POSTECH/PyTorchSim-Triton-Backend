// tritongpu-torchsim-regularize-layouts: below TritonGPU, every encoding spelled as the LinearLayout it
// is and unit extents folded with their layouts renamed, so the passes below read bases and
// nothing else.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DECL_TRITONGPUTORCHSIMREGULARIZELAYOUTS
#define GEN_PASS_DEF_TRITONGPUTORCHSIMREGULARIZELAYOUTS
#include "TorchsimExt/Passes.h.inc"

// A tensor type with its distributed encoding spelled as the linear layout it is on its own
// shape -- a linear one written for a larger tile included -- less the register bases that step
// nothing (one lane holding the same element twice). A shape that is no power of two -- a window
// cut from a tile, a padded scan buffer -- is laid out as its power-of-two hull is, and uses part
// of it.
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

// `t`'s linear layout collapsed by `groups` into `shape`: a group's coordinate is the row-major
// number of its members' -- the elements stay where they are, only their names change.
static Type collapsedType(RankedTensorType t, ArrayRef<int64_t> shape,
                          ArrayRef<ReassociationIndices> groups) {
  auto enc = dyn_cast_or_null<LinearEncodingAttr>(t.getEncoding());
  if (!enc)
    return RankedTensorType::get(shape, t.getElementType());
  MLIRContext *ctx = t.getContext();
  const LinearLayout &ll = enc.getLinearLayout();
  auto names = standardOutDimNames(ctx, t.getRank());
  auto newNames = standardOutDimNames(ctx, shape.size());
  LinearLayout::BasesT bases;
  for (const auto &[in, vecs] : ll.getBases()) {
    auto &out = bases[in];
    for (const std::vector<int32_t> &b : vecs) {
      std::vector<int32_t> nb;
      for (const ReassociationIndices &g : groups) {
        int32_t v = 0;
        for (int64_t d : g)
          v = v * ll.getOutDimSize(names[d]) + b[d];
        nb.push_back(v);
      }
      out.push_back(nb);
    }
  }
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  for (auto [i, g] : llvm::enumerate(groups)) {
    int32_t n = 1;
    for (int64_t d : g)
      n *= ll.getOutDimSize(names[d]);
    outDims.push_back({newNames[i], n});
  }
  return RankedTensorType::get(
      shape, t.getElementType(),
      LinearEncodingAttr::get(ctx, LinearLayout(std::move(bases), outDims, ll.isSurjective())));
}

// Upstream's fold of unit extents, its two reshapes carrying the layout through: a collapsed
// operand's layout is its source's renamed, a result goes back to its own type. Never to rank 0
// -- a rank-0 tensor has no layout to carry.
static void foldUnitExtents(ModuleOp mod) {
  linalg::ControlDropUnitDims control;
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
  (void)applyPatternsGreedily(mod, std::move(patterns));
}

class TritonGPUTorchsimRegularizeLayoutsPass
    : public impl::TritonGPUTorchsimRegularizeLayoutsBase<
          TritonGPUTorchsimRegularizeLayoutsPass> {
  // A shape change renames coordinates and moves nothing, so an end of one that carries no
  // layout has its other end's, composed through it (or back): to a fixpoint.
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

} // namespace gpu
} // namespace triton
} // namespace mlir
