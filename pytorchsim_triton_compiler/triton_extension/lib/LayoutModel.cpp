// The layout port's C++ end (LayoutModel.h): the table a pack registers, and each question put to it
// with the layouts flattened to linear-layout bases.
#include "TorchsimExt/LayoutModel.h"

#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

#include <vector>

using namespace mlir;

namespace {

const TsLayoutModel *registered = nullptr;

// A tensor type's layout as the port spells it, with the storage its pointers point into.
struct FlatLayout {
  std::vector<int64_t> shape;
  std::vector<int32_t> reg, lane;
  TsLayout view;

  explicit FlatLayout(RankedTensorType t) : shape(t.getShape()) {
    triton::LinearLayout ll = triton::gpu::toLinearLayout(t);
    MLIRContext *ctx = t.getContext();
    int32_t nReg = flatten(ll, StringAttr::get(ctx, "register"), reg);
    int32_t nLane = flatten(ll, StringAttr::get(ctx, "lane"), lane);
    view = {int32_t(shape.size()), shape.data(), elemBits(t.getElementType()), nReg, nLane, reg.data(),
            lane.data()};
  }

  // An element's bits; a pointer an address of 64.
  static int32_t elemBits(Type t) { return t.isIntOrFloat() ? int32_t(t.getIntOrFloatBitWidth()) : 64; }

  static int32_t flatten(const triton::LinearLayout &ll, StringAttr dim, std::vector<int32_t> &out) {
    if (!ll.hasInDim(dim))
      return 0;
    const auto &bases = ll.getBases().lookup(dim);
    for (const auto &b : bases)
      out.insert(out.end(), b.begin(), b.end());
    return int32_t(bases.size());
  }
};

} // namespace

extern "C" __attribute__((visibility("default"))) void
torchsim_set_layout_model(const TsLayoutModel *model) {
  registered = model;
}

namespace torchsim {

const TsLayoutModel *layoutModel() {
  return registered && registered->version == TS_LAYOUT_MODEL_VERSION ? registered : nullptr;
}

bool layoutFree(TsOpKind op) {
  const TsLayoutModel *m = layoutModel();
  return m && m->layout_free && m->layout_free(op) > 0;
}

bool laneFolds(llvm::StringRef combiner, int32_t elemBits) {
  const TsLayoutModel *m = layoutModel();
  if (!m || !m->lane_folds)
    return true;
  std::string name = combiner.str();
  return m->lane_folds(name.c_str(), elemBits) != 0;
}

bool layoutLegal(TsOpKind op, RankedTensorType operand, RankedTensorType result) {
  const TsLayoutModel *m = layoutModel();
  if (!m || !m->layout_legal)
    return true;
  FlatLayout from(operand), to(result);
  return m->layout_legal(op, &from.view, &to.view) != 0;
}

std::optional<double> moveCost(RankedTensorType from, RankedTensorType to) {
  const TsLayoutModel *m = layoutModel();
  if (!m || !m->convert_cost)
    return std::nullopt;
  FlatLayout a(from), b(to);
  double cost = m->convert_cost(&a.view, &b.view);
  return cost < 0 ? std::nullopt : std::optional<double>(cost);
}

std::optional<double> convertCost(RankedTensorType src, Attribute dstEnc) {
  return moveCost(src, src.cloneWithEncoding(dstEnc));
}

std::optional<double> rematCost(TsOpKind kind, RankedTensorType at) {
  const TsLayoutModel *m = layoutModel();
  if (!m || !m->remat_cost)
    return std::nullopt;
  FlatLayout flat(at);
  double cost = m->remat_cost(kind, &flat.view);
  return cost < 0 ? std::nullopt : std::optional<double>(cost);
}

} // namespace torchsim
