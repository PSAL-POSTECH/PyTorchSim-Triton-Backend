// The layouts of pytorchsim.join, split and cat (ShapeOpLayouts.h).
#include "TorchsimExt/ShapeOpLayouts.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"

using namespace mlir;
using mlir::triton::LinearLayout;

namespace {

// `ll` with a new last axis of `size` (axis == its rank) or `axis` doubled, the new bit a register bit.
static LinearLayout withAxis(const LinearLayout &ll, int axis, int32_t size, MLIRContext *ctx) {
  auto bases = ll.getBases();
  auto outs = to_vector(ll.getOutDims());
  bool grows = axis == int(outs.size());
  if (grows) {
    for (auto &[dim, vecs] : bases)
      for (auto &v : vecs)
        v.push_back(0);
    outs.push_back({StringAttr::get(ctx, "dim" + std::to_string(axis)), size});
  }
  std::vector<int32_t> bit(outs.size(), 0);
  bit[axis] = grows ? 1 : outs[axis].second;
  bases[StringAttr::get(ctx, "register")].push_back(bit);
  if (!grows)
    outs[axis].second *= 2;
  return LinearLayout(std::move(bases), outs, /*requireSurjective=*/false);
}

// `ll` with its last axis dropped (half 1) or `axis` halved to `half`, that bit gone from every basis.
static LinearLayout withoutAxis(const LinearLayout &ll, int axis, int32_t half) {
  auto bases = ll.getBases();
  auto outs = to_vector(ll.getOutDims());
  bool drops = axis == int(outs.size()) - 1 && half == 1;
  for (auto &[dim, vecs] : bases)
    for (auto &v : vecs) {
      if (drops)
        v.pop_back();
      else
        v[axis] &= half - 1;
    }
  if (drops)
    outs.pop_back();
  else
    outs[axis].second = half;
  return LinearLayout(std::move(bases), outs, /*requireSurjective=*/false);
}

} // namespace

namespace torchsim {

Attribute partLayout(TsOpKind kind, RankedTensorType whole) {
  LinearLayout ll = triton::gpu::toLinearLayout(whole);
  int rank = whole.getRank();
  LinearLayout part = kind == TS_CAT ? withoutAxis(ll, 0, int32_t(whole.getShape()[0] / 2))
                                     : withoutAxis(ll, rank - 1, 1);
  return triton::gpu::inferEncodingFromLinearLayout(whole.getContext(), std::move(part),
                                                    whole.getEncoding());
}

Attribute withReducedAxis(RankedTensorType result, int axis, int64_t extent) {
  if (!llvm::isPowerOf2_64(extent))
    return {};
  MLIRContext *ctx = result.getContext();
  LinearLayout ll = triton::gpu::toLinearLayout(result);
  auto bases = ll.getBases();
  for (auto &[dim, vecs] : bases)
    for (auto &v : vecs)
      v.insert(v.begin() + axis, 0);
  auto &regs = bases[StringAttr::get(ctx, "register")];
  for (int64_t step = 1; step < extent; step *= 2) {
    std::vector<int32_t> bit(result.getRank() + 1, 0);
    bit[axis] = int32_t(step);
    regs.push_back(bit);
  }
  SmallVector<std::pair<StringAttr, int32_t>> outs;
  for (int d = 0, r = 0; d <= result.getRank(); ++d)
    outs.push_back({StringAttr::get(ctx, "dim" + std::to_string(d)),
                    d == axis ? int32_t(extent) : int32_t(result.getShape()[r++])});
  return triton::gpu::inferEncodingFromLinearLayout(
      ctx, LinearLayout(std::move(bases), outs, /*requireSurjective=*/false), result.getEncoding());
}

Attribute wholeLayout(TsOpKind kind, RankedTensorType part) {
  LinearLayout ll = triton::gpu::toLinearLayout(part);
  LinearLayout whole = kind == TS_CAT ? withAxis(ll, 0, 0, part.getContext())
                                      : withAxis(ll, part.getRank(), 2, part.getContext());
  return triton::gpu::inferEncodingFromLinearLayout(part.getContext(), std::move(whole),
                                                    part.getEncoding());
}

} // namespace torchsim
