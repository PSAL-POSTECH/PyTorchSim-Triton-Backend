// The layout facts the passes in Python take from Triton's own LinearLayout code (lib/Tools, built into
// this extension against the same MLIR) rather than restate: a reshape's bases (reshapeLayout).
#include "PyTorchSim/LinearLayouts.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using mlir::triton::LinearLayout;

/// The context the dim names live in: LinearLayout names its dims by StringAttr, and loads no dialect.
static MLIRContext &namesContext() {
  static MLIRContext *ctx = new MLIRContext(MLIRContext::Threading::DISABLED);
  return *ctx;
}

/// Why `bases` over `src` is no LinearLayout, or empty: the checks LinearLayout's constructor would end
/// the process on, and Triton's power-of-two out-dims.
static std::string refusal(const pytorchsim::NamedBases &bases, const std::vector<int64_t> &src,
                           const std::vector<int64_t> &dst) {
  auto pow2 = [](const std::vector<int64_t> &s) {
    return llvm::all_of(s, [](int64_t n) { return n > 0 && llvm::isPowerOf2_64(n); });
  };
  if (!pow2(src) || !pow2(dst))
    return "a shape that is not all powers of two: LinearLayout has none other";
  int64_t a = 1, b = 1;
  for (int64_t n : src)
    a *= n;
  for (int64_t n : dst)
    b *= n;
  if (a != b)
    return "shapes of different element counts";
  for (const auto &[name, vecs] : bases)
    for (const std::vector<int32_t> &v : vecs) {
      if (v.size() != src.size())
        return "a basis of another rank than the source shape";
      for (size_t d = 0; d < v.size(); ++d)
        if (v[d] < 0 || v[d] >= src[d])
          return "a basis outside the source shape";
    }
  return "";
}

pytorchsim::NamedBases pytorchsim::reshapeBases(const NamedBases &bases, const std::vector<int64_t> &src,
                                                const std::vector<int64_t> &dst, std::string &why) {
  why = refusal(bases, src, dst);
  if (!why.empty())
    return {};
  MLIRContext *ctx = &namesContext();
  std::vector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> in;
  for (const auto &[name, vecs] : bases)
    in.push_back({StringAttr::get(ctx, name), vecs});
  LinearLayout ll(in, triton::standardOutDimPairs(ctx, src), /*requireSurjective=*/false);
  LinearLayout out = triton::reshapeLayout(ctx, ll, dst);
  NamedBases got;
  for (const auto &[name, vecs] : out.getBases())
    got.push_back({name.str(), vecs});
  return got;
}
