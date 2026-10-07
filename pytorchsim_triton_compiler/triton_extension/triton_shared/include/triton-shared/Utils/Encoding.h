#ifndef TRITON_SHARED_UTILS_ENCODING_H
#define TRITON_SHARED_UTILS_ENCODING_H

// The layout encoding a value this library makes stands in, derived from the
// encoding of the value it stands for. One place, so every analysis answers alike.

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace triton {

// `enc` with the axes `dropDims` sliced away: what a piece of a tile in `enc`
// carries once those axes are gone. Nothing dropped is `enc` itself; null where
// `enc` is no distributed layout and something has to go.
inline Attribute sliceEncoding(Attribute enc, ArrayRef<int64_t> dropDims) {
  if (!enc || dropDims.empty())
    return enc;
  auto cur = dyn_cast<gpu::DistributedEncodingTrait>(enc);
  if (!cur)
    return nullptr;
  SmallVector<int64_t> sorted(dropDims.begin(), dropDims.end());
  llvm::sort(sorted);
  for (int64_t d : llvm::reverse(sorted))
    cur = cast<gpu::DistributedEncodingTrait>(
        gpu::SliceEncodingAttr::get(enc.getContext(), d, cur));
  return cur;
}

// The 1-D range along `axis` of a rank-`rank` tile in `enc`: every other axis sliced away.
inline Attribute axisEncoding(Attribute enc, int64_t rank, int64_t axis) {
  SmallVector<int64_t> others;
  for (int64_t d = 0; d < rank; d++)
    if (d != axis)
      others.push_back(d);
  return sliceEncoding(enc, others);
}

// The encoding of a `srcShape` tile in `srcEnc` reshaped to `dstShape`, as the
// dialect that owns the encoding infers it. Null where there is none or no answer.
inline Attribute reshapeEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                                 ArrayRef<int64_t> dstShape) {
  if (!srcEnc || srcShape == dstShape)
    return srcEnc;
  auto *infer =
      srcEnc.getDialect().getRegisteredInterface<DialectInferLayoutInterface>();
  Attribute dstEnc;
  if (!infer || failed(infer->inferReshapeOpEncoding(
                    srcShape, srcEnc, dstShape, dstEnc, /*allowReorder=*/false,
                    std::nullopt)))
    return nullptr;
  return dstEnc;
}

// The encoding a `shape` value takes when it replaces a value of type `replaced`:
// the same layout, reshaped when a split made the shape differ. Null where
// `replaced` carries none or the two do not hold the same elements.
inline Attribute encodingReplacing(Type replaced, ArrayRef<int64_t> shape) {
  auto t = dyn_cast_or_null<RankedTensorType>(replaced);
  if (!t || !t.getEncoding())
    return nullptr;
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  if (t.getNumElements() != n)
    return nullptr;
  return reshapeEncoding(t.getShape(), t.getEncoding(), shape);
}

} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_UTILS_ENCODING_H
