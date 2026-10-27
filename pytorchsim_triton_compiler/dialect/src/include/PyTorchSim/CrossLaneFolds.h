#ifndef PYTORCHSIM_CROSS_LANE_FOLDS_H
#define PYTORCHSIM_CROSS_LANE_FOLDS_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"

namespace pytorchsim {
/// One combiner a target's cross-lane unit folds a reduce by: its op class (and that class's name), the
/// fold the unit makes, and whether the combiner drops a NaN where the unit's fold keeps one.
struct CrossLaneFold {
  mlir::TypeID (*combiner)();
  llvm::StringRef (*name)();
  llvm::StringRef fold;
  bool dropsNaN;
};

/// What a target's cross-lane unit folds: these combiners over elements of `element`'s type. A target
/// states it once, in C++ its CMakeLists.txt names (FOLDS); the anchor and the target's lowering read it.
struct CrossLaneFolds {
  llvm::ArrayRef<CrossLaneFold> folds;
  mlir::Type (*element)(mlir::MLIRContext *) = nullptr;
};

/// The folds of the target whose dialect is `dialect` (none for a target with no FOLDS); failure when no
/// registered target has that dialect.
mlir::FailureOr<CrossLaneFolds> crossLaneFoldsOf(llvm::StringRef dialect);

/// The fold `combiner` makes over elements of `elem` in `folds`, or null when the unit takes neither.
const CrossLaneFold *foldOf(const CrossLaneFolds &folds, mlir::Operation *combiner, mlir::Type elem);
}

#endif
