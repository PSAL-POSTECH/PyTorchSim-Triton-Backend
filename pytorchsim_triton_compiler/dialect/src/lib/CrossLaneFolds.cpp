#include "PyTorchSim/CrossLaneFolds.h"
#include "PyTorchSim/TargetFolds.h.inc"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

/// The target's folds by its dialect: the build's list of targets (TargetFolds.h.inc, from every target).
FailureOr<pytorchsim::CrossLaneFolds> pytorchsim::crossLaneFoldsOf(llvm::StringRef dialect) {
#define PYTORCHSIM_TARGET_FOLD(name, folds)                                                                 \
  if (dialect == name)                                                                                      \
    return folds();
  PYTORCHSIM_TARGET_FOLD_LIST
#undef PYTORCHSIM_TARGET_FOLD
  if (llvm::is_contained({PYTORCHSIM_TARGET_NAMES}, dialect))
    return CrossLaneFolds{};
  return failure();
}

const pytorchsim::CrossLaneFold *pytorchsim::foldOf(const CrossLaneFolds &folds, Operation *combiner,
                                                  Type elem) {
  if (!folds.element || elem != folds.element(elem.getContext()))
    return nullptr;
  auto it = llvm::find_if(folds.folds, [&](const CrossLaneFold &f) {
    return combiner->getName().getTypeID() == f.combiner();
  });
  return it == folds.folds.end() ? nullptr : &*it;
}
