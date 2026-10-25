// The answers of a machine with neither a matrix unit nor a unit that moves data between lanes: each
// lane computes its own share of a dot on RVV alone (rvv_generic.lower_matmul_fma) -- B and the
// product with N on the lanes, A held whole by every lane -- and no reduce keeps its axis on the
// lanes, since nothing folds across them.
#include "TorchsimExt/LayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

namespace {

class RvvGenericLayoutRules : public LayoutRules {
public:
  DenseMap<Operation *, DotForm> dotForms(ArrayRef<DotOp> dots, unsigned) const override {
    DenseMap<Operation *, DotForm> forms;
    for (DotOp dot : dots) {
      int n = cast<RankedTensorType>(dot.getC().getType()).getRank() - 1;
      forms[dot] = {DotForm::kEveryLane, n, n, false};
    }
    return forms;
  }

  bool foldsAcross(ReduceOp) const override { return false; }
};

}

const LayoutRules &rvvGenericLayoutRules() {
  static const RvvGenericLayoutRules rules;
  return rules;
}

}
