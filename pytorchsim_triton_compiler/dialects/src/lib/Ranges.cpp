// What upstream MLIR's integer range analysis proves of a scope's values, and the nsw flag it reads, asked
// and set by the passes in Python through _pytorchsim_ext; pytorchsim.program_id and lane_id state ranges.
#include "PyTorchSim-c/Dialects.h"

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

/// The signed range of each of `values` under the analysis over `scope`, `known[i]` false where it proves
/// nothing, the value is no integer or its range is wider than 64 bits: 0, or -1 when the solver fails.
extern "C" int pytorchsimSignedRanges(MlirOperation scope, const MlirValue *values, int n, int64_t *lo,
                                      int64_t *hi, bool *known) {
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<dataflow::IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(unwrap(scope))))
    return -1;
  for (int i = 0; i < n; ++i) {
    auto *state = solver.lookupState<dataflow::IntegerValueRangeLattice>(unwrap(values[i]));
    unsigned width = state && !state->getValue().isUninitialized()
                         ? state->getValue().getValue().smin().getBitWidth() : 0;
    known[i] = width > 0 && width <= 64;
    if (!known[i])
      continue;
    const ConstantIntRanges &r = state->getValue().getValue();
    lo[i] = r.smin().getSExtValue();
    hi[i] = r.smax().getSExtValue();
  }
  return 0;
}

/// Whether `op` states no signed wrap (1, 0), -1 when it states no overflow flags.
extern "C" int pytorchsimHasNoSignedWrap(MlirOperation c_op) {
  auto op = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(unwrap(c_op));
  return op ? int(op.hasNoSignedWrap()) : -1;
}

/// nsw added to the overflow flags `op` has: 0, or -1 when it states no overflow flags.
extern "C" int pytorchsimMarkNoSignedWrap(MlirOperation c_op) {
  auto op = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(unwrap(c_op));
  if (!op)
    return -1;
  arith::IntegerOverflowFlags flags = op.getOverflowAttr().getValue() | arith::IntegerOverflowFlags::nsw;
  op->setAttr(op.getIntegerOverflowAttrName(), arith::IntegerOverflowFlagsAttr::get(op->getContext(), flags));
  return 0;
}
