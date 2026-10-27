#include "Dialect/PyTorchSimTpu/IR/PyTorchSimTpuDialect.h"
#include "Dialect/PyTorchSimTpu/IR/PyTorchSimTpuOps.h"

using namespace mlir;
using namespace pytorchsim::tpu;

#include "PyTorchSim/PyTorchSimTpuOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PyTorchSim/PyTorchSimTpuOps.cpp.inc"

void PyTorchSimTpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PyTorchSim/PyTorchSimTpuOps.cpp.inc"
      >();
}

/// A packed view is the bank it is made from, seen as other elements.
Value VpuPackedViewOp::getViewSource() { return getBank(); }
