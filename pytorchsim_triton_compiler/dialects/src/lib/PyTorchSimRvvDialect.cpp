#include "PyTorchSim/PyTorchSimRvvDialect.h"
#include "PyTorchSim/PyTorchSimRvvOps.h"

using namespace mlir;
using namespace pytorchsim::rvv;

#include "PyTorchSim/PyTorchSimRvvOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PyTorchSim/PyTorchSimRvvOps.cpp.inc"

void PyTorchSimRvvDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PyTorchSim/PyTorchSimRvvOps.cpp.inc"
      >();
}

/// A bank view is the buffer it is made from, seen with other strides.
Value BankViewOp::getViewSource() { return getSource(); }
