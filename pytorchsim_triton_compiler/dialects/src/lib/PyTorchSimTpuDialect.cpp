#include "PyTorchSim/PyTorchSimTpuDialect.h"
#include "PyTorchSim/PyTorchSimTpuOps.h"

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

/// A bank view and a bank's words are the buffer they are made from, seen with other strides or elements.
Value VpuBankViewOp::getViewSource() { return getSource(); }
Value VpuBankWordsOp::getViewSource() { return getBank(); }
