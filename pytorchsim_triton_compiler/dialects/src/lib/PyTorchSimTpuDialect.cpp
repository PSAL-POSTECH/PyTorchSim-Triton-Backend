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
