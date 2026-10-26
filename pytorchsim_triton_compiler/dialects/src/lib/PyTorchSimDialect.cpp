#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/PyTorchSimOps.h"

using namespace mlir;
using namespace pytorchsim;

#include "PyTorchSim/PyTorchSimOpsDialect.cpp.inc"

void PyTorchSimDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PyTorchSim/PyTorchSimOps.cpp.inc"
      >();
}
