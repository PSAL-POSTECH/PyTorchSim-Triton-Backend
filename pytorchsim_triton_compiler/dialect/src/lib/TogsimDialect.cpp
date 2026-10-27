#include "PyTorchSim/TogsimDialect.h"
#include "PyTorchSim/TogsimOps.h"

using namespace mlir;
using namespace togsim;

#include "PyTorchSim/TogsimOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PyTorchSim/TogsimOps.cpp.inc"

void TogsimDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PyTorchSim/TogsimOps.cpp.inc"
      >();
}
