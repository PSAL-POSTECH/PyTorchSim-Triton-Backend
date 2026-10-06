//===- TorchSimDialect.cpp -------------------------------------*- C++ -*-===//
#include "TorchSim/TorchSimDialect.h"
#include "TorchSim/TorchSimOps.h"

using namespace mlir;
using namespace torchsim;

#include "TorchSim/TorchSimOpsDialect.cpp.inc"

void TorchSimDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TorchSim/TorchSimOps.cpp.inc"
      >();
}
