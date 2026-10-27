#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/PyTorchSimOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pytorchsim;

#include "PyTorchSim/PyTorchSimOpsDialect.cpp.inc"
#include "PyTorchSim/PyTorchSimEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "PyTorchSim/PyTorchSimAttrs.cpp.inc"

void PyTorchSimDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "PyTorchSim/PyTorchSimAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "PyTorchSim/PyTorchSimOps.cpp.inc"
      >();
}
