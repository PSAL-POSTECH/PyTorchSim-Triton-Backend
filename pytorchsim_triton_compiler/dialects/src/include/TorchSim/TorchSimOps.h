//===- TorchSimOps.h -------------------------------------------*- C++ -*-===//
#ifndef TORCHSIM_TORCHSIMOPS_H
#define TORCHSIM_TORCHSIMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TorchSim/TorchSimOps.h.inc"

#endif // TORCHSIM_TORCHSIMOPS_H
