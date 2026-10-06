//===- Dialects.h - the C API this dialect is registered through -*- C -*-===//
//
// The registration hook MLIR's own dialects use: the macro mints
// `mlirGetDialectHandle__torchsim__`, and the python extension hands that handle
// to a registry the bindings already own. See dialect/python/TorchSimExtension.cpp.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHSIM_C_DIALECTS_H
#define TORCHSIM_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TorchSim, torchsim);

#ifdef __cplusplus
}
#endif

#endif // TORCHSIM_C_DIALECTS_H
