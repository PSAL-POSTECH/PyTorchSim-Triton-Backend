#ifndef TORCHSIM_C_DIALECTS_H
#define TORCHSIM_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TorchSim, torchsim);

/// Whether `op` carries MLIR's ElementwiseMappable traits: it maps onto vectors and tensors
/// element by element, whatever dialect it is in.
MLIR_CAPI_EXPORTED bool torchsimIsElementwiseMappable(MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif
