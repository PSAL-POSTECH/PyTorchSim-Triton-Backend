#ifndef TORCHSIM_C_DIALECTS_H
#define TORCHSIM_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TorchSim, torchsim);

/* What torchsimEffectsOn answers: bits of the effects one op states on one value. */
enum {
  TORCHSIM_EFFECT_UNKNOWN = -1,
  TORCHSIM_EFFECT_READ = 1,
  TORCHSIM_EFFECT_WRITE = 2,
  TORCHSIM_EFFECT_ALLOCATE = 4,
  TORCHSIM_EFFECT_FREE = 8,
  TORCHSIM_EFFECT_SYNC = 16,
};

int torchsimEffectsOn(MlirOperation op, MlirValue value);
MlirValue torchsimViewRoot(MlirValue value);
bool torchsimIsLaneVariant(MlirOperation op);
bool torchsimIsMemoryEffectFree(MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif
