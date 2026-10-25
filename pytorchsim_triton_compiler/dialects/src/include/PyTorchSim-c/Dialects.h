#ifndef TORCHSIM_C_DIALECTS_H
#define TORCHSIM_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PyTorchSim, pytorchsim);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Togsim, togsim);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PyTorchSimTpu, pytorchsim_tpu);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PyTorchSimExtern, pytorchsim_extern);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PyTorchSimRvv, pytorchsim_rvv);

/* What pytorchsimEffectsOn and pytorchsimEffectsOf answer: bits of the effects one op states. */
enum {
  TORCHSIM_EFFECT_UNKNOWN = -1,
  TORCHSIM_EFFECT_READ = 1,
  TORCHSIM_EFFECT_WRITE = 2,
  TORCHSIM_EFFECT_ALLOCATE = 4,
  TORCHSIM_EFFECT_FREE = 8,
  TORCHSIM_EFFECT_SYNC = 16,
};

int pytorchsimEffectsOn(MlirOperation op, MlirValue value);
int pytorchsimEffectsOf(MlirOperation op);
MlirValue pytorchsimViewRoot(MlirValue value);
bool pytorchsimIsLaneVariant(MlirOperation op);
bool pytorchsimIsMemoryEffectFree(MlirOperation op);
int pytorchsimLoopIvs(MlirOperation op, MlirValue *out, int cap);
int pytorchsimLoopBound(MlirOperation op, int which, int k, MlirValue *value, int64_t *constant);
int pytorchsimLoopIterArgs(MlirOperation op, MlirValue *args, MlirValue *inits, MlirValue *yielded, int cap);
bool pytorchsimIsBranch(MlirOperation op);
int pytorchsimBranchControls(MlirOperation op, MlirValue *out, int cap);
bool pytorchsimIsLoop(MlirOperation op);
int pytorchsimRegionInputs(MlirValue arg, MlirValue *out, int cap);
int pytorchsimTerminatorControls(MlirOperation op, MlirValue *out, int cap);

#ifdef __cplusplus
}
#endif

#endif
