#ifndef PYTORCHSIM_C_DIALECTS_H
#define PYTORCHSIM_C_DIALECTS_H

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
  PYTORCHSIM_EFFECT_UNKNOWN = -1,
  PYTORCHSIM_EFFECT_READ = 1,
  PYTORCHSIM_EFFECT_WRITE = 2,
  PYTORCHSIM_EFFECT_ALLOCATE = 4,
  PYTORCHSIM_EFFECT_FREE = 8,
  PYTORCHSIM_EFFECT_SYNC = 16,
};

int pytorchsimEffectsOn(MlirOperation op, MlirValue value);
int pytorchsimEffectsOf(MlirOperation op);
MlirValue pytorchsimViewRoot(MlirValue value);
bool pytorchsimIsLaneVariant(MlirOperation op);
bool pytorchsimIsViewLike(MlirOperation op);
bool pytorchsimIsBankView(MlirOperation op);
bool pytorchsimIsBankWords(MlirOperation op);
bool pytorchsimIsMemoryEffectFree(MlirOperation op);
int pytorchsimLoopIvs(MlirOperation op, MlirValue *out, int cap);
int pytorchsimLoopBound(MlirOperation op, int which, int k, MlirValue *value, int64_t *constant);
int pytorchsimLoopIterArgs(MlirOperation op, MlirValue *args, MlirValue *inits, MlirValue *yielded, int cap);
bool pytorchsimIsBranch(MlirOperation op);
int pytorchsimBranchControls(MlirOperation op, MlirValue *out, int cap);
bool pytorchsimIsLoop(MlirOperation op);
int pytorchsimRegionInputs(MlirValue arg, MlirValue *out, int cap);
int pytorchsimTerminatorControls(MlirOperation op, MlirValue *out, int cap);
int pytorchsimSignedRanges(MlirOperation scope, const MlirValue *values, int n, int64_t *lo, int64_t *hi,
                           bool *known);
int pytorchsimHasNoSignedWrap(MlirOperation op);
int pytorchsimMarkNoSignedWrap(MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif
