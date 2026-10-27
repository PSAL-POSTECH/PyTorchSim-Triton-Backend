#ifndef PYTORCHSIM_C_DIALECTS_H
#define PYTORCHSIM_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PyTorchSim, pytorchsim);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Togsim, togsim);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PyTorchSimExtern, pytorchsim_extern);

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
bool pytorchsimIsRetypedView(MlirOperation op);
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
/* Every target's dialect (the build's registration list, Targets.h.inc) into `registry`. */
void pytorchsimInsertTargetDialects(MlirDialectRegistry registry);
/* Whether `op` is an op of a target's dialect. */
bool pytorchsimIsTargetOp(MlirOperation op);
/* The cross-lane folds of the target whose dialect is `dialect` (pytorchsim::crossLaneFoldsOf): how many,
   and each combiner's op name, fold and whether it drops a NaN in the first `cap`; -1 for no target's. */
int pytorchsimCrossLaneFolds(MlirStringRef dialect, MlirStringRef *names, MlirStringRef *folds,
                             bool *dropsNaN, int cap);
/* The element type that target's cross-lane unit folds, or a null type when it folds nothing or is no
   target. */
MlirType pytorchsimCrossLaneFoldType(MlirContext ctx, MlirStringRef dialect);
/* A pytorchsim.direction attribute's case (pytorchsim::Direction), or -1 when `attr` is none. */
int pytorchsimDirectionValue(MlirAttribute attr);
/* The pytorchsim.direction attribute of case `value`, or a null attribute when it names no case. */
MlirAttribute pytorchsimDirectionAttrGet(MlirContext ctx, int value);

#ifdef __cplusplus
}
#endif

#endif
