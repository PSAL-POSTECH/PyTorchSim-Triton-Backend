#include "PyTorchSim-c/Dialects.h"

#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/PyTorchSimOps.h"
#include "PyTorchSim/TogsimDialect.h"
#include "PyTorchSim/Targets.h.inc"
#include "PyTorchSim/CrossLaneFolds.h"
#include "mlir/CAPI/Support.h"
#include "PyTorchSim/PyTorchSimExternDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PyTorchSim, pytorchsim,
                                      ::pytorchsim::PyTorchSimDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Togsim, togsim, ::togsim::TogsimDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PyTorchSimExtern, pytorchsim_extern,
                                      ::pytorchsim::external::PyTorchSimExternDialect)

namespace pytorchsim {
void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry &);
void registerHoistLoopInvariantsPass();
}

extern "C" void pytorchsimAttachInterfaces(MlirDialectRegistry registry) {
  ::pytorchsim::registerBufferizableOpInterfaceExternalModels(*unwrap(registry));
}

extern "C" void pytorchsimInsertTargetDialects(MlirDialectRegistry registry) {
  unwrap(registry)->insert<PYTORCHSIM_TARGET_DIALECTS>();
}

extern "C" bool pytorchsimIsTargetOp(MlirOperation op) {
  mlir::Dialect *d = unwrap(op)->getDialect();
  return d && llvm::isa<PYTORCHSIM_TARGET_DIALECTS>(d);
}

extern "C" int pytorchsimCrossLaneFolds(MlirStringRef dialect, MlirStringRef *names, MlirStringRef *folds,
                                        bool *dropsNaN, int cap) {
  auto got = ::pytorchsim::crossLaneFoldsOf(unwrap(dialect));
  if (mlir::failed(got))
    return -1;
  int n = int(got->folds.size());
  for (int k = 0; k < n && k < cap; ++k) {
    names[k] = wrap(got->folds[k].name());
    folds[k] = wrap(got->folds[k].fold);
    dropsNaN[k] = got->folds[k].dropsNaN;
  }
  return n;
}

extern "C" MlirType pytorchsimCrossLaneFoldType(MlirContext ctx, MlirStringRef dialect) {
  auto got = ::pytorchsim::crossLaneFoldsOf(unwrap(dialect));
  if (mlir::failed(got) || !got->element)
    return MlirType{nullptr};
  return wrap(got->element(unwrap(ctx)));
}

extern "C" int pytorchsimDirectionValue(MlirAttribute attr) {
  auto d = llvm::dyn_cast_or_null<::pytorchsim::DirectionAttr>(unwrap(attr));
  return d ? int(d.getValue()) : -1;
}

extern "C" MlirAttribute pytorchsimDirectionAttrGet(MlirContext ctx, int value) {
  auto d = ::pytorchsim::symbolizeDirection(uint32_t(value));
  if (value < 0 || !d)
    return MlirAttribute{nullptr};
  return wrap(::pytorchsim::DirectionAttr::get(unwrap(ctx), *d));
}

extern "C" void pytorchsimRegisterPasses() {
  static bool once = (::pytorchsim::registerHoistLoopInvariantsPass(), true);
  (void)once;
}
