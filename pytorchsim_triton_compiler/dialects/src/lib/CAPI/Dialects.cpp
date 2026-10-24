#include "PyTorchSim-c/Dialects.h"

#include "PyTorchSim/PyTorchSimDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PyTorchSim, pytorchsim,
                                      ::pytorchsim::PyTorchSimDialect)

namespace pytorchsim {
void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry &);
void registerHoistLoopInvariantsPass();
}

extern "C" void pytorchsimAttachInterfaces(MlirDialectRegistry registry) {
  ::pytorchsim::registerBufferizableOpInterfaceExternalModels(*unwrap(registry));
}

extern "C" void pytorchsimRegisterPasses() {
  static bool once = (::pytorchsim::registerHoistLoopInvariantsPass(), true);
  (void)once;
}
