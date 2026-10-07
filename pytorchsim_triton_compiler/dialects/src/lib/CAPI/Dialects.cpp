#include "TorchSim-c/Dialects.h"

#include "TorchSim/TorchSimDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TorchSim, torchsim,
                                      ::torchsim::TorchSimDialect)

namespace torchsim {
void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry &);
}

extern "C" void torchsimAttachInterfaces(MlirDialectRegistry registry) {
  ::torchsim::registerBufferizableOpInterfaceExternalModels(*unwrap(registry));
}

bool torchsimIsElementwiseMappable(MlirOperation op) {
  return mlir::OpTrait::hasElementwiseMappableTraits(unwrap(op));
}
