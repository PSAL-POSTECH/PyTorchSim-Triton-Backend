#include "PyTorchSim-c/Dialects.h"

#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/TogsimDialect.h"
#include "PyTorchSim/PyTorchSimTpuDialect.h"
#include "PyTorchSim/PyTorchSimExternDialect.h"
#include "PyTorchSim/PyTorchSimRvvDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PyTorchSim, pytorchsim,
                                      ::pytorchsim::PyTorchSimDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Togsim, togsim, ::togsim::TogsimDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PyTorchSimTpu, pytorchsim_tpu,
                                      ::pytorchsim::tpu::PyTorchSimTpuDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PyTorchSimExtern, pytorchsim_extern,
                                      ::pytorchsim::external::PyTorchSimExternDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PyTorchSimRvv, pytorchsim_rvv,
                                      ::pytorchsim::rvv::PyTorchSimRvvDialect)

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
