//===- TorchSimExtension.cpp - the dialect as a python module -------------===//
//
// One function: put `torchsim` in a registry the bindings hand us. The registry
// crosses the language boundary through MLIR's own type caster, so nothing here
// touches a capsule or a raw pointer.
//
//===----------------------------------------------------------------------===//
#include "TorchSim-c/Dialects.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

// The interface that teaches bufferization what a transfer reads and writes is an
// EXTERNAL MODEL, so inserting the dialect is not enough -- it has to be attached to the
// same registry. `torchsimAttachInterfaces` is that call, in lib/CAPI/Dialects.cpp.
extern "C" void torchsimAttachInterfaces(MlirDialectRegistry registry);

NB_MODULE(_torchsim_ext, m) {
  m.doc() = "the torchsim dialect, registered into the bindings that import this";
  m.def(
      "insert_into_registry",
      [](MlirDialectRegistry registry) {
        mlirDialectHandleInsertDialect(mlirGetDialectHandle__torchsim__(),
                                       registry);
        torchsimAttachInterfaces(registry);
      },
      nb::arg("registry"),
      "Insert `torchsim` into this registry -- every Context built from it "
      "then knows the dialect's ops.");
}
