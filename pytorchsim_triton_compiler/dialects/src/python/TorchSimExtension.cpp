#include "TorchSim-c/Dialects.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

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
  m.def("is_elementwise_mappable", &torchsimIsElementwiseMappable, nb::arg("op"),
        "Whether `op` carries MLIR's ElementwiseMappable traits.");
}
