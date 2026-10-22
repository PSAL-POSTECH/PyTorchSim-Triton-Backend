#include "TorchSim-c/Dialects.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

extern "C" void torchsimAttachInterfaces(MlirDialectRegistry registry);
extern "C" void torchsimRegisterPasses();

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
  m.def("register_passes", &torchsimRegisterPasses,
        "Register this extension's passes by name, so a textual pipeline can run them.");
  m.def(
      "effects_on",
      [](MlirOperation op, MlirValue value) { return torchsimEffectsOn(op, value); },
      nb::arg("op"), nb::arg("value"),
      "TORCHSIM_EFFECT_* bits of what `op` states it does to `value`; -1 when it states nothing.");
  m.def(
      "is_lane_variant", [](MlirOperation op) { return torchsimIsLaneVariant(op); },
      nb::arg("op"), "Whether `op` declares LaneVariant: its result differs between lanes.");
  m.def(
      "is_memory_effect_free", [](MlirOperation op) { return torchsimIsMemoryEffectFree(op); },
      nb::arg("op"), "Whether `op` states no memory effect on any value or resource.");
  m.def(
      "view_root", [](MlirValue value) { return torchsimViewRoot(value); }, nb::arg("value"),
      "The value `value` is a view of, through every view op in its own memory space.");
}
