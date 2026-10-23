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
      "effects_of", [](MlirOperation op) { return torchsimEffectsOf(op); }, nb::arg("op"),
      "TORCHSIM_EFFECT_* bits of every effect `op` states, its regions' included; -1 when it or an "
      "op in it states nothing.");
  m.def(
      "is_lane_variant", [](MlirOperation op) { return torchsimIsLaneVariant(op); },
      nb::arg("op"), "Whether `op` declares LaneVariant: its result differs between lanes.");
  m.def(
      "is_memory_effect_free", [](MlirOperation op) { return torchsimIsMemoryEffectFree(op); },
      nb::arg("op"), "Whether `op` states no memory effect on any value or resource.");
  m.def(
      "loop_ivs",
      [](MlirOperation op) -> nb::object {
        MlirValue v[8];
        int n = torchsimLoopIvs(op, v, 8);
        if (n < 0 || n > 8)
          return nb::none();
        nb::list out;
        for (int i = 0; i < n; ++i)
          out.append(nb::cast(v[i]));
        return out;
      },
      nb::arg("op"), "The ivs LoopLikeOpInterface states of `op`, or None when it is no such loop.");
  m.def(
      "loop_bounds",
      [](MlirOperation op, int which) -> nb::object {
        nb::list out;
        for (int k = 0;; ++k) {
          MlirValue v{nullptr};
          int64_t c = 0;
          int got = torchsimLoopBound(op, which, k, &v, &c);
          if (got < 0)
            break;
          if (got == 0)
            out.append(nb::cast(v));
          else
            out.append(nb::int_(c));
        }
        return out;
      },
      nb::arg("op"), nb::arg("which"),
      "Lower (0), upper (1) bounds or steps (2) of a loop: a Value or an int each.");
  m.def(
      "loop_iter_args",
      [](MlirOperation op) -> nb::object {
        MlirValue a[64], i[64], y[64];
        int n = torchsimLoopIterArgs(op, a, i, y, 64);
        if (n < 0 || n > 64)
          return nb::none();
        nb::list out;
        for (int k = 0; k < n; ++k)
          out.append(nb::make_tuple(nb::cast(a[k]), nb::cast(i[k]),
                                    mlirValueIsNull(y[k]) ? nb::none() : nb::cast(y[k])));
        return out;
      },
      nb::arg("op"), "(iter_arg, init, yielded or None) of each iter_arg of a loop, or None.");
  m.def(
      "branch_controls",
      [](MlirOperation op) -> nb::object {
        if (!torchsimIsBranch(op))
          return nb::none();
        MlirValue v[16];
        int n = torchsimBranchControls(op, v, 16);
        if (n < 0 || n > 16)
          return nb::none();
        nb::list out;
        for (int k = 0; k < n; ++k)
          out.append(nb::cast(v[k]));
        return out;
      },
      nb::arg("op"), "The operands of a branch that pick its region, or None when it is no branch.");
  m.def(
      "is_loop", [](MlirOperation op) { return torchsimIsLoop(op); }, nb::arg("op"),
      "Whether `op` is a LoopLikeOpInterface loop, ivs or not.");
  m.def(
      "region_inputs",
      [](MlirValue arg) -> nb::object {
        MlirValue v[64];
        int n = torchsimRegionInputs(arg, v, 64);
        if (n < 0 || n > 64)
          return nb::none();
        nb::list out;
        for (int k = 0; k < n; ++k)
          out.append(nb::cast(v[k]));
        return out;
      },
      nb::arg("arg"), "What flows into a region's block argument from its op and terminators, or None.");
  m.def(
      "terminator_controls",
      [](MlirOperation op) -> nb::object {
        MlirValue v[16];
        int n = torchsimTerminatorControls(op, v, 16);
        if (n < 0 || n > 16)
          return nb::none();
        nb::list out;
        for (int k = 0; k < n; ++k)
          out.append(nb::cast(v[k]));
        return out;
      },
      nb::arg("op"), "The operands of a region terminator forwarded nowhere, or None when it is none.");
  m.def(
      "view_root", [](MlirValue value) { return torchsimViewRoot(value); }, nb::arg("value"),
      "The value `value` is a view of, through every view op in its own memory space.");
}
