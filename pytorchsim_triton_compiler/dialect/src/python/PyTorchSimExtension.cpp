#include "PyTorchSim-c/Dialects.h"
#include "PyTorchSim/LinearLayouts.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <memory>
#include <vector>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

extern "C" void pytorchsimAttachInterfaces(MlirDialectRegistry registry);
extern "C" void pytorchsimRegisterPasses();

NB_MODULE(_pytorchsim_ext, m) {
  m.doc() = "the pytorchsim, pytorchsim_extern and togsim dialects and every target's, "
            "registered into the bindings that import this";
  m.def(
      "insert_into_registry",
      [](MlirDialectRegistry registry) {
        mlirDialectHandleInsertDialect(mlirGetDialectHandle__pytorchsim__(),
                                       registry);
        mlirDialectHandleInsertDialect(mlirGetDialectHandle__togsim__(), registry);
        pytorchsimInsertTargetDialects(registry);
        mlirDialectHandleInsertDialect(mlirGetDialectHandle__pytorchsim_extern__(), registry);
        pytorchsimAttachInterfaces(registry);
      },
      nb::arg("registry"),
      "Insert `pytorchsim`, `pytorchsim_extern`, `togsim` and every target's dialect into this "
      "registry -- every Context built from it then knows their ops.");
  m.def("register_passes", &pytorchsimRegisterPasses,
        "Register this extension's passes by name, so a textual pipeline can run them.");
  m.def(
      "effects_on",
      [](MlirOperation op, MlirValue value) { return pytorchsimEffectsOn(op, value); },
      nb::arg("op"), nb::arg("value"),
      "PYTORCHSIM_EFFECT_* bits of what `op` states it does to `value`; -1 when it states nothing.");
  m.def(
      "effects_of", [](MlirOperation op) { return pytorchsimEffectsOf(op); }, nb::arg("op"),
      "PYTORCHSIM_EFFECT_* bits of every effect `op` states, its regions' included; -1 when it or an "
      "op in it states nothing.");
  m.def(
      "is_lane_variant", [](MlirOperation op) { return pytorchsimIsLaneVariant(op); },
      nb::arg("op"), "Whether `op` declares LaneVariant: its result differs between lanes.");
  m.def(
      "is_view_like", [](MlirOperation op) { return pytorchsimIsViewLike(op); }, nb::arg("op"),
      "Whether `op` is a ViewLikeOpInterface op: its result is a view of its source.");
  m.def(
      "is_bank_view", [](MlirOperation op) { return pytorchsimIsBankView(op); }, nb::arg("op"),
      "Whether `op` declares BankView: its source as one lane's bank.");
  m.def(
      "is_retyped_view", [](MlirOperation op) { return pytorchsimIsRetypedView(op); }, nb::arg("op"),
      "Whether `op` declares RetypedView: a bank view read as another element type.");
  m.def(
      "is_memory_effect_free", [](MlirOperation op) { return pytorchsimIsMemoryEffectFree(op); },
      nb::arg("op"), "Whether `op` states no memory effect on any value or resource.");
  m.def(
      "loop_ivs",
      [](MlirOperation op) -> nb::object {
        MlirValue v[8];
        int n = pytorchsimLoopIvs(op, v, 8);
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
          int got = pytorchsimLoopBound(op, which, k, &v, &c);
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
        int n = pytorchsimLoopIterArgs(op, a, i, y, 64);
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
        if (!pytorchsimIsBranch(op))
          return nb::none();
        MlirValue v[16];
        int n = pytorchsimBranchControls(op, v, 16);
        if (n < 0 || n > 16)
          return nb::none();
        nb::list out;
        for (int k = 0; k < n; ++k)
          out.append(nb::cast(v[k]));
        return out;
      },
      nb::arg("op"), "The operands of a branch that pick its region, or None when it is no branch.");
  m.def(
      "is_target_op", [](MlirOperation op) { return pytorchsimIsTargetOp(op); }, nb::arg("op"),
      "Whether `op` is an op of a target's dialect.");
  m.def(
      "cross_lane_folds",
      [](const std::string &dialect) {
        MlirStringRef names[16], folds[16];
        bool drops[16];
        int n = pytorchsimCrossLaneFolds(mlirStringRefCreate(dialect.data(), dialect.size()), names, folds,
                                         drops, 16);
        if (n < 0 || n > 16)
          throw nb::value_error("no target has that dialect");
        nb::list out;
        for (int k = 0; k < n; ++k)
          out.append(nb::make_tuple(std::string(names[k].data, names[k].length),
                                    std::string(folds[k].data, folds[k].length), drops[k]));
        return out;
      },
      nb::arg("dialect"), "(combiner op name, fold, drops a NaN) of each fold of its cross-lane unit.");
  m.def(
      "cross_lane_fold_type",
      [](const std::string &dialect, MlirContext ctx) -> nb::object {
        MlirType t = pytorchsimCrossLaneFoldType(ctx, mlirStringRefCreate(dialect.data(), dialect.size()));
        return mlirTypeIsNull(t) ? nb::none() : nb::cast(t);
      },
      nb::arg("dialect"), nb::arg("context"),
      "The element type that target's cross-lane unit folds, or None.");
  m.def(
      "direction_value", [](MlirAttribute attr) { return pytorchsimDirectionValue(attr); },
      nb::arg("attr"), "The pytorchsim::Direction case of a pytorchsim.direction attribute; -1 for another.");
  m.def(
      "direction_attr",
      [](int value, MlirContext ctx) -> nb::object {
        MlirAttribute a = pytorchsimDirectionAttrGet(ctx, value);
        if (mlirAttributeIsNull(a))
          throw nb::value_error("no pytorchsim::Direction case has that value");
        return nb::cast(a);
      },
      nb::arg("value"), nb::arg("context"), "The pytorchsim.direction attribute of case `value`.");
  m.def(
      "is_loop", [](MlirOperation op) { return pytorchsimIsLoop(op); }, nb::arg("op"),
      "Whether `op` is a LoopLikeOpInterface loop, ivs or not.");
  m.def(
      "region_inputs",
      [](MlirValue arg) -> nb::object {
        MlirValue v[64];
        int n = pytorchsimRegionInputs(arg, v, 64);
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
        int n = pytorchsimTerminatorControls(op, v, 16);
        if (n < 0 || n > 16)
          return nb::none();
        nb::list out;
        for (int k = 0; k < n; ++k)
          out.append(nb::cast(v[k]));
        return out;
      },
      nb::arg("op"), "The operands of a region terminator forwarded nowhere, or None when it is none.");
  m.def(
      "reshape_layout",
      [](const pytorchsim::NamedBases &bases, const std::vector<int64_t> &src,
         const std::vector<int64_t> &dst) {
        std::string why;
        pytorchsim::NamedBases got = pytorchsim::reshapeBases(bases, src, dst, why);
        if (!why.empty())
          throw nb::value_error(("reshape_layout refuses " + why).c_str());
        return got;
      },
      nb::arg("bases"), nb::arg("src_shape"), nb::arg("dst_shape"),
      "[(in-dim, bases)] of a layout over `src_shape` after a row-major reshape to `dst_shape`: Triton's "
      "reshapeLayout. Refuses shapes not of powers of two.");
  m.def(
      "signed_ranges",
      [](MlirOperation scope, std::vector<MlirValue> values) -> nb::object {
        int n = (int)values.size();
        std::vector<int64_t> lo(n), hi(n);
        std::unique_ptr<bool[]> known(new bool[n]);
        if (pytorchsimSignedRanges(scope, values.data(), n, lo.data(), hi.data(), known.get()) < 0)
          return nb::none();
        nb::list out;
        for (int i = 0; i < n; ++i)
          out.append(known[i] ? nb::object(nb::make_tuple(lo[i], hi[i])) : nb::none());
        return out;
      },
      nb::arg("scope"), nb::arg("values"),
      "(smin, smax) of each value as upstream's integer range analysis over `scope` proves it, None "
      "where it proves nothing; None when the analysis fails.");
  m.def(
      "has_no_signed_wrap", [](MlirOperation op) { return pytorchsimHasNoSignedWrap(op); },
      nb::arg("op"), "1 or 0 by `op`'s overflow flags, -1 when it states none.");
  m.def(
      "mark_no_signed_wrap", [](MlirOperation op) { return pytorchsimMarkNoSignedWrap(op); },
      nb::arg("op"), "nsw added to `op`'s overflow flags: 0, or -1 when it states none.");
  m.def(
      "view_root", [](MlirValue value) { return pytorchsimViewRoot(value); }, nb::arg("value"),
      "The value `value` is a view of, through every view op in its own memory space.");
}
