"""Which model library and TOGSim config a target is run on: the library whose unit is its
matmul_unit and the config of the machine it is named for, unless TORCHSIM_COMPILE_VCIX_MODEL /
_TOGSIM_CONFIG name one for every target.
    python3 tests/runtime/device_choice_cases.py        exit 0 iff every case holds"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402
from runtime.device import config  # noqa: E402

TARGETS = os.path.join(ROOT, "tests", "targets")
TPUV6E = os.path.join(TARGETS, "tpuv6e.json")
OVERRIDES = ("TORCHSIM_COMPILE_VCIX_MODEL", "TORCHSIM_COMPILE_TOGSIM_CONFIG")


def _systolic_elsewhere():
    """A machine this repo ships no target for: the v6e's numbers under another name, on the systolic array."""
    raw = _target.load(TPUV6E).restated_for("systolic")
    return _target.parse(dict(raw, name="elsewhere"), "<case>")


def _without_overrides(case):
    saved = {k: os.environ.pop(k) for k in OVERRIDES if k in os.environ}
    try:
        return case()
    finally:
        os.environ.update(saved)


def _tpuv6e():
    t = _target.load(TPUV6E)
    so, yml = os.path.basename(config.vcix_model(t)), config.togsim_config(t)
    want = ("tpuv6_mxu", "libtpuv6.so", "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml")
    return (t.matmul_unit, so, os.path.basename(yml)) == want and os.path.isfile(yml), (so, yml)


def _systolic_is_libtpu_and_no_config_is_guessed():
    t = _systolic_elsewhere()
    so = os.path.basename(config.vcix_model(t))
    try:
        got = config.togsim_config(t)
    except ValueError as e:
        return so == "libtpu.so" and "TORCHSIM_COMPILE_TOGSIM_CONFIG" in str(e), (so, str(e))
    return False, (so, got)


def _overrides_win():
    os.environ.update(TORCHSIM_COMPILE_VCIX_MODEL="/x/m.so", TORCHSIM_COMPILE_TOGSIM_CONFIG="/x/c.yml")
    try:
        got = [(os.path.basename(config.vcix_model(t)), config.togsim_config(t))
               for t in (_target.load(TPUV6E), _systolic_elsewhere())]
    finally:
        for k in OVERRIDES:
            os.environ.pop(k)
    return got == [("m.so", "/x/c.yml")] * 2, got


def _every_unit_has_a_model_and_every_target_a_config():
    names = sorted(_target.load(p).name for p in glob.glob(os.path.join(TARGETS, "*.json")))
    got = (sorted(config.VCIX_MODELS), names, sorted(config.TOGSIM_CONFIGS))
    return got == (sorted(_target.MATMUL_UNITS), names, names), got


CASES = {
    "tpuv6e: libtpuv6.so, its own TOGSim config": lambda: _without_overrides(_tpuv6e),
    "a systolic target elsewhere: libtpu.so, no config": lambda: _without_overrides(
        _systolic_is_libtpu_and_no_config_is_guessed),
    "the overrides win for every target": lambda: _without_overrides(_overrides_win),
    "every unit a model, every target a config": _every_unit_has_a_model_and_every_target_a_config,
}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:52} {'PASS' if ok else 'FAIL'}  {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
