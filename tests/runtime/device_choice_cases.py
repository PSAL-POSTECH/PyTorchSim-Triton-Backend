"""Which model library and TOGSim config a target is run on: the ones for its matmul_unit, unless
TORCHSIM_COMPILE_VCIX_MODEL / _TOGSIM_CONFIG name one for every target.
    python3 tests/runtime/device_choice_cases.py        exit 0 iff every case holds"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402
from runtime.device import config  # noqa: E402

TARGETS = os.path.join(ROOT, "tests", "targets")
OVERRIDES = ("TORCHSIM_COMPILE_VCIX_MODEL", "TORCHSIM_COMPILE_TOGSIM_CONFIG")


def _chosen(name):
    t = _target.load(os.path.join(TARGETS, name))
    return os.path.basename(config.vcix_model(t)), config.togsim_config(t)


def _without_overrides(case):
    saved = {k: os.environ.pop(k) for k in OVERRIDES if k in os.environ}
    try:
        return case()
    finally:
        os.environ.update(saved)


def _systolic():
    so, yml = _chosen("tpuv6e.json")
    return (so, os.path.basename(yml)) == ("libtpu.so", "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"), (so, yml)


def _mxu():
    so, yml = _chosen("tpuv6e_mxu.json")
    want = ("libtpu_mxu.so", "systolic_ws_256x256_c1_simple_noc_tpuv6e_mxu.yml")
    return (so, os.path.basename(yml)) == want and os.path.isfile(yml), (so, yml)


def _overrides_win():
    os.environ.update(TORCHSIM_COMPILE_VCIX_MODEL="/x/m.so", TORCHSIM_COMPILE_TOGSIM_CONFIG="/x/c.yml")
    try:
        got = _chosen("tpuv6e_mxu.json")
    finally:
        for k in OVERRIDES:
            os.environ.pop(k)
    return got == ("m.so", "/x/c.yml"), got


def _every_unit_has_both():
    units = set(_target.MATMUL_UNITS)
    return set(config.VCIX_MODELS) == units == set(config.TOGSIM_CONFIGS), sorted(units)


CASES = {
    "a systolic target: libtpu.so, tpuv6e": lambda: _without_overrides(_systolic),
    "an mxu target: libtpu_mxu.so, tpuv6e_mxu": lambda: _without_overrides(_mxu),
    "the overrides win for every target": lambda: _without_overrides(_overrides_win),
    "every matmul_unit has a model and a config": _every_unit_has_both,
}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:44} {'PASS' if ok else 'FAIL'}  {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
