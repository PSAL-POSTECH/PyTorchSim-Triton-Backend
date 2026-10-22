"""isa spad-vector-memory is llc's +spad-vector-memory alone, and toolchain.LLC is checked to honour it;
honoured_by_llc fails on an llc without the feature, which ignores it and compiles the space-0 copy.
    python3 tests/runtime/llc_witness_cases.py        exit 0 iff every case holds"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

V6E = "tpuv6e"


def _asking(base=V6E):
    t = _target.load(base)
    return _target.parse(dict(t.as_dict(), isa=t.isa + [toolchain.SPAD_VECTOR_MEMORY]), "<case>")


def _spelled_for_llc_only():
    plain, asking = _target.load(V6E), _asking()
    got = (asking.llc_mattr(), asking.gcc_march(), asking.spike_isa())
    return got == (plain.llc_mattr() + ",+spad-vector-memory", plain.gcc_march(),
                   plain.spike_isa()), got


def _with_llc(path, t):
    saved = toolchain.LLC
    toolchain.LLC = path
    toolchain._LLC_HONOURS.clear()
    try:
        toolchain.check_llc_honours(t)
        return None
    except toolchain.ToolchainError as e:
        return str(e)
    finally:
        toolchain.LLC = saved
        toolchain._LLC_HONOURS.clear()


def _not_asked_not_run():
    said = _with_llc("/nonexistent/llc", _target.load(V6E))
    return said is None, said


def _missing_llc_named():
    said = _with_llc("/nonexistent/llc", _asking())
    return said is not None and "/nonexistent/llc" in said, said


def _honoured_by_llc():
    toolchain.ensure_bindings()
    said = _with_llc(toolchain.LLC, _asking())
    return said is None, {"llc": toolchain.LLC, "said": said}


CASES = {"spelled_for_llc_only": _spelled_for_llc_only,
         "not_asked_not_run": _not_asked_not_run,
         "missing_llc_named": _missing_llc_named,
         "honoured_by_llc": _honoured_by_llc}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:24} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:600]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
