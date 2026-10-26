"""A target is its kind (passes.target_kind.TargetKind): what a new kind writes, and where a kind that writes
no unit stops a kernel. python3 tests/passes/target_kind_cases.py        exit 0 iff every case holds"""
import json
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext import tpu_kind as tpu  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    b11_lower_convert, b12_convert_pytorchsim_to_target_kind as b12, b13_schedule_matmul, b14_schedule_vpu,
    lower, transform)
from pytorchsim_triton_compiler.lowering.passes.target_kind import (  # noqa: E402
    PassGateError, TargetKind, target_kind)


def _machine(name, *isa):
    accel = next((e for e in isa if _target.is_accel(e)), None)
    return types.SimpleNamespace(name=name, isa=["m", "a", "f", "d", "c", "v", "zvl256b", *isa],
                                 kind=_target.ACCELERATORS.get(accel))


def _refused(fn):
    try:
        fn()
    except PassGateError as e:
        return str(e)
    raise AssertionError("taken")


def the_tpu_kind_is_the_v6e_s_and_the_passes_that_ask_it_open_to_custom():
    assert target_kind(_machine("v", "pytorchsim_tpuv6e")) is tpu.KIND
    assert lower.TO_CUSTOM[:3] == [b12, b13_schedule_matmul, b14_schedule_vpu]


def every_accelerator_of_a_family_runs_the_family_s_kind():
    members = [e for e, f in _target.ACCELERATORS.items() if f == "tpu"]
    assert members, _target.ACCELERATORS
    for accel in members:
        assert target_kind(_machine(accel, accel)) is tpu.KIND, accel


def an_isa_naming_no_accelerator_or_an_unknown_one_has_no_kind():
    said = _refused(lambda: target_kind(_machine("plain")))
    assert "names no accelerator" in said and "plain" in said, said
    said = _refused(lambda: target_kind(_machine("y", "pytorchsim_tpuv9")))
    assert "pytorchsim_tpuv9" in said and "no family" in said, said


def a_kind_missing_data_movement_is_refused_when_it_is_made_naming_what_it_lacks():
    class Bare(TargetKind):
        DMA_DIMS = 4

        def lower_transfers(self, module, transfers, vectorlane):
            pass
    try:
        Bare()
    except TypeError as e:
        said = str(e)
    else:
        raise AssertionError("made")
    assert all(m in said for m in ("INDEX_KEY", "insn", "walk", "sram_footprints")), said
    assert "DMA_DIMS" not in said and "lower_transfers" not in said, said


def the_transform_asks_the_kind_once_the_machine_is_known():
    first = [title for title, _ in transform.PHASES].index(transform.MACHINE_KNOWN)
    assert any(b11_lower_convert in ps for _, ps in transform.PHASES[first:])


# A kind that writes only TargetKind's abstract members, data movement, borrowed from the tpu kind; the case
# kernel run through the suite's driver (Spike and the torch compare after the compile).
_MINIMAL = r"""
import os, runpy, sys
root, case, work = sys.argv[1:4]
sys.path.insert(0, root)
from pytorchsim_triton_compiler.lowering.ext import tpu_kind
from pytorchsim_triton_compiler.lowering.passes import target_kind

class Minimal(target_kind.TargetKind):
    DMA_DIMS, INDEX_KEY = tpu_kind.KIND.DMA_DIMS, tpu_kind.KIND.INDEX_KEY

    def lower_transfers(self, *a):
        tpu_kind.KIND.lower_transfers(*a)

    def insn(self, *a):
        return tpu_kind.KIND.insn(*a)

    def walk(self, *a):
        return tpu_kind.KIND.walk(*a)

    def sram_footprints(self, *a):
        return tpu_kind.KIND.sram_footprints(*a)

target_kind._KINDS["tpu_kind"] = Minimal()
sys.argv = ["pytorchsim-compile", case, "--workdir", work]
runpy.run_path(os.path.join(root, "tests", "pytorchsim-compile"), run_name="__main__")
"""


def _minimal(case):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([sys.executable, "-c", _MINIMAL, ROOT,
                              os.path.join(ROOT, "tests", "cases", case), d],
                             capture_output=True, text=True, cwd=ROOT)
    return out.stdout + out.stderr


def a_kind_that_writes_only_data_movement_runs_a_kernel_that_needs_no_unit():
    said = _minimal("smoke/add.py")
    assert "ALL STAGES PASSED" in said, said[-3000:]


def a_unit_op_the_kind_does_not_lower_is_refused_by_the_pass_that_asks_it():
    said = _minimal("smoke/gemm.py")
    assert "convert-pytorchsim-to-target-kind declared these would not survive it" in said, said[-3000:]
    assert "pytorchsim.dot" in said, said[-3000:]


_NO_KIND = r"""
import copy, json, sys, tempfile
sys.path.insert(0, sys.argv[1])
from pytorchsim_triton_compiler import cli, machine
t = copy.copy(machine.target())
t.isa, t.name = [e for e in t.isa if e != "pytorchsim_tpuv6e"], "no-kind"
machine._LAZY["TARGET"] = t
d = tempfile.mkdtemp()
print("NO-KIND " + json.dumps({"rc": cli.main([sys.argv[2], "--workdir", d, "--to", "transform"])}))
"""


def a_target_without_a_kind_is_refused_before_any_stage_runs():
    out = subprocess.run([sys.executable, "-c", _NO_KIND, ROOT,
                          os.path.join(ROOT, "tests", "cases", "smoke", "add.py")],
                         capture_output=True, text=True)
    said = out.stdout + out.stderr
    assert "names no accelerator" in said and "no-kind" in said, said
    assert "[1/" not in said, said      # no stage started
    got = next((ln for ln in out.stdout.splitlines() if ln.startswith("NO-KIND ")), None)
    assert got is None or json.loads(got[8:])["rc"] != 0, said


def the_stage_list_and_the_log_name_the_passes_that_ask_the_kind():
    import tempfile
    env = dict(os.environ, PYTORCHSIM_COMPILE_TARGET="tpuv6e")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"),
                          "--list-stages"], env=env, capture_output=True, text=True).stdout
    assert ("convert-pytorchsim-to-target-kind -> pytorchsim-schedule-matmul -> pytorchsim-schedule-vpu -> "
            "convert-to-vcix") in out, out
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"),
                        os.path.join(ROOT, "tests", "cases", "smoke", "gemm.py"), "--workdir", d,
                        "--to", "lower"], env=env, capture_output=True, check=True)
        log = open(os.path.join(d, "stage.log")).read()
    assert "[lower] TO_CUSTOM for tpuv6e " in log and "convert-pytorchsim-to-target-kind" in log


CASES = [the_tpu_kind_is_the_v6e_s_and_the_passes_that_ask_it_open_to_custom,
         every_accelerator_of_a_family_runs_the_family_s_kind,
         an_isa_naming_no_accelerator_or_an_unknown_one_has_no_kind,
         a_kind_missing_data_movement_is_refused_when_it_is_made_naming_what_it_lacks,
         the_transform_asks_the_kind_once_the_machine_is_known,
         a_kind_that_writes_only_data_movement_runs_a_kernel_that_needs_no_unit,
         a_unit_op_the_kind_does_not_lower_is_refused_by_the_pass_that_asks_it,
         a_target_without_a_kind_is_refused_before_any_stage_runs,
         the_stage_list_and_the_log_name_the_passes_that_ask_the_kind]

if __name__ == "__main__":
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
