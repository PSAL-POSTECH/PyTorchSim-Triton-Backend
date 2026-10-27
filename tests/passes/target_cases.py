"""A subtarget lowers by its target (targets.compiler.Target): what a new target writes, and where a target
that writes no unit stops a kernel. python3 tests/passes/target_cases.py        exit 0 iff every case holds"""
import json
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402
from pytorchsim_triton_compiler import targets  # noqa: E402
from pytorchsim_triton_compiler.targets.tpu import compiler as tpu  # noqa: E402
from pytorchsim_triton_compiler.passes import lower, transform  # noqa: E402
from pytorchsim_triton_compiler.targets import target  # noqa: E402
from pytorchsim_triton_compiler.targets.compiler import PassGateError, Target  # noqa: E402


def _machine(name, *isa):
    accel = next((e for e in isa if _target.is_accel(e)), None)
    return types.SimpleNamespace(name=name, isa=["m", "a", "f", "d", "c", "v", "zvl256b", *isa],
                                 target=targets.subtargets().get(accel))


def _refused(fn):
    try:
        fn()
    except PassGateError as e:
        return str(e)
    raise AssertionError("taken")


def the_tpu_target_is_the_v6e_s_and_the_passes_that_ask_it_open_to_custom():
    assert type(target(_machine("v", "pytorchsim_tpuv6e"))) is tpu.TpuTarget
    t = tpu.TpuTarget()
    assert lower.TO_CUSTOM[0] is lower.TARGET_LOWER and transform.TARGET_TRANSFORM in transform.PHASES[-1][1]
    assert [m.__name__.rsplit('.', 1)[-1] for m in t.LOWER_PASSES] == [
        "convert_pytorchsim_to_pytorchsim_tpu", "schedule_mxu_stream", "schedule_vpu_stream"], t.LOWER_PASSES


def every_accelerator_a_target_lists_runs_that_target():
    members = [e for e, f in targets.subtargets().items() if f == "tpu"]
    assert members, targets.subtargets()
    for accel in members:
        assert type(target(_machine(accel, accel))) is tpu.TpuTarget, accel


def an_isa_naming_no_accelerator_or_an_unknown_one_has_no_target():
    said = _refused(lambda: target(_machine("plain")))
    assert "names no accelerator" in said and "plain" in said, said
    said = _refused(lambda: target(_machine("y", "pytorchsim_tpuv9")))
    assert "pytorchsim_tpuv9" in said and "no target lists it" in said, said


def a_target_missing_an_abstract_member_is_refused_when_it_is_made_naming_what_it_lacks():
    class Bare(Target):
        DMA_DIMS = 4

        def lower_transfers(self, module, transfers, vectorlane):
            pass
    try:
        Bare()
    except TypeError as e:
        said = str(e)
    else:
        raise AssertionError("made")
    lacks = ("INDEX_KEY", "insn", "walk", "sram_footprints", "TRANSFORM_PASSES", "DIALECT")
    assert all(m in said for m in lacks), said
    assert "DMA_DIMS" not in said and "lower_transfers" not in said, said


def the_transform_asks_the_target_once_the_machine_is_known():
    first = [title for title, _ in transform.PHASES].index(transform.MACHINE_KNOWN)
    assert any(transform.TARGET_TRANSFORM in ps for _, ps in transform.PHASES[first:])


# A target that writes only Target's abstract members (data movement, bank view, layout conversions),
# borrowed from the tpu target; the case kernel run through the suite's driver (Spike and the torch compare).
_MINIMAL = r"""
import os, runpy, sys
root, case, work = sys.argv[1:4]
sys.path.insert(0, root)
from pytorchsim_triton_compiler import targets
from pytorchsim_triton_compiler.targets.compiler import Target
from pytorchsim_triton_compiler.targets.tpu.compiler import TpuTarget
TPU = TpuTarget()

class Minimal(Target):
    DMA_DIMS, INDEX_KEY = TPU.DMA_DIMS, TPU.INDEX_KEY
    DIALECT = TPU.DIALECT

    TRANSFORM_PASSES = TPU.TRANSFORM_PASSES

    def lower_transfers(self, *a):
        TPU.lower_transfers(*a)

    def insn(self, *a):
        return TPU.insn(*a)

    def walk(self, *a):
        return TPU.walk(*a)

    def sram_footprints(self, *a):
        return TPU.sram_footprints(*a)

targets._TARGETS["tpu"] = Minimal()
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


def a_target_that_writes_only_the_abstract_members_runs_a_kernel_that_needs_no_unit():
    said = _minimal("smoke/add.py")
    assert "ALL STAGES PASSED" in said, said[-3000:]


def a_unit_op_the_target_does_not_lower_is_refused_by_the_pass_that_asks_it():
    said = _minimal("smoke/gemm.py")
    assert "the target's LOWER_PASSES () left" in said, said[-3000:]
    assert "pytorchsim.dot" in said, said[-3000:]


_NO_TARGET = r"""
import copy, json, sys, tempfile
sys.path.insert(0, sys.argv[1])
from pytorchsim_triton_compiler import cli
from pytorchsim_triton_compiler.driver import machine
t = copy.copy(machine.subtarget())
t.isa, t.name = [e for e in t.isa if e != "pytorchsim_tpuv6e"], "no-target"
machine._LAZY["SUBTARGET"] = t
d = tempfile.mkdtemp()
print("NO-TARGET " + json.dumps({"rc": cli.main([sys.argv[2], "--workdir", d, "--to", "transform"])}))
"""


def a_subtarget_without_a_target_is_refused_before_any_stage_runs():
    out = subprocess.run([sys.executable, "-c", _NO_TARGET, ROOT,
                          os.path.join(ROOT, "tests", "cases", "smoke", "add.py")],
                         capture_output=True, text=True)
    said = out.stdout + out.stderr
    assert "names no accelerator" in said and "no-target" in said, said
    assert "[1/" not in said, said      # no stage started
    got = next((ln for ln in out.stdout.splitlines() if ln.startswith("NO-TARGET ")), None)
    assert got is None or json.loads(got[10:])["rc"] != 0, said


def the_stage_list_and_the_log_name_the_passes_that_ask_the_target():
    import tempfile
    env = dict(os.environ, PYTORCHSIM_COMPILE_TARGET="tpuv6e")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"),
                          "--list-stages"], env=env, capture_output=True, text=True).stdout
    assert ("convert-pytorchsim-to-pytorchsim-tpu -> pytorchsim-schedule-mxu-stream -> "
            "pytorchsim-schedule-vpu-stream -> convert-to-vcix") in out, out
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"),
                        os.path.join(ROOT, "tests", "cases", "smoke", "gemm.py"), "--workdir", d,
                        "--to", "lower"], env=env, capture_output=True, check=True)
        log = open(os.path.join(d, "stage.log")).read()
    assert "[lower] TO_CUSTOM for tpuv6e " in log and "convert-pytorchsim-to-pytorchsim-tpu" in log


def a_target_s_dialect_is_registered_from_its_own_directory():
    """Breaks if the tpu target's dialect is not the one the build registers from targets/tpu/CMakeLists.txt
    (an op of it unregistered, or not told a target's op), if a common op counts as a target's, if discovery
    misses targets/tpu, or if the operand table and the .td stamp leave the target's ops out."""
    from pytorchsim_triton_compiler import toolchain
    toolchain.ensure_bindings()
    from mlir import ir
    from pytorchsim_triton_compiler.dialect import pytorchsim
    from pytorchsim_triton_compiler.targets.tpu import pytorchsim_tpu
    pytorchsim.register()
    tpu = os.path.join(ROOT, "pytorchsim_triton_compiler", "targets", "tpu")
    assert os.path.exists(os.path.join(tpu, "CMakeLists.txt")), tpu
    assert targets.names() == ("tpu",), targets.names()
    text = ("func.func @f(%m: memref<1x4xbf16, 1>) {\n"
            '  %w = "pytorchsim_tpu.vpu.packed_view"(%m) : (memref<1x4xbf16, 1>) -> memref<1x2xi32, 1>\n'
            '  %v = "pytorchsim.bank_view"(%m) : (memref<1x4xbf16, 1>) -> memref<1x4xbf16, 1>\n'
            '  %l = "pytorchsim.lane_id"() <{lanes = 4 : i64}> : () -> vector<4xi32>\n  return\n}\n')
    with ir.Context(), ir.Location.unknown():
        m = ir.Module.parse(text)
        ops = list(m.body.operations[0].regions[0].blocks[0].operations)
        assert isinstance(ops[0].opview, pytorchsim_tpu.VpuPackedViewOp), ops[0]
        assert isinstance(ops[1].opview, pytorchsim.BankViewOp), ops[1]
        assert pytorchsim.is_target_op(ops[0]) and not pytorchsim.is_target_op(ops[1])
        assert not pytorchsim.is_target_op(ops[2]) and not pytorchsim.is_target_op(ops[3])
        assert pytorchsim.operand_number(ops[0], "bank") == 0
        assert pytorchsim.operand_number(ops[1], "source") == 0
    stamp = pytorchsim._load_generated("_pytorchsim_stamp")
    assert "PyTorchSimTpuOps.td" in stamp.TARGET_TD_SOURCES, stamp.TARGET_TD_SOURCES


_ANCHOR = r"""
import os, sys, tempfile
from pytorchsim_triton_compiler import frontend as triton_compile
from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend
ctx = ir.context(); ir.load_dialects(ctx); make_backend(GPUTarget("cpu", 0, 0)).load_dialects(ctx)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "k.mlir")
    open(p, "w").write('module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, '
                       '"ttg.threads-per-warp" = 128 : i32} { tt.func @k() { tt.return } }')
    pm = ir.pass_manager(ctx)
    triton_compile._anchor(pm, passes, 128, anchor=sys.argv[1])
    pm.run(ir.parse_mlir_module(p, ctx), p)
print("ANCHORED")
"""


def _anchored(option):
    from pytorchsim_triton_compiler import toolchain
    env = toolchain.triton_env()
    env["PYTHONPATH"] = ROOT + os.pathsep + env["PYTHONPATH"]
    got = subprocess.run([sys.executable, "-c", _ANCHOR, option], env=env, capture_output=True, text=True)
    return got.stdout + got.stderr


def a_target_s_lane_reduces_are_stated_once_and_the_anchor_reads_them_by_its_dialect():
    """Breaks if the tpu target's lowering and the TritonGPU anchor read the XLU's lane reduces from two
    places (the anchor handed op names and a type spelling again), or if the anchor takes a dialect no target
    has."""
    from pytorchsim_triton_compiler.dialect import pytorchsim
    from pytorchsim_triton_compiler.targets.tpu import pytorchsim_tpu
    from pytorchsim_triton_compiler import frontend
    from pytorchsim_triton_compiler.targets.tpu import xlu
    folds = pytorchsim.cross_lane_folds(pytorchsim_tpu.DIALECT)
    assert xlu.XLU_FOLDS == {name: f.fold for name, f in folds.items()}, xlu.XLU_FOLDS
    assert xlu.XLU_FOLDS_DROPPING_NAN == {"arith.maxnumf", "arith.minnumf"}, xlu
    assert frontend.anchor_options() == f"target={pytorchsim_tpu.DIALECT}"
    try:
        pytorchsim.cross_lane_folds("pytorchsim_notarget")
    except ValueError:
        pass
    else:
        raise AssertionError("a dialect no target registers has lane reduces")
    assert "ANCHORED" in _anchored(frontend.anchor_options())
    said = _anchored("target=pytorchsim_notarget")
    assert "ANCHORED" not in said and "is no registered target's dialect" in said, said[-1500:]


def a_pass_s_mlir_name_is_one_rule_of_its_file_name():
    """Breaks if a pass's name is decided by probing for an upstream dialect: a module of conversion/ is
    convert-<name>, one named convert_* keeps it, any other takes the pytorchsim- prefix, an upstream
    dialect's name at its head included; or if a pass is not found by its name (what it FORBIDS)."""
    from pytorchsim_triton_compiler.passes import forbids, pass_name
    to_llvm = "pytorchsim_triton_compiler.passes.conversion.pytorchsim_to_llvm"
    assert pass_name(to_llvm) == "convert-pytorchsim-to-llvm"
    assert pass_name("convert_pytorchsim_to_pytorchsim_tpu") == "convert-pytorchsim-to-pytorchsim-tpu"
    assert pass_name("vectorize") == "pytorchsim-vectorize"
    assert pass_name("arith_cleanup") == "pytorchsim-arith-cleanup"
    assert forbids("pytorchsim-legalize-vectors")[0] and forbids("pytorchsim-legalize-vectors")[2]
    assert forbids("convert-to-vcix")[2].endswith(os.path.join("conversion", "to_vcix.py"))


CASES = [the_tpu_target_is_the_v6e_s_and_the_passes_that_ask_it_open_to_custom,
         a_target_s_dialect_is_registered_from_its_own_directory,
         a_target_s_lane_reduces_are_stated_once_and_the_anchor_reads_them_by_its_dialect,
         every_accelerator_a_target_lists_runs_that_target,
         an_isa_naming_no_accelerator_or_an_unknown_one_has_no_target,
         a_target_missing_an_abstract_member_is_refused_when_it_is_made_naming_what_it_lacks,
         the_transform_asks_the_target_once_the_machine_is_known,
         a_target_that_writes_only_the_abstract_members_runs_a_kernel_that_needs_no_unit,
         a_unit_op_the_target_does_not_lower_is_refused_by_the_pass_that_asks_it,
         a_subtarget_without_a_target_is_refused_before_any_stage_runs,
         the_stage_list_and_the_log_name_the_passes_that_ask_the_target,
         a_pass_s_mlir_name_is_one_rule_of_its_file_name]

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
