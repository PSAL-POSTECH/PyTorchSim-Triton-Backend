"""failure.json as a caller reads it: the kind the type of the refusal names and that kind's facts,
for a scratchpad over its budget, "error" for a type no kind names, and none left behind by a run
that passes or stops before any stage.
    python3 tests/runtime/failure_cases.py        exit 0 iff every case gets what it names"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler import pipeline  # noqa: E402
from pytorchsim_triton_compiler.contract import failure  # noqa: E402
from pytorchsim_triton_compiler.stage import Ctx, SpadOverflow, SpillNotAllowed, StageError  # noqa: E402

def _compile(kernel, workdir):
    """The compiler in its own process, as a caller runs it: the failure.json it left, or None."""
    subprocess.run([sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"), kernel,
                    "--workdir", workdir], capture_output=True, text=True)
    return failure.read(workdir)


def _stale(workdir):
    """A failure.json an earlier run in `workdir` left."""
    failure.write(workdir, "binary", failure.SPAD_OVERFLOW, "over", usage=2, budget=1)


def _spad_overflow():
    with tempfile.TemporaryDirectory() as d:
        got = _compile(os.path.join(ROOT, "tests/cases/mechanism/tile/tile_spad_over_budget.py"), d)
    ok = (got is not None and (got["stage"], got["kind"]) == ("binary", "spad-overflow")
          and isinstance(got["usage"], int) and isinstance(got["budget"], int)
          and got["usage"] > got["budget"] > 0)
    return ok, got and {k: v for k, v in got.items() if k != "message"}


def _passing_run_leaves_none():
    with tempfile.TemporaryDirectory() as d:
        _stale(d)
        got = _compile(os.path.join(ROOT, "tests/cases/smoke/add.py"), d)
        built = os.path.exists(os.path.join(d, "kernel.json"))
    return got is None and built, {"failure": got, "kernel.json": built}


def _unloadable_file_leaves_none():
    with tempfile.TemporaryDirectory() as d:
        kernel = os.path.join(d, "unloadable.py")
        with open(kernel, "w") as f:
            f.write('raise ImportError("not a kernel file")\n')
        _stale(d)
        got = _compile(kernel, d)
    return got is None, {"failure": got}


def _kind_is_the_type():
    raised = {"spad": SpadOverflow("binary", "over", usage=3, budget=2),
              "spill": SpillNotAllowed("binary", "spilled"),
              "stage": StageError("lower", "refused", cmd="llc x.ll", artifact="x.ll"),
              "other": ValueError("not a stage's")}
    got = {}
    with tempfile.TemporaryDirectory() as d:
        ctx = Ctx(None, __file__, d)
        for what, e in raised.items():
            pipeline._write_failure(ctx, "transform", e)
            got[what] = {k: v for k, v in failure.read(d).items() if k not in ("format", "message")}
    return got == {
        "spad": {"stage": "binary", "kind": "spad-overflow", "usage": 3, "budget": 2},
        "spill": {"stage": "binary", "kind": "spill-not-allowed"},
        "stage": {"stage": "lower", "kind": "error", "cmd": "llc x.ll", "artifact": "x.ll"},
        "other": {"stage": "transform", "kind": "error"}}, got


CASES = {"kind_is_the_type": _kind_is_the_type,
         "spad_overflow_leaves_its_numbers": _spad_overflow,
         "passing_run_leaves_none": _passing_run_leaves_none,
         "unloadable_file_leaves_none": _unloadable_file_leaves_none}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:36} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
