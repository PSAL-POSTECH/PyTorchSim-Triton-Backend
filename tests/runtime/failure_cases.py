"""failure.json as a caller reads it (the refusal's type names its kind, none after a pass), and what the
harness takes as a declared stop: the declared Refusal at the declared step, never an expect_ir count or
a past run's record; a non-pass kernel whose every step passes fails.
    python3 tests/runtime/failure_cases.py        exit 0 iff every case gets what it names"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run_checks  # noqa: E402
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


ADD = os.path.join(ROOT, "tests/cases/smoke/add.py")
LAST = "    make_inputs=make_inputs,\n)"


def _refusing_add(d, expect_ir, stop):
    """smoke/add.py, said to be refused at `stop`, with `expect_ir` its only way to stop."""
    with open(ADD) as f:
        src = f.read()
    assert src.count(LAST) == 1
    src = src.replace(LAST, "    make_inputs=make_inputs,\n    expect=\"refuse\", expect_why=\"nothing "
                      "refuses an add\",\n    expect_stop=%s,\n"
                      "    expect_ir=%r,\n)" % (stop, expect_ir))
    src = src.replace("import KernelSpec, Arg, IN, OUT\n", "import KernelSpec, Arg, IN, OUT\n"
                      "from pytorchsim_triton_compiler.contract.stop import Reason, Stop\n")
    src = src.replace("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
                      "sys.path.insert(0, %r)" % ROOT)
    path = os.path.join(d, "refusing_add.py")
    with open(path, "w") as f:
        f.write(src)
    return path


def _not_a_stop(expect_ir, stop):
    with tempfile.TemporaryDirectory() as d:
        kernel = _refusing_add(d, expect_ir, stop)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"), kernel,
                            "--workdir", os.path.join(d, "w")], capture_output=True, text=True)
    said = [l for l in r.stdout.splitlines() if "AS EXPECTED" in l or "expect_ir" in l]
    reached = any("expect_ir says" in l for l in said)
    return r.returncode != 0 and reached and not any("AS EXPECTED" in l for l in said), {
        "rc": r.returncode, "said": said[:3]}


def _raised_where_found():
    """Breaks if _stops_as_declared takes an ExpectIrMismatch, a RuntimeError, for a stop naming that class."""
    return _not_a_stop({"07-regularized.mlir": {"func.func": 99}},
                       'Stop("triton-compile", "builtins.RuntimeError")')


def _held_until_binary():
    return _not_a_stop({"08-transformed.mlir": {"func.func": 99}},
                       'Stop("transform", Reason.DTYPE_NOT_RUN)')


def _judged(stop, step, e):
    from types import SimpleNamespace
    from pytorchsim_triton_compiler import pipeline
    return pipeline._stops_as_declared(SimpleNamespace(expect_stop=stop), step, e)


def _refusal(reason_name):
    from pytorchsim_triton_compiler.contract.stop import Reason, Refusal

    class Declared(NotImplementedError, Refusal):
        reason = Reason[reason_name]
    return Declared("stopped")


def _wrapped(inner):
    from pytorchsim_triton_compiler.stage import StageError
    try:
        try:
            raise inner
        except Exception as e:
            raise StageError("transform", str(e)) from e
    except StageError as outer:
        return outer


def _stop_is_step_and_reason():
    """Only the declared Refusal at the declared step, found through __cause__, is the stop."""
    from pytorchsim_triton_compiler.contract.stop import Reason, Stop
    stop = Stop("transform", Reason.DTYPE_NOT_RUN)
    got = {"declared_through_its_cause": _judged(stop, "transform", _wrapped(_refusal("DTYPE_NOT_RUN"))),
           "unrelated_error": _judged(stop, "transform", _wrapped(KeyError("k"))),
           "generic_not_implemented": _judged(stop, "transform", NotImplementedError("n")),
           "another_reason": _judged(stop, "transform", _refusal("FILL_VARIES")),
           "another_step": _judged(stop, "lower", _refusal("DTYPE_NOT_RUN"))}
    return got == {"declared_through_its_cause": True, "unrelated_error": False,
                   "generic_not_implemented": False, "another_reason": False,
                   "another_step": False}, got


def _open_names_its_exception():
    from pytorchsim_triton_compiler.contract.stop import Stop
    stop = Stop("transform", "builtins.IndexError")
    got = {"IndexError": _judged(stop, "transform", IndexError("i")),
           "KeyError": _judged(stop, "transform", KeyError("k"))}
    return got == {"IndexError": True, "KeyError": False}, got


def _non_pass_declares_its_stop():
    from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec
    from pytorchsim_triton_compiler.contract.stop import Reason, Stop
    base = dict(name="k", kernel=None, signature={}, constexprs={}, args=[], grid=(1,),
                reference=None, make_inputs=None)
    got = {}
    for what, kw in (("refuse_without_stop", dict(expect="refuse", expect_why="w")),
                     ("pass_with_stop", dict(expect_stop=Stop("lower", Reason.SPAD_OVERFLOW))),
                     ("refuse_with_stop", dict(expect="refuse", expect_why="w",
                                               expect_stop=Stop("lower", Reason.SPAD_OVERFLOW)))):
        try:
            KernelSpec(**base, **kw)
            got[what] = "accepted"
        except ValueError:
            got[what] = "refused"
    return got == {"refuse_without_stop": "refused", "pass_with_stop": "refused",
                   "refuse_with_stop": "accepted"}, got


def _compiled(kernel, workdir):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"), kernel,
                        "--workdir", workdir], capture_output=True, text=True)
    return r.returncode, [l for l in r.stdout.splitlines() if "AS EXPECTED" in l or " FAIL " in l]


def _a_stale_record_is_not_read():
    """A record an earlier run left does not name the reason of a C++ stop in this one."""
    from pytorchsim_triton_compiler.contract import stop
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "w")
        os.makedirs(w)
        stop.record(w, stop.Refused(stop.Reason.ATOMICS, "an earlier run's"))
        rc, said = _compiled(os.path.join(
            ROOT, "tests/cases/mechanism/vcix/vcix_layout_the_machine_cannot_take.py"), w)
    return rc == 0 and any("REFUSED AS EXPECTED" in l for l in said), {"rc": rc, "said": said[:2]}


def _lifted(expect, to=None):
    """smoke/add.py declared `expect` with a stop it never reaches: every step passes; with `to`, the
    run ends there, before the declared step."""
    with open(ADD) as f:
        src = f.read()
    src = src.replace(LAST, "    make_inputs=make_inputs,\n    expect=%r, expect_why=\"it used to stop\",\n"
                      "    expect_stop=Stop(\"transform\", Reason.DTYPE_NOT_RUN),\n)" % expect)
    src = src.replace("import KernelSpec, Arg, IN, OUT\n", "import KernelSpec, Arg, IN, OUT\n"
                      "from pytorchsim_triton_compiler.contract.stop import Reason, Stop\n")
    src = src.replace("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
                      "sys.path.insert(0, %r)" % ROOT)
    with tempfile.TemporaryDirectory() as d:
        kernel = os.path.join(d, "lifted.py")
        with open(kernel, "w") as f:
            f.write(src)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"), kernel,
                            "--workdir", os.path.join(d, "w")] + (["--to", to] if to else []),
                           capture_output=True, text=True)
    said = [l for l in r.stdout.splitlines()
            if "HAS LIFTED" in l or "HAS CLOSED" in l or "NOT EXERCISED" in l]
    if to:
        return r.returncode == 0 and any("NOT EXERCISED" in l for l in said), {
            "rc": r.returncode, "said": said}
    return r.returncode != 0 and len(said) == 1, {"rc": r.returncode, "said": said}


CASES = {"kind_is_the_type": _kind_is_the_type,
         "spad_overflow_leaves_its_numbers": _spad_overflow,
         "passing_run_leaves_none": _passing_run_leaves_none,
         "unloadable_file_leaves_none": _unloadable_file_leaves_none,
         "expect_ir_mismatch_raised_is_not_a_stop": _raised_where_found,
         "expect_ir_mismatch_held_is_not_a_stop": _held_until_binary,
         "stop_is_step_and_reason": _stop_is_step_and_reason,
         "open_names_its_exception": _open_names_its_exception,
         "non_pass_declares_its_stop": _non_pass_declares_its_stop,
         "a_stale_record_is_not_read": _a_stale_record_is_not_read,
         "blocked_whose_every_step_passes_fails": lambda: _lifted("blocked"),
         "open_whose_every_step_passes_fails": lambda: _lifted("open"),
         "a_run_that_ends_before_the_stop_does_not_judge_it": lambda: _lifted("blocked", to="triton-compile")}


def main():
    return run_checks(CASES)


if __name__ == "__main__":
    sys.exit(main())
