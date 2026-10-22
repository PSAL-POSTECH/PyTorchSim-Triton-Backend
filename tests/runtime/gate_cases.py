"""What the harness takes as a kernel's declared stop: the declared Refusal at the declared step, never
an expect_ir count (raised where found or held until binary); the vocabulary for it loads no MLIR.
    python3 tests/runtime/gate_cases.py        exit 0 iff every case gets what it names"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
sys.path.insert(0, ROOT)

ADD = os.path.join(ROOT, "tests/cases/smoke/add.py")
LAST = "    make_inputs=make_inputs,\n)"


def _refusing_add(d, expect_ir):
    """smoke/add.py, said to be refused, with `expect_ir` its only way to stop."""
    with open(ADD) as f:
        src = f.read()
    assert src.count(LAST) == 1
    src = src.replace(LAST, "    make_inputs=make_inputs,\n    expect=\"refuse\", expect_why=\"nothing "
                      "refuses an add\",\n    expect_stop=Stop(\"transform\", Reason.DTYPE_NOT_RUN),\n"
                      "    expect_ir=%r,\n)" % (expect_ir,))
    src = src.replace("import KernelSpec, Arg, IN, OUT\n", "import KernelSpec, Arg, IN, OUT\n"
                      "from pytorchsim_triton_compiler.contract.stop import Reason, Stop\n")
    src = src.replace("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
                      "sys.path.insert(0, %r)" % ROOT)
    path = os.path.join(d, "refusing_add.py")
    with open(path, "w") as f:
        f.write(src)
    return path


def _not_a_stop(expect_ir):
    with tempfile.TemporaryDirectory() as d:
        kernel = _refusing_add(d, expect_ir)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"), kernel,
                            "--workdir", os.path.join(d, "w")], capture_output=True, text=True)
    said = [l for l in r.stdout.splitlines() if "AS EXPECTED" in l or "expect_ir" in l]
    reached = any("expect_ir says" in l for l in said)
    return r.returncode != 0 and reached and not any("AS EXPECTED" in l for l in said), {
        "rc": r.returncode, "said": said[:3]}


def _raised_where_found():
    return _not_a_stop({"07-regularized.mlir": {"func.func": 99}})


def _held_until_binary():
    return _not_a_stop({"08-transformed.mlir": {"func.func": 99}})


def _stop_vocabulary_loads_no_mlir():
    """A kernel file names its stop from contract.stop in the process that runs Triton, which must
    not load this repo's MLIR bindings."""
    r = subprocess.run([sys.executable, "-c", "import sys; from pytorchsim_triton_compiler.contract "
                        "import stop; print(stop.Stop('lower', stop.Reason.SPAD_OVERFLOW).step, "
                        "sorted(m for m in sys.modules if m == 'mlir' or m.startswith('mlir.')))"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.split() == ["lower", "[]"], {
        "rc": r.returncode, "out": r.stdout.strip(), "err": r.stderr.strip()[-200:]}


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


CASES = {"expect_ir_mismatch_raised_is_not_a_stop": _raised_where_found,
         "expect_ir_mismatch_held_is_not_a_stop": _held_until_binary,
         "stop_vocabulary_loads_no_mlir": _stop_vocabulary_loads_no_mlir,
         "stop_is_step_and_reason": _stop_is_step_and_reason,
         "open_names_its_exception": _open_names_its_exception,
         "non_pass_declares_its_stop": _non_pass_declares_its_stop}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:42} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
