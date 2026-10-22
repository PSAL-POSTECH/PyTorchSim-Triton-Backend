"""What the harness accepts as a kernel's declared stop: never an expect_ir count that differs, raised
where found or held until binary; and the vocabulary a kernel names its stop in loads no MLIR.
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

ADD = os.path.join(ROOT, "tests/cases/smoke/add.py")
LAST = "    make_inputs=make_inputs,\n)"


def _refusing_add(d, expect_ir):
    """smoke/add.py, said to be refused, with `expect_ir` its only way to stop."""
    with open(ADD) as f:
        src = f.read()
    assert src.count(LAST) == 1
    src = src.replace(LAST, "    make_inputs=make_inputs,\n    expect=\"refuse\", expect_why=\"nothing "
                      "refuses an add\",\n    expect_ir=%r,\n)" % (expect_ir,))
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
    return r.returncode != 0 and not any("AS EXPECTED" in l for l in said), {
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


CASES = {"expect_ir_mismatch_raised_is_not_a_stop": _raised_where_found,
         "expect_ir_mismatch_held_is_not_a_stop": _held_until_binary,
         "stop_vocabulary_loads_no_mlir": _stop_vocabulary_loads_no_mlir}


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
