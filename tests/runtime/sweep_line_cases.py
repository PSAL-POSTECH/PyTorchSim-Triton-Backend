"""The line tests/run.py gives a failing kernel: the stage its banner names and the stage log's own
words -- for an expect_ir count, the counts under the headline, not "failed with no diagnostic".
    python3 tests/runtime/sweep_line_cases.py        exit 0 iff every case gets what it names"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run  # noqa: E402
from pytorchsim_triton_compiler import pipeline  # noqa: E402

EUP = os.path.join(ROOT, "tests/cases/mechanism/ops/ops_exp_on_the_eup_pipelined.py")
KERNEL = '''import importlib.util
spec = importlib.util.spec_from_file_location("eup", {path!r})
eup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eup)
kernel, make_inputs, reference = eup.kernel, eup.make_inputs, eup.reference
SPEC = eup.SPEC
SPEC.expect_ir = {{"09-custom.mlir": dict(eup.EUP_COUNTS, **{{{op!r}: {want}}})}}
'''


def _expect_ir_count_differs():
    """ops_exp_on_the_eup_pipelined declaring one more torchsim.eup_pipeline than lower leaves."""
    op = "torchsim.eup_pipeline"
    with tempfile.TemporaryDirectory() as d:
        import importlib.util
        spec = importlib.util.spec_from_file_location("eup", EUP)
        eup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eup)
        have = eup.EUP_COUNTS[op]
        kernel = os.path.join(d, "eup_count_differs.py")
        with open(kernel, "w") as f:
            f.write(KERNEL.format(path=EUP, op=op, want=have + 1))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                            "--workdir", os.path.join(d, "w")], cwd=ROOT,
                           capture_output=True, text=True)
    got = run.why_failed(r.stdout + r.stderr, r.returncode)
    want = (f"{pipeline.stage_of('lower')}:lower: 09-custom.mlir does not say what the kernel "
            f"declares: {op} is {have}, expect_ir says {have + 1}")
    return r.returncode != 0 and got == want, {"got": got, "want": want}


CASES = {"expect_ir_count_differs_says_the_counts": _expect_ir_count_differs}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:40} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:400]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
