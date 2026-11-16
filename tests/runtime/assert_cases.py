"""A tl.device_assert stops the program on Spike with its message, a scalar one and a tensor one (each tile's
lanes reduced to one i1 first). Each case compiles tests/cases/mechanism/ops/ops_assert.py with one condition
made false and reads the message in what Spike printed; the unchanged kernel runs to its end, and the binary
gem5 times, which runs over data no one reads, holds none.
    python3 tests/runtime/assert_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("PYTORCHSIM_COMPILE_TARGET", suite_target.NAME)
KERNEL = os.path.join(ROOT, "tests/cases/mechanism/ops/ops_assert.py")


def _run(replace=None, workdir=None, flags=()):
    """(exit code, output) of the compile and Spike run of ops_assert.py, `replace` = (old, new) applied to it."""
    with open(KERNEL) as f:
        text = f.read()
    if replace:
        assert replace[0] in text, replace
        text = text.replace(*replace)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ops_assert.py")
        with open(path, "w") as f:
            f.write(text.replace("os.path.dirname(os.path.dirname(os.path.dirname(\n    os.path.dirname(os.path.abspath(__file__)))))",
                                 repr(ROOT)))
        done = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), path,
                               "--workdir", workdir or os.path.join(d, "w"), *flags], capture_output=True, text=True, cwd=ROOT)
    return done.returncode, done.stdout + done.stderr


def _holding_asserts_run_to_the_end():
    code, out = _run()
    return code == 0 and "ALL STAGES PASSED" in out, out[-300:]


def _a_false_scalar_assert_stops_with_its_message():
    code, out = _run(('tl.device_assert(n > 0, "n must be positive")', 'tl.device_assert(n < 0, "n must be positive")'))
    return code != 0 and "Assertion `n must be positive` failed" in out and "[3/4] spike                   FAIL" in out, out[-400:]


def _a_false_tensor_assert_stops_with_its_message():
    code, out = _run(('tl.device_assert(x == x, "no nans")', 'tl.device_assert(x > 1000.0, "no nans")'))
    return code != 0 and "Assertion `no nans` failed" in out and "[3/4] spike                   FAIL" in out, out[-400:]


def _several_tensor_asserts_of_two_axes_run_to_the_end():
    """Several asserts of two axes, each reduced by pytorchsim-reduce-tensor-assert to its own scalar assert;
    the GPT-2 redefinition of %a came from the text edit this pass replaced, and ops_assert_tensors.py is its shape."""
    code, out = _run(('tl.device_assert(x == x, "no nans")', 'tl.device_assert(x == x, "no nans")\n'
                      '        y = tl.reshape(x, (8, 16))\n'
                      '        tl.device_assert(y == y, "two axes")\n'
                      '        tl.device_assert(y < 1000.0, "small")\n'
                      '        for i in range(2):\n'
                      '            tl.device_assert(y + i < 1000.0, "in a loop")'))
    return code == 0 and "ALL STAGES PASSED" in out, out[-400:]


def _the_timed_binary_holds_no_assert():
    with tempfile.TemporaryDirectory() as w:
        code, out = _run(workdir=w, flags=("--tog",))
        text = {n: open(os.path.join(w, n)).read() for n in ("10-lowered.mlir", "10-sample-lowered.mlir")}
    return code == 0 and "assert_msg" in text["10-lowered.mlir"] and "assert_msg" not in text["10-sample-lowered.mlir"], out[-300:]


def main():
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("_") and name.endswith(("end", "message", "assert")):
            ok, got = fn()
            failed += not ok
            print(f"{name.strip('_'):60} {'PASS' if ok else 'FAIL'}  {'' if ok else got}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
