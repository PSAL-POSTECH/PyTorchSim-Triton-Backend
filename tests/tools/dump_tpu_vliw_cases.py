"""tools/dump_tpu_vliw.py refuses, in one line and before it writes anything, an output directory
that is in use, a Python without jax + libtpu, and sizes its workload does not take. No libtpu needed.
    python3 tests/tools/dump_tpu_vliw_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "tools", "dump_tpu_vliw.py")


def _run(*args):
    """The tool under a Python that sees no site-packages (-I -S), so jax is absent whoever runs this."""
    return subprocess.run([sys.executable, "-I", "-S", TOOL, *args], capture_output=True, text=True)


def _refused(r, *words):
    """Exit status 1 and exactly one line on stderr, holding every one of `words`."""
    assert r.returncode == 1, (r.returncode, r.stderr)
    assert len(r.stderr.strip().splitlines()) == 1, r.stderr
    for w in words:
        assert w in r.stderr, r.stderr


def a_directory_holding_a_file_is_refused_and_left_alone():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "kept"), "w").close()
        _refused(_run(d, "matmul", "1", "4", "4"), d, "not an empty directory")
        assert os.listdir(d) == ["kept"]


def a_file_as_the_output_directory_is_refused():
    with tempfile.NamedTemporaryFile() as f:
        _refused(_run(f.name, "matmul", "1", "4", "4"), "not an empty directory")


def a_python_without_jax_is_refused_before_the_directory_is_made():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out")
        _refused(_run(out, "matmul", "1", "4", "4"), "jax", "libtpu", "setup/toolchain.sh jaxtpu")
        assert not os.path.exists(out)


def an_empty_directory_gets_as_far_as_the_missing_jax():
    with tempfile.TemporaryDirectory() as d:
        _refused(_run(d, "matmul", "1", "4", "4"), "jax")
        assert os.listdir(d) == []


def a_matmul_of_two_sizes_is_refused():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out")
        _refused(_run(out, "matmul", "4", "4"), "M K N")
        assert not os.path.exists(out)


def an_unknown_workload_is_a_usage_error():
    with tempfile.TemporaryDirectory() as d:
        r = _run(os.path.join(d, "out"), "conv", "4", "4", "4")
        assert r.returncode == 2 and "invalid choice" in r.stderr, (r.returncode, r.stderr)


CASES = [a_directory_holding_a_file_is_refused_and_left_alone, a_file_as_the_output_directory_is_refused,
         a_python_without_jax_is_refused_before_the_directory_is_made,
         an_empty_directory_gets_as_far_as_the_missing_jax, a_matmul_of_two_sizes_is_refused,
         an_unknown_workload_is_a_usage_error]


def main():
    return run(CASES)


if __name__ == "__main__":
    sys.exit(main())
