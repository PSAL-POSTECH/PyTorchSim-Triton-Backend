"""togcheck takes a skipped sized DMA only where it computes the clamp empty: mask_negative_offset passes, and
its TOG with the load guarded off in every item (a skeleton mis-sizing the extent to <= 0) fails.
    python3 tests/runtime/togcheck_skip_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import tog_record  # noqa: E402

CASE = os.path.join(ROOT, "tests", "cases", "mechanism", "mask", "mask_negative_offset.py")


def _built(wd):
    """The case compiled with --tog into `wd`."""
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), CASE, "--workdir", wd,
                        "--tog"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"the build failed: {(p.stdout + p.stderr)[-600:]}")


def the_load_skipped_in_program_0_alone_passes(wd):
    """Breaks if togcheck rejects the fill program 0 reads (its clamp is empty there) or the skip itself."""
    problems, _ = tog_record.check(wd)
    assert problems == [], problems


def a_load_skipped_where_its_clamp_moves_elements_fails(wd):
    """Breaks if togcheck trusts the guard: program 1's clamp moves 128 elements, and a TOG that runs no load
    there (the guard made false) must be a problem, not a note and an excused read."""
    from pytorchsim_triton_compiler.togsim import lower_to_emitc as l2e
    cpp = os.path.join(wd, "tog.cpp")
    text = open(cpp).read()
    at = text.rfind("if (", 0, text.index("togsim::Load"))
    guard = text[at:text.index(")", at) + 1]
    text = text.replace(guard, "if (false)")
    open(cpp, "w").write(text)
    l2e.finish_so(l2e.start_so(cpp, os.path.join(wd, "tog.so")))
    problems, _ = tog_record.check(wd)
    assert any("runs no dma of dir=0 arg=0" in p for p in problems), problems


CASES = [the_load_skipped_in_program_0_alone_passes, a_load_skipped_where_its_clamp_moves_elements_fails]


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        wd = os.path.join(d, "k")
        _built(wd)
        for case in CASES:
            try:
                case(wd)
                print(f"PASS  {case.__name__}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:400]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
