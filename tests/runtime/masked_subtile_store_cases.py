"""mask_m_tail_subtile_store built with its TOG: the TOG stores the [256, 64] the subtile's clamp moves, and
togcheck takes it as the IR's [256, 128] sized at run time (build_skeleton.run_time_sized_axes, the rule both
read). python3 tests/runtime/masked_subtile_store_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import tog_record  # noqa: E402
from pytorchsim_triton_compiler.contract import kernel_object as ko  # noqa: E402

CASE = os.path.join(ROOT, "tests", "cases", "mechanism", "mask", "mask_m_tail_subtile_store.py")


def main():
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), CASE, "--workdir", d,
                            "--to", "binary", "--tog"], cwd=ROOT, capture_output=True, text=True)
        stores = [e["dims"] for e in tog_record.record(d, ko.shape_args(ko.read(d)))
                  if e["op"] == "dma" and e["dir"] == 1] if p.returncode == 0 else None
    cases = {"the build and its togcheck pass": (p.returncode == 0, (p.stdout + p.stderr)[-400:]),
             "the TOG stores the 64 tail rows of the 3 subtiles inside N": (stores == [[256, 64]] * 3, stores)}
    failed = 0
    for name, (ok, got) in cases.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": {got}"))
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
