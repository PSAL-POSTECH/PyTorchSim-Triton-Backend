#!/usr/bin/env python3
"""The vector runner times softmax only against rows measured at its kernels' blocks: every softmax workload of the
bundle has a measured row whose block_rows is its kernel's XBLOCK (or xnumel), and a row at another block is
refused, naming the workload, before anything runs.
   python3 validation/vector/simulate_cases.py        exit 0 iff every case holds"""
import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulate_shapes as vec  # noqa: E402

BUNDLE = os.path.abspath(vec.sim.BUNDLE)


def _spec(name):
    d = os.path.join(BUNDLE, name)
    sub = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))][0]
    return os.path.join(d, sub, [f for f in os.listdir(os.path.join(d, sub)) if f.endswith("_spec.py")][0])


def _blocks_read_off_the_kernels():
    """A kernel of 128 rows a program says XBLOCK 128; one of a single row says xnumel 1 and no XBLOCK."""
    got = (vec.kernel_block_rows(_spec("vector_softmax_1024x1024_bf16")),
           vec.kernel_block_rows(_spec("vector_softmax_1x1024_bf16")))
    return got == (128, 1), got


def _the_measured_rows_match_the_bundle():
    try:
        vec.check_softmax_blocks(BUNDLE)
        return True, "every softmax workload at its kernel's block"
    except SystemExit as e:
        return False, str(e)[:300]


def _another_block_is_refused():
    with open(vec.SOFTMAX_MEASURED, newline="") as fh:
        rd = csv.DictReader(fh)
        rows, cols = list(rd), rd.fieldnames
    for r in rows:
        if (r["R"], r["C"]) == ("1024", "1024"):
            r["block_rows"] = "1024"
    path = os.path.join(tempfile.mkdtemp(prefix="vecsim-"), "pallas.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    try:
        vec.check_softmax_blocks(BUNDLE, path)
        return False, "not refused"
    except SystemExit as e:
        msg = str(e)
        return "vector_softmax_1024x1024_bf16: kernel block 128, measured at 1024" in msg, msg[:300]


CASES = {
    "blocks are read off the kernels' constexprs": _blocks_read_off_the_kernels,
    "the measured softmax rows are at the bundle's blocks": _the_measured_rows_match_the_bundle,
    "a row at another block is refused, naming it": _another_block_is_refused,
}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:52} {'PASS' if ok else 'FAIL'}  {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
