#!/usr/bin/env python3
"""The vector runner times softmax and topk only against rows measured at their kernels' blocks: every such workload
of the bundle has measured rows whose block_rows is its kernel's XBLOCK (or xnumel), and a row at another block is
refused, naming the workload, before anything runs.
   python3 validation/vector/simulate_cases.py        exit 0 iff every case holds"""
import csv
import json
import os
import shutil
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
        vec.check_blocks(BUNDLE)
        return True, "every softmax and topk workload at its kernel's block"
    except SystemExit as e:
        return False, str(e)[:300]


def _moved(program, key, block):
    """MEASURED_AT_BLOCK with `program`'s rows at `key` (its R, C columns' values) measured at `block`."""
    path, cols = vec.MEASURED_AT_BLOCK[program]
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        rows, names = list(rd), rd.fieldnames
    for r in rows:
        if (r[cols[0]], r[cols[1]]) == key:
            r["block_rows"] = block
    moved = os.path.join(tempfile.mkdtemp(prefix="vecsim-"), os.path.basename(path))
    with open(moved, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    return dict(vec.MEASURED_AT_BLOCK, **{program: (moved, cols)})


def _refused(bundle, measured, want):
    try:
        vec.check_blocks(bundle, measured)
        return False, "not refused"
    except SystemExit as e:
        return want in str(e), str(e)[:300]


def _another_block_is_refused():
    return _refused(BUNDLE, _moved("softmax", ("1024", "1024"), "1024"),
                    "vector_softmax_1024x1024_bf16: kernel block 128, measured at 1024")


def _a_topk_row_at_another_block_is_refused():
    """A bundle whose topk 128x128 workload launches a kernel of 128 rows a program (the 1024x1024 softmax's)."""
    bundle = tempfile.mkdtemp(prefix="vecsim-bundle-")
    name = "vector_topk_128x128x8_f32"
    shutil.copytree(os.path.join(BUNDLE, "vector_softmax_1024x1024_bf16"), os.path.join(bundle, name))
    with open(os.path.join(bundle, name, "manifest.json")) as fh:
        m = json.load(fh)
    m.update(workload=name, program="topk", R=128, C=128, K=8, dtype="f32")
    with open(os.path.join(bundle, name, "manifest.json"), "w") as fh:
        json.dump(m, fh)
    held, _ = _refused(bundle, vec.MEASURED_AT_BLOCK, "")
    if held:
        return False, "the measured topk rows at block 128 were refused"
    return _refused(bundle, _moved("topk", ("128", "128"), "64"), f"{name}: kernel block 128, measured at 64")


CASES = {
    "blocks are read off the kernels' constexprs": _blocks_read_off_the_kernels,
    "the measured softmax and topk rows are at the bundle's blocks": _the_measured_rows_match_the_bundle,
    "a row at another block is refused, naming it": _another_block_is_refused,
    "a topk row at another block is refused, naming it": _a_topk_row_at_another_block_is_refused,
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
