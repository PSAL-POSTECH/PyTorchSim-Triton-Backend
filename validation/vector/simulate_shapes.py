#!/usr/bin/env python3
"""Time this suite's workloads in PyTorchSim's kernel bundle on --target with TOGSim exactly as validation/gemm does
(its runner, run here with program, R and C naming a workload); results.json holds the machine and per workload the
exit code, cycles and what it ran on; exit 1 if any failed."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from suite_module import load  # noqa: E402
KEYS = ("program", "R", "C")


sim = load("gemm", "simulate_shapes")

#: The programs the report compares with a kernel measured at the bundle kernel's block: the measured rows, each
#: at the block it was measured at (block_rows), and the columns that are the workload's R and C.
MEASURED_AT_BLOCK = {
    "softmax": (os.path.join(HERE, "tpu_v6e_softmax_pallas.csv"), ("R", "C")),
    "topk": (os.path.join(HERE, "tpu_v6e_topk_pallas.csv"), ("T", "E")),
}


def kernel_block_rows(spec_path):
    """The rows one program of a bundled row kernel (softmax, topk) takes: its XBLOCK constexpr, else its xnumel
    (one program over every row), read off the kernel's SPEC."""
    spec = importlib.util.spec_from_file_location("row_kernel_spec", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = mod.SPEC.constexprs
    if "XBLOCK" in c:
        return int(c["XBLOCK"])
    if "xnumel" in c:
        return int(c["xnumel"])
    raise SystemExit(f"{spec_path}: a row kernel with neither XBLOCK nor xnumel among its constexprs")


def check_blocks(bundle, measured=None):
    """Each softmax and topk workload in the bundle is measured at its own kernel's block, in every measured row
    of it, a block past the workload's R rows counting as R (one program over every row); refused, naming every
    one that is not, before anything runs."""
    import csv
    import json
    measured = measured or MEASURED_AT_BLOCK
    rows = {}
    for program, (path, (rc, cc)) in measured.items():
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.setdefault((program, int(r[rc]), int(r[cc])), []).append(int(r["block_rows"]))
    off = []
    for name in sorted(os.listdir(bundle)):
        path = os.path.join(bundle, name, "manifest.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            m = json.load(fh)
        if m["suite"] != os.path.basename(HERE) or m["program"] not in measured:
            continue
        got = rows.get((m["program"], m["R"], m["C"]), [None])
        for la in m["launches"]:
            want = kernel_block_rows(os.path.join(bundle, name, la["dir"], la["kernel"] + "_spec.py"))
            off += [f"{name}: kernel block {want}, measured at {g}" for g in sorted(set(got), key=str)
                    if g is None or min(g, m["R"]) != min(want, m["R"])]
    if off:
        raise SystemExit(f"{len(off)} workloads are not measured at their kernel's block "
                         f"({', '.join(os.path.basename(p) for p, _ in measured.values())}): " + "; ".join(off[:5]))


def bundle_arg(argv):
    """The --bundle the runner will read, or its default."""
    for i, a in enumerate(argv):
        if a == "--bundle" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if a.startswith("--bundle="):
            return os.path.abspath(a.split("=", 1)[1])
    return os.path.abspath(sim.BUNDLE)


if __name__ == "__main__":
    if "--merge" not in sys.argv:
        check_blocks(bundle_arg(sys.argv))
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__))
