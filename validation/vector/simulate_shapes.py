#!/usr/bin/env python3
"""Time this suite's workloads in PyTorchSim's kernel bundle on --target with TOGSim exactly as validation/gemm does
(its runner, run here with program, R and C naming a workload); results.json holds the machine and per workload the
exit code, cycles and what it ran on; exit 1 if any failed."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("program", "R", "C")


def gemm_runner():
    """validation/gemm/simulate_shapes.py as a module of its own name, so it never shadows this file."""
    path = os.path.join(os.path.dirname(HERE), "gemm", "simulate_shapes.py")
    spec = importlib.util.spec_from_file_location("gemm_simulate_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sim = gemm_runner()

#: The measured softmax rows the report compares with, each at the block it was measured at (block_rows).
SOFTMAX_MEASURED = os.path.join(HERE, "tpu_v6e_softmax_pallas.csv")


def kernel_block_rows(spec_path):
    """The rows one program of a bundled softmax kernel takes: its XBLOCK constexpr, else its xnumel (one program
    over every row), read off the kernel's SPEC."""
    spec = importlib.util.spec_from_file_location("softmax_spec", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = mod.SPEC.constexprs
    if "XBLOCK" in c:
        return int(c["XBLOCK"])
    if "xnumel" in c:
        return int(c["xnumel"])
    raise SystemExit(f"{spec_path}: a softmax kernel with neither XBLOCK nor xnumel among its constexprs")


def check_softmax_blocks(bundle, measured=SOFTMAX_MEASURED):
    """Each softmax workload in the bundle is measured at its own kernel's block; refused, naming every one that
    is not, before anything runs."""
    import csv
    import json
    with open(measured, newline="") as fh:
        rows = {(int(r["R"]), int(r["C"])): int(r["block_rows"]) for r in csv.DictReader(fh)}
    off = []
    for name in sorted(os.listdir(bundle)):
        path = os.path.join(bundle, name, "manifest.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            m = json.load(fh)
        if m["suite"] != os.path.basename(HERE) or m["program"] != "softmax":
            continue
        got = rows.get((m["R"], m["C"]))
        for la in m["launches"]:
            want = kernel_block_rows(os.path.join(bundle, name, la["dir"], la["kernel"] + "_spec.py"))
            if got != want:
                off.append(f"{name}: kernel block {want}, measured at {got}")
    if off:
        raise SystemExit(f"{len(off)} softmax workloads are not measured at their kernel's block "
                         f"({os.path.basename(measured)}): " + "; ".join(off[:5]))


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
        check_softmax_blocks(bundle_arg(sys.argv))
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__))
