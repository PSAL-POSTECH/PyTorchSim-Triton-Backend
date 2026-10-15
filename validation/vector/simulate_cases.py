"""kernels/ and shapes.csv are what emit_kernels.py writes, its blocks are XLA's, a run is named by program, R and C,
and the report joins, counts FLOP and shows failures; a stub stands in for torchsim-compile.
   python3 validation/vector/simulate_cases.py        exit 0 iff every case holds"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_kernels as ek  # noqa: E402
import emit_report  # noqa: E402
from simulate_shapes import KEYS, sim  # noqa: E402

STUB = '''import json, os, sys, yaml
a = sys.argv; wd = a[a.index("--workdir") + 1]; os.makedirs(wd, exist_ok=True)
if a[a.index("--to") + 1] == "binary":
    if "STUB_FAIL" in os.environ and os.environ["STUB_FAIL"] in a[1]:
        sys.exit(3)
    path = os.environ["TORCHSIM_COMPILE_TARGET"]
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    said = dict(raw, file=path, name=os.path.splitext(os.path.basename(path))[0])
    json.dump({"target": said}, open(os.path.join(wd, "kernel.json"), "w"))
else:
    json.dump({"total_cycles": 3502, "config": os.environ["STUB_CONFIG"],
               "model": os.environ["STUB_MODEL"]}, open(os.path.join(wd, "togsim.json"), "w"))
'''


def _shapes():
    import csv
    with open(os.path.join(HERE, "shapes.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def _kernels_are_generated():
    """Every measured shape has its kernel, and each file and shapes.csv row is the generator's, byte for byte."""
    on_disk, stale = _shapes(), []
    want = [ek.render(*s) for s in ek.measured_shapes()]
    for (row, text), got in zip(want, on_disk):
        with open(os.path.join(HERE, "kernels", row["kernel"] + ".py")) as fh:
            if fh.read() != text or {k: str(v) for k, v in row.items()} != got:
                stale.append(row["kernel"])
    files = len([f for f in os.listdir(os.path.join(HERE, "kernels")) if f.endswith(".py")])
    ok = len(want) == len(on_disk) == files and not stale
    return ok, (len(want), len(on_disk), files, stale[:5])


def _blocks_are_xlas():
    """The block of every program fits 2^20 elements, and the rule picks the form the shape names."""
    got = {k: (ek.render(*k)[0]["form"], ek.render(*k)[0]["BM"], ek.render(*k)[0]["BC"]) for k in (
        ("add1", 3072, 1024), ("softmax", 4096, 768), ("softmax", 1, 6144), ("softmax", 8, 384),
        ("max", 16384, 8192), ("softmax", 32768, 128))}
    want = {("add1", 3072, 1024): ("rows", 1024, 1024), ("softmax", 4096, 768): ("split", 1024, 512),
            ("softmax", 1, 6144): ("rows", 1, 8192), ("softmax", 8, 384): ("chunks", 8, 4),
            ("max", 16384, 8192): ("walk", 128, 8192), ("softmax", 32768, 128): ("rows", 8192, 128)}
    def elements(s):
        BM, BC, C = int(s["BM"]), int(s["BC"]), int(s["C"])
        return {"split": BM * C, "chunks": ek.LANES * BM * BC}.get(s["form"], BM * BC)
    over = [s["kernel"] for s in _shapes() if elements(s) > ek.BUDGET]
    return got == want and not over, (got, over)


def _run(shape, fail=None):
    """One shape of this set through the stub on the suite's machine."""
    tmp = tempfile.mkdtemp(prefix="vecsim-")
    stub = os.path.join(tmp, "stub.py")
    with open(stub, "w") as fh:
        fh.write(STUB)
    os.makedirs(os.path.join(tmp, "logs"))
    machine = sim.plan_machine(sim.TARGET)
    os.environ.update(STUB_CONFIG=machine["togsim_config"], STUB_MODEL=machine["model"])
    os.environ.pop("STUB_FAIL", None)
    if fail:
        os.environ["STUB_FAIL"] = fail
    driver, sim.DRIVER = sim.DRIVER, stub
    try:
        got = sim.simulate_shape(shape, os.path.join(tmp, "work"), os.path.join(tmp, "logs"), 60, machine,
                                 os.path.join(HERE, "kernels"), KEYS)
    finally:
        sim.DRIVER = driver
    return machine, got


def _results(machine, done):
    path = os.path.join(tempfile.mkdtemp(prefix="vecsim-"), "results.json")

    class Args:
        jobs, timeout = 1, 60
    sim.write_results(path, Args, machine, done, len(done), KEYS)
    with open(path) as fh:
        return json.load(fh)


def _a_run_is_named_by_program_r_c():
    by = {s["kernel"]: s for s in _shapes()}
    machine, a = _run(by["softmax_4096x768"])
    _, b = _run(by["add1_8x128"])
    res = _results(machine, [a, b])
    names = [(s["program"], s["R"], s["C"], s["kernel"]) for s in res["shapes"]]
    want = [("add1", 8, 128, "add1_8x128"), ("softmax", 4096, 768, "softmax_4096x768")]
    return names == want and a["rc"] == 0 and a["us"] == 3502 / machine["core_freq_mhz"], names


def _report_joins_counts_and_shows_failures():
    """softmax counts 5 FLOP per element over 4·R·C bytes; a failed kernel stays a row with no ratio."""
    by = {s["kernel"]: s for s in _shapes()}
    machine, a = _run(by["softmax_4096x768"])
    _, b = _run(by["max_256x128"], fail="max_256x128")
    res = _results(machine, [a, b])
    real = emit_report.measured(os.path.join(HERE, "tpu_v6e_softmax_torchxla.csv"),
                                os.path.join(HERE, "tpu_v6e_load_torchxla.csv"))
    shapes = os.path.join(tempfile.mkdtemp(prefix="vecsim-"), "shapes.csv")
    with open(os.path.join(HERE, "shapes.csv")) as src, open(shapes, "w") as dst:
        lines = src.read().splitlines()
        dst.write("\n".join([lines[0]] + [ln for ln in lines if ",softmax_4096x768," in ln or ",max_256x128," in ln]) + "\n")
    rows = emit_report.join_rows(shapes, real, res)
    sm = next(r for r in rows if r["program"] == "softmax")
    mx = next(r for r in rows if r["program"] == "max")
    page = emit_report.emit_html(rows, emit_report.count_chips(rows), {
        "commit": "0" * 40, "archive_commit": "23665c0", "date": "now",
        "ran_on": emit_report.gemm_report().ran_on(res["machine"])}, {"hbm_gbps": 1638.4}, res["machine"])
    md = emit_report.emit_summary(rows, emit_report.count_chips(rows), {"commit": "0" * 40, "ran_on": "x"})
    ok = (sm["flops"] == 5 * 4096 * 768 and sm["bytes"] == 4 * 4096 * 768 and sm["ratio"] is not None
          and mx["ratio"] is None and mx["failed_step"] == "binary" and "max_256x128" in md
          and "Failed (1)" in md and "blocked as XLA blocks" in page and "tpuv6e (vcix_tpuv6e)" in page
          and "23665c0" in page)
    return ok, (sm["flops"], sm["bytes"], mx["failed_step"], mx["rc"])


def _shards_partition_longest_first():
    """The two parts of this set are disjoint, cover it, and each runs largest R*C first, ties in key order."""
    shapes = _shapes()
    parts = [[s["kernel"] for s in sim.shard(shapes, k, 2, KEYS)] for k in (0, 1)]
    flat = parts[0] + parts[1]
    rc = {s["kernel"]: int(s["R"]) * int(s["C"]) for s in shapes}
    ordered = all(rc[a] >= rc[b] for p in parts for a, b in zip(p, p[1:]))
    ok = (sorted(flat) == sorted(rc) and len(set(flat)) == len(flat) and ordered
          and parts[0][0] == "add1_16384x8192" and parts[1][0] == "max_16384x8192")
    return ok, ([len(p) for p in parts], parts[0][:2], parts[1][:2])


CASES = {
    "kernels/ and shapes.csv are the generator's": _kernels_are_generated,
    "every block is XLA's: <= 2^20 elements": _blocks_are_xlas,
    "a run is named and sorted by program, R, C": _a_run_is_named_by_program_r_c,
    "the report counts FLOP and shows a failure": _report_joins_counts_and_shows_failures,
    "shards partition the set, largest first": _shards_partition_longest_first,
}


def main():
    os.environ.pop("TORCHSIM_COMPILE_VCIX_MODEL", None)
    sim.config.VCIX_MODEL = None
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:48} {'PASS' if ok else 'FAIL'}  {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
