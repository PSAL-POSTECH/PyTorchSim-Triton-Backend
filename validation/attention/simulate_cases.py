"""The kernels simulate_shapes generates are one per measured shape with blocks the policy allows, a run is named by
kind, Hq, Hkv and S, --ci times exactly the shapes under the cap, and the report counts the archive's FLOP and tells a
failure from a shape left out; a stub stands in for torchsim-compile.
   python3 validation/attention/simulate_cases.py        exit 0 iff every case holds"""
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

GEN = os.path.join(tempfile.mkdtemp(prefix="attnsim-"), "generated")
SHAPES = sim.generate(HERE, sim.TILING, GEN)
BY = {s["kernel"]: s for s in SHAPES}


def _stub():
    tmp = tempfile.mkdtemp(prefix="attnsim-")
    stub = os.path.join(tmp, "stub.py")
    with open(stub, "w") as fh:
        fh.write(STUB)
    machine = sim.plan_machine(sim.TARGET)
    os.environ.update(STUB_CONFIG=machine["togsim_config"], STUB_MODEL=machine["model"])
    os.environ.pop("STUB_FAIL", None)
    return tmp, stub, machine


def _one_kernel_per_measured_shape():
    """Every measured shape got its kernel and shapes.csv row, and the policy the run used sits beside them."""
    files = len([f for f in os.listdir(os.path.join(GEN, "kernels")) if f.endswith(".py")])
    want = len(ek.measured_shapes())
    return want == len(SHAPES) == files == 56 and os.path.exists(os.path.join(GEN, "tiling.py")), (want, len(SHAPES), files)


def _blocks_are_the_policys():
    """prefill BQ = BKV divides S and is splash's 1024 or STATIC_BQ's; decode BKV divides S, at most 32 pages, and
    GP is a power of two holding the group, at least MIN_DOT_M."""
    bad = []
    for s in SHAPES:
        S, BQ, BKV, GP, G = int(s["S"]), int(s["BQ"]), int(s["BKV"]), int(s["GP"]), int(s["Hq"]) // int(s["Hkv"])
        if s["kind"] == "prefill":
            ok = BQ == BKV and S % BQ == 0 and BQ == ek.tiling.STATIC_BQ.get(S, min(S, ek.tiling.SPLASH_BLOCK))
        else:
            ok = S % BKV == 0 and BKV <= 4096 and GP >= max(G, ek.tiling.MIN_DOT_M) and GP & (GP - 1) == 0
        if not ok:
            bad.append(s["kernel"])
    return not bad, bad


def _ci_is_the_cap():
    """ci=1 exactly where the cost is within CI_CAP, every decode is in, and the prefill left out are the large S."""
    wrong = [s["kernel"] for s in SHAPES if (s["ci"] == "1") != (int(s["cost"]) <= ek.CI_CAP)]
    out = sorted(s["kernel"] for s in SHAPES if s["ci"] == "0")
    ok = not wrong and all(s["ci"] == "1" for s in SHAPES if s["kind"] == "decode") and "prefill_64x8_s8192" in out
    return ok, (wrong, len(out))


def _run(shape, fail=None):
    tmp, stub, machine = _stub()
    if fail:
        os.environ["STUB_FAIL"] = fail
    os.makedirs(os.path.join(tmp, "logs"))
    driver, sim.DRIVER = sim.DRIVER, stub
    try:
        got = sim.simulate_shape(shape, os.path.join(tmp, "work"), os.path.join(tmp, "logs"), 60, machine,
                                 os.path.join(GEN, "kernels"), KEYS)
    finally:
        sim.DRIVER = driver
    return machine, got


def _results(machine, done):
    path = os.path.join(tempfile.mkdtemp(prefix="attnsim-"), "results.json")

    class Args:
        jobs, timeout = 1, 60
    sim.write_results(path, Args, machine, done, len(done), KEYS)
    with open(path) as fh:
        return json.load(fh)


def _a_run_is_named_by_kind_hq_hkv_s():
    machine, a = _run(BY["prefill_32x8_s512"])
    _, b = _run(BY["decode_64x8_s8192"])
    res = _results(machine, [a, b])
    names = [(s["kind"], s["Hq"], s["Hkv"], s["S"], s["kernel"]) for s in res["shapes"]]
    want = [("decode", 64, 8, 8192, "decode_64x8_s8192"), ("prefill", 32, 8, 512, "prefill_32x8_s512")]
    return names == want and a["rc"] == 0 and a["us"] == 3502 / machine["core_freq_mhz"], names


def _ci_flag_times_the_capped_set():
    """simulate_shapes --ci hands the pool the ci=1 shapes only and expects that many."""
    tmp, stub, machine = _stub()
    argv, driver = sys.argv, sim.DRIVER
    sys.argv, sim.DRIVER = ["simulate_shapes.py", "--out", os.path.join(tmp, "out"), "--ci", "--jobs", "8"], stub
    try:
        rc = sim.main(here=HERE, keys=KEYS)
    finally:
        sys.argv, sim.DRIVER = argv, driver
    with open(os.path.join(tmp, "out", "results.json")) as fh:
        res = json.load(fh)
    ran = sorted(s["kernel"] for s in res["shapes"])
    want = sorted(s["kernel"] for s in SHAPES if s["ci"] == "1")
    return rc == 0 and ran == want and res["expected"] == len(want), (rc, len(ran), len(want))


def _report_counts_flops_and_tells_failed_from_capped():
    """FLOP and bytes are the archive's; a failed kernel and a shape over the cap both have no ratio, and only the first
    is a failure."""
    machine, a = _run(BY["prefill_16x8_s1024"])
    _, b = _run(BY["decode_16x8_s128"], fail="decode_16x8_s128")
    res = _results(machine, [a, b])
    real = emit_report.measured(os.path.join(HERE, "tpu_v6e_attention_torchxla.csv"))
    shapes = os.path.join(tempfile.mkdtemp(prefix="attnsim-"), "shapes.csv")
    keep = (",prefill_16x8_s1024,", ",decode_16x8_s128,", ",prefill_64x8_s8192,")
    with open(os.path.join(GEN, "shapes.csv")) as src, open(shapes, "w") as dst:
        lines = src.read().splitlines()
        dst.write("\n".join([lines[0]] + [ln for ln in lines if any(k in ln for k in keep)]) + "\n")
    rows = emit_report.join_rows(shapes, real, res)
    by = {r["kernel"]: r for r in rows}
    pf, dc, cap = by["prefill_16x8_s1024"], by["decode_16x8_s128"], by["prefill_64x8_s8192"]
    chips = emit_report.count_chips(rows)
    page = emit_report.emit_html(rows, chips, {
        "commit": "0" * 40, "archive_commit": "b169399", "date": "now",
        "ran_on": emit_report.gemm_report().ran_on(res["machine"])}, {"peak_bf16_tflops": 918, "hbm_gbps": 1638.4})
    md = emit_report.emit_summary(rows, chips, {"commit": "0" * 40, "ran_on": "x"})
    ok = (pf["flops"] == 4 * 16 * 128 * 1024 * 1025 // 2 and pf["bytes"] == 2 * 128 * 1024 * (2 * 16 + 2 * 8)
          and pf["ratio"] is not None and dc["ratio"] is None and dc["failed_step"] == "binary" and not dc["skipped"]
          and cap["skipped"] and cap["ratio"] is None and "Failed (1)" in md and "prefill_64x8_s8192" in md
          and ("1", "over the CI cap, not run", False) in chips and ("1 / 2", "simulation failed", True) in chips
          and "tpuv6e (vcix_tpuv6e)" in page and "b169399" in page)
    return ok, (pf["flops"], pf["bytes"], dc["failed_step"], cap["rc"], chips)


def _shards_partition_by_cost():
    """The two parts are disjoint, cover the set, and each runs the largest cost first."""
    parts = [[s["kernel"] for s in sim.shard(SHAPES, k, 2, KEYS)] for k in (0, 1)]
    flat = parts[0] + parts[1]
    c = {s["kernel"]: int(s["cost"]) for s in SHAPES}
    ordered = all(c[a] >= c[b] for p in parts for a, b in zip(p, p[1:]))
    ok = sorted(flat) == sorted(c) and len(set(flat)) == len(flat) and ordered and parts[0][0] == "prefill_64x8_s8192"
    return ok, ([len(p) for p in parts], parts[0][:2], parts[1][:2])


CASES = {
    "one generated kernel per measured shape": _one_kernel_per_measured_shape,
    "every block is the policy's": _blocks_are_the_policys,
    "ci marks exactly the shapes under the cap": _ci_is_the_cap,
    "a run is named and sorted by kind, Hq, Hkv, S": _a_run_is_named_by_kind_hq_hkv_s,
    "--ci times the capped set and expects it": _ci_flag_times_the_capped_set,
    "the report counts FLOP, a failure, a cap": _report_counts_flops_and_tells_failed_from_capped,
    "shards partition the set, largest cost first": _shards_partition_by_cost,
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
