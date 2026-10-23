"""simulate_shapes times the suite's workloads of a kernel bundle on --target (default the suite's tpuv6e, whatever
the environment names), sums each workload's launches and records what it ran on; a synthetic bundle stands in for
PyTorchSim's and a stub for torchsim-compile.
   python3 validation/gemm/simulate_cases.py        exit 0 iff every case holds"""
import csv
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_report  # noqa: E402
import simulate_shapes as sim  # noqa: E402

V6E_YML = "tpuv6e.yml"
CYCLES = 3502
_HOLD = tempfile.mkdtemp(prefix="simcases-")
# Another machine file: the v6e's, copied under another name.
ELSEWHERE = os.path.join(_HOLD, "elsewhere.yml")
shutil.copyfile(sim._target.resolve(sim.TARGET), ELSEWHERE)
STUB = '''import json, os, sys, yaml
a = sys.argv; wd = a[a.index("--workdir") + 1]; os.makedirs(wd, exist_ok=True)
with open(os.path.join(wd, "steps.jsonl"), "a") as fh:
    fh.write(json.dumps(a[a.index("--from") + 1] if "--from" in a else None) + "\\n")
if a[a.index("--to") + 1] == "binary":
    path = os.environ["TORCHSIM_COMPILE_TARGET"]
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    said = dict(raw, file=path, name=os.path.splitext(os.path.basename(path))[0])
    open(os.path.join(wd, "cycle.elf"), "w").write(os.environ.get("STUB_ELF", "elf"))
    json.dump({"target": said, "artifacts": {"cycle_elf": "cycle.elf"}}, open(os.path.join(wd, "kernel.json"), "w"))
else:
    json.dump({"total_cycles": int(os.environ.get("STUB_CYCLES", %d)), "config": os.environ["STUB_CONFIG"],
               "model": os.environ["STUB_MODEL"]}, open(os.path.join(wd, "togsim.json"), "w"))
''' % CYCLES


def _workload(bundle, name, suite, dims, launches):
    """A workload of `bundle`: `launches` names its kernels in launch order."""
    out = []
    for kernel in launches:
        d = f"{sorted(set(launches)).index(kernel):02d}_{kernel}"
        os.makedirs(os.path.join(bundle, name, d), exist_ok=True)
        open(os.path.join(bundle, name, d, kernel + "_spec.py"), "w").close()
        out.append({"kernel": kernel, "dir": d, "tile_source": None})
    with open(os.path.join(bundle, name, "manifest.json"), "w") as fh:
        json.dump(dict(dims, workload=name, suite=suite, dtype="bf16", launches=out), fh)


def _bundle(torch=None):
    """Three GEMM workloads (one launches a kernel twice and another once), an attention and a vector one, and
    layer workloads on either side of the layer suite's token cap."""
    bundle = tempfile.mkdtemp(prefix="simcases-bundle-")
    with open(os.path.join(bundle, "bundle.json"), "w") as fh:
        json.dump({"pytorchsim": "0" * 40, "torch": torch or importlib.metadata.version("torch")}, fh)
    _workload(bundle, "gemm_32x768x768_bf16", "gemm", {"M": 32, "K": 768, "N": 768}, ["mm_a"])
    _workload(bundle, "gemm_64x768x768_bf16", "gemm", {"M": 64, "K": 768, "N": 768}, ["mm_b"])
    _workload(bundle, "gemm_1x2048x2048_bf16", "gemm", {"M": 1, "K": 2048, "N": 2048}, ["red", "red", "add"])
    _workload(bundle, "attention_prefill_16x8x8192_bf16", "attention",
              {"kind": "prefill", "Hq": 16, "Hkv": 8, "S": 8192}, ["_prefill_kernel"])
    _workload(bundle, "vector_add1_8x128_bf16", "vector", {"program": "add1", "R": 8, "C": 128}, ["add"])
    for section, name, T, S in (("layer", "prefill", 4096, 4096), ("layer", "prefill", 8192, 8192),
                                ("layer", "decode", 1, 4096), ("layer", "decode", 1, 8192),
                                ("op", "rmsnorm", 4096, 0), ("op", "rmsnorm", 8192, 0)):
        _workload(bundle, f"layer_{section}_{name}_{T}x{S}_bf16", "layer",
                  {"section": section, "name": name, "T": T, "S": S}, ["k"])
    return bundle


BUNDLE = _bundle()
GEN = os.path.join(_HOLD, "generated")
SHAPES = {s["kernel"]: s for s in sim.load_bundle(BUNDLE, "gemm", sim.KEYS, GEN)}


def _run(machine, model=None, shape="gemm_32x768x768_bf16", reuse=None):
    """One workload through the stub on `machine`; the stub reports `model` (default the planned one); `reuse` is
    simulate_shape's."""
    tmp = tempfile.mkdtemp(prefix="simcases-")
    stub = os.path.join(tmp, "stub.py")
    with open(stub, "w") as fh:
        fh.write(STUB)
    os.makedirs(os.path.join(tmp, "logs"))
    os.environ.update(STUB_CONFIG=machine["togsim_config"], STUB_MODEL=model or machine["model"])
    driver, sim.DRIVER = sim.DRIVER, stub
    try:
        kw = {"reuse": reuse} if reuse is not None else {}
        return sim.simulate_shape(SHAPES[shape], os.path.join(tmp, "work"), os.path.join(tmp, "logs"), 60, machine, **kw)
    finally:
        sim.DRIVER = driver


def _ran_on(got):
    return (got["rc"], got["target"], got["accel"], os.path.basename(got["model"]),
            os.path.basename(got["togsim_config"]), got.get("us"))


def _the_suites_workloads_and_their_launches():
    """The suite's workloads alone, each kernel once with how often it launches; shapes.csv lists them and the
    bundle.json the run used sits beside it."""
    got = {k: [(d["dir"], d["launches"]) for d in s["dirs"]] for k, s in SHAPES.items()}
    want = {"gemm_1x2048x2048_bf16": [("01_red", 2), ("00_add", 1)],
            "gemm_32x768x768_bf16": [("00_mm_a", 1)], "gemm_64x768x768_bf16": [("00_mm_b", 1)]}
    with open(os.path.join(GEN, "shapes.csv"), newline="") as fh:
        listed = {r["kernel"]: (r["kernels"], r["launches"]) for r in csv.DictReader(fh)}
    ok = (got == want and listed["gemm_1x2048x2048_bf16"] == ("2", "3")
          and os.path.exists(os.path.join(GEN, "bundle.json")))
    return ok, (got, listed)


def _another_torch_is_refused():
    """A bundle built under another torch than the one that would load its kernels is not timed."""
    try:
        sim.load_bundle(_bundle(torch="0.0.0"), "gemm", sim.KEYS, os.path.join(_HOLD, "refused"))
    except sim.BundleError as e:
        return "torch 0.0.0" in str(e), str(e)
    return False, "timed"


def _launches_are_summed():
    """A workload's cycles are its kernels' cycles once per launch."""
    machine = sim.plan_machine(sim.TARGET)
    got = _run(machine, shape="gemm_1x2048x2048_bf16")
    ok = (got["rc"] == 0 and got["total_cycles"] == 3 * CYCLES
          and got["us"] == 3 * CYCLES / machine["core_freq_mhz"] and len(got["kernels"]) == 2)
    return ok, (got["rc"], got.get("total_cycles"), got["kernels"])


def _spike_only_before_an_indirect_dma():
    """A kernel with an indirect DMA runs Spike and then timing; any other goes from its binary to timing alone."""
    machine = sim.plan_machine(sim.TARGET)
    said, sim.has_indirect_dma = sim.has_indirect_dma, lambda wd: os.path.basename(wd) == "01_red"
    try:
        got = _run(machine, shape="gemm_1x2048x2048_bf16")
    finally:
        sim.has_indirect_dma = said
    steps = {}
    for k in got["kernels"]:
        with open(os.path.join(got["workdir"], k["dir"], "steps.jsonl")) as fh:
            steps[k["dir"]] = ([json.loads(line) for line in fh], k["spike"])
    want = {"01_red": ([None, "spike"], True), "00_add": ([None, "timing"], False)}
    return got["rc"] == 0 and steps == want, steps


def _default_is_v6e():
    os.environ["TORCHSIM_COMPILE_TARGET"] = ELSEWHERE
    machine = sim.plan_machine(sim.TARGET)
    got = _ran_on(_run(machine))
    want = (0, "tpuv6e", "vcix_tpuv6e", "libtpuv6e.so", V6E_YML, CYCLES / machine["core_freq_mhz"])
    return sim.TARGET == "tpuv6e" and got == want, got


def _another_machine_file_is_its_own_config():
    """--target of another machine file runs on that file: the file is the TOGSim config."""
    machine = sim.plan_machine(ELSEWHERE)
    got = _ran_on(_run(machine))
    want = (0, "elsewhere", "vcix_tpuv6e", "libtpuv6e.so", "elsewhere.yml", CYCLES / machine["core_freq_mhz"])
    return got == want, got


def _other_model_fails():
    got = _run(sim.plan_machine(sim.TARGET), model="/elsewhere/libtpu.so")
    ok = got["failed_step"] == "machine" and "model /elsewhere/libtpu.so" in str(got["rc"]) and "us" not in got
    return ok, (got["failed_step"], got["rc"])


def _results_and_report_name_it():
    machine = sim.plan_machine(sim.TARGET)
    got = _run(machine)
    path = os.path.join(tempfile.mkdtemp(prefix="simcases-"), "results.json")

    class Args:
        jobs, timeout = 1, 60
    sim.write_results(path, Args, machine, [got], 1)
    with open(path) as fh:
        res = json.load(fh)
    head = emit_report.ran_on(res["machine"])
    ok = (res["machine"]["accel"] == "vcix_tpuv6e" and res["shapes"][0]["model"] == machine["model"]
          and "tpuv6e (vcix_tpuv6e)" in head and "libtpuv6e.so" in head and V6E_YML in head)
    return ok, head


def _report_sets_it_against_main():
    """With main's results.json, a workload whose cycles moved is counted and listed, one that did not is the same,
    and one main never ran has no ratio."""
    rows = [{"M": 32, "K": 768, "N": 768, "sim_us": 2.0, "total_cycles": 2000},
            {"M": 64, "K": 768, "N": 768, "sim_us": 1.0, "total_cycles": 1000},
            {"M": 1, "K": 2048, "N": 2048, "sim_us": 1.0, "total_cycles": 1000}]
    base = os.path.join(tempfile.mkdtemp(prefix="simcases-"), "results.json")
    with open(base, "w") as fh:
        json.dump({"shapes": [{"M": 32, "K": 768, "N": 768, "rc": 0, "us": 1.0, "total_cycles": 1000},
                              {"M": 64, "K": 768, "N": 768, "rc": 0, "us": 1.0, "total_cycles": 1000}]}, fh)
    emit_report.attach_baseline(rows, base, ("M", "K", "N"))
    text = "\n".join(emit_report.baseline_summary(rows, lambda r: f"{r['M']}", "f" * 40))
    ok = ([r["vs_main"] for r in rows] == [2.0, 1.0, None]
          and "1 same cycles · 0 faster · 1 slower · 0 newly fail · 1 now run" in text and "| 32 |" in text)
    return ok, text


def _attention_ci_times_every_shape():
    """The attention suite ranks a workload by its score elements and CI times every one, a prefill of S 8192 too."""
    path = os.path.join(os.path.dirname(HERE), "attention", "simulate_shapes.py")
    spec = importlib.util.spec_from_file_location("attention_simulate_shapes", path)
    att = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(att)
    got = sim.load_bundle(BUNDLE, "attention", att.KEYS, os.path.join(_HOLD, "attention"),
                          att.score_elements)
    want = 16 * 8192 * 8193 // 2
    return [(s["cost"], s["ci"]) for s in got] == [(want, 1)], [(s["kernel"], s["cost"], s["ci"]) for s in got]


def _layer_ci_is_the_token_cap():
    """The layer suite's --ci times a layer of S <= 4096 and an op of T <= 4096, whatever the rank orders by."""
    path = os.path.join(os.path.dirname(HERE), "layer", "simulate_shapes.py")
    spec = importlib.util.spec_from_file_location("layer_simulate_shapes", path)
    lay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lay)
    got = {s["kernel"]: s["ci"] for s in sim.load_bundle(BUNDLE, "layer", lay.KEYS, os.path.join(_HOLD, "layer"),
                                                         lay.token_pairs, lay.under_cap)}
    want = {"layer_layer_prefill_4096x4096_bf16": 1, "layer_layer_prefill_8192x8192_bf16": 0,
            "layer_layer_decode_1x4096_bf16": 1, "layer_layer_decode_1x8192_bf16": 0,
            "layer_op_rmsnorm_4096x0_bf16": 1, "layer_op_rmsnorm_8192x0_bf16": 0}
    return got == want, got


def _table(rows):
    """A measured gemm table of `rows`, each (M, K, N, source) with the columns join_rows reads."""
    path = tempfile.mktemp(prefix="simcases-real-", suffix=".csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["M", "K", "N", "source", "run_id", "status", "latency_ns", "flops", "bytes"])
        w.writerows([M, K, N, src, "run/" + src, "ok", 1000, 2 * M * K * N, 2 * (M * K + K * N + M * N)]
                    for M, K, N, src in rows)
    return path


def _gemm_ci_caps_the_capped_table_only():
    """gemm's --ci leaves out a workload of the capped table over the cap and times every other one; the report
    names it over the cap, not failed, with the source its measured row gives."""
    table = _table([(64, 768, 768, "q")])
    got = {s["kernel"]: s["ci"] for s in sim.load_bundle(BUNDLE, "gemm", sim.KEYS, os.path.join(_HOLD, "gemmci"),
                                                         in_ci=sim.under_cap(table, 32))}
    want = {"gemm_32x768x768_bf16": 1, "gemm_64x768x768_bf16": 0, "gemm_1x2048x2048_bf16": 1}
    other = _table([(32, 768, 768, "a"), (1, 2048, 2048, "a")])
    rows = emit_report.join_rows(os.path.join(_HOLD, "gemmci", "shapes.csv"), [other, table], {"shapes": []})
    said = {r["kernel"]: (r["source"], r["skipped"], r["rc"]) for r in rows}
    ok = (got == want and said["gemm_64x768x768_bf16"] == ("q", True, "over the CI cap")
          and said["gemm_32x768x768_bf16"] == ("a", False, "not run"))
    return ok, (got, said)


def _a_shape_measured_twice_is_refused():
    """Two measured tables that both hold one shape are refused, not joined to whichever came last."""
    try:
        emit_report.measured([_table([(32, 768, 768, "a")]), _table([(32, 768, 768, "q")])])
    except SystemExit as e:
        return "measured twice" in str(e), str(e)
    return False, "joined"


def _shards_partition_longest_first():
    """Every part of 2 and of 3 is fixed, disjoint from the others, covers the set, and runs largest first."""
    shapes, bad = list(SHAPES.values()), []
    for n in (2, 3):
        parts = [[s["kernel"] for s in sim.shard(shapes, k, n, sim.KEYS)] for k in range(n)]
        again = [[s["kernel"] for s in sim.shard(shapes, k, n, sim.KEYS)] for k in range(n)]
        flat = [k for p in parts for k in p]
        if parts != again or sorted(flat) != sorted(s["kernel"] for s in shapes) or len(set(flat)) != len(flat):
            bad.append(f"{n} parts")
        by = {s["kernel"]: sim.cost(s, sim.KEYS) for s in shapes}
        if any(by[a] < by[b] for p in parts for a, b in zip(p, p[1:])):
            bad.append(f"{n} parts not largest first")
    order = [s["kernel"] for s in sim.longest_first(shapes, sim.KEYS)]
    return not bad and order[0] == "gemm_64x768x768_bf16", (bad, order)


def _merged_shards_are_one_run():
    """Two shards' results.json and logs/ merge into exactly what one run of both workloads writes."""
    machine = sim.plan_machine(sim.TARGET)
    a, b = _run(machine), _run(machine, shape="gemm_64x768x768_bf16")

    class Args:
        jobs, timeout = 1, 60
    root = tempfile.mkdtemp(prefix="simcases-")
    one = os.path.join(root, "one.json")
    sim.write_results(one, Args, machine, [b, a], 2)
    paths = []
    for k, got in enumerate((a, b)):
        d = os.path.join(root, f"shard{k}")
        os.makedirs(os.path.join(d, "logs"))
        shutil.copytree(GEN, os.path.join(d, "generated"))
        open(os.path.join(d, "logs", got["kernel"] + ".log"), "w").close()
        paths.append(os.path.join(d, "results.json"))
        sim.write_results(paths[-1], Args, machine, [got], 1, sim.KEYS, {"index": k, "of": 2, "total": 2})
    rc = sim.merge(paths[::-1], os.path.join(root, "out"))
    with open(one) as fh, open(os.path.join(root, "out", "results.json")) as gh:
        want, got = json.load(fh), json.load(gh)
    logs = sorted(os.listdir(os.path.join(root, "out", "logs")))
    ok = (rc == 0 and got == want and logs == sorted(g["kernel"] + ".log" for g in (a, b))
          and os.path.exists(os.path.join(root, "out", "generated", "shapes.csv")))
    return ok, (rc, got.get("expected"), [s["kernel"] for s in got["shapes"]], logs)


def _a_timeout_is_reported_not_failed():
    """A workload past --timeout leaves the run's exit code 0; one that failed to compile makes it 1."""
    slow = [{"rc": 0}, {"rc": "timeout"}]
    broken = slow + [{"rc": 1}]
    got = (sim.outcome(slow, "results.json"), sim.outcome(broken, "results.json"))
    return got == (0, 1), got


def _vector_measured_time_leaves_out_the_lead_in():
    """vector: a row of a table that measures the lead-in is compared with its latency less it, a row of a
    table that does not is compared whole, and a row of the former without one is refused."""
    path = os.path.join(os.path.dirname(HERE), "vector", "emit_report.py")
    spec = importlib.util.spec_from_file_location("vector_emit_report", path)
    vec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vec)
    shapes = os.path.join(tempfile.mkdtemp(prefix="simcases-"), "shapes.csv")
    with open(shapes, "w") as fh:
        fh.write("program,R,C,kernel,kernels,launches\nmax,8,128,m,1,1\nsoftmax,8,128,s,1,1\n")
    row = {"bytes": "2048", "status": "ok", "run_id": "r"}
    real = {("max", 8, 128): dict(row, latency_ns="1365", lead_in_ns="300"),
            ("softmax", 8, 128): dict(row, latency_ns="3831")}
    results = {"shapes": [{"kernel": "m", "rc": 0, "us": 1.0}, {"kernel": "s", "rc": 0, "us": 1.0}]}
    got = {r["program"]: (round(r["real_us"], 3), r["lead_in_us"]) for r in vec.join_rows(shapes, real, results)}
    real[("max", 8, 128)]["lead_in_ns"] = ""
    try:
        vec.join_rows(shapes, real, results)
        refused = False
    except SystemExit:
        refused = True
    return got == {"max": (1.065, 0.3), "softmax": (3.831, 0.0)} and refused, (got, refused)


def _steps(got):
    out = {}
    for k in got["kernels"]:
        with open(os.path.join(got["workdir"], k["dir"], "steps.jsonl")) as fh:
            out[k["dir"]] = [json.loads(line) for line in fh]
    return out


def _earlier(machine, shape="gemm_1x2048x2048_bf16"):
    """An earlier run of `shape` as --reuse reads it: its results.json, written and read back."""
    path = os.path.join(tempfile.mkdtemp(prefix="simcases-"), "results.json")

    class Args:
        jobs, timeout = 1, 60
    sim.write_results(path, Args, machine, [_run(machine, shape=shape)], 1)
    return sim.reusable(path)


def _reuse_takes_an_unchanged_kernels_cycles():
    """A kernel whose timing key an earlier run holds takes that run's cycles and is not timed again; the key is
    recorded either way."""
    machine = sim.plan_machine(sim.TARGET)
    reuse = _earlier(machine)
    os.environ["STUB_CYCLES"] = "7"
    try:
        got = _run(machine, shape="gemm_1x2048x2048_bf16", reuse=reuse)
    finally:
        os.environ.pop("STUB_CYCLES")
    ok = (got["rc"] == 0 and got["total_cycles"] == 3 * CYCLES and all(k["reused"] for k in got["kernels"])
          and all(k["timing_key"] in reuse for k in got["kernels"])
          and all(s == [None] for s in _steps(got).values()))
    return ok, (got["rc"], got.get("total_cycles"), _steps(got))


def _a_changed_timing_input_is_timed_again():
    """A kernel whose cycle ELF differs from the earlier run's has another key: it is timed, not reused."""
    machine = sim.plan_machine(sim.TARGET)
    reuse = _earlier(machine)
    os.environ.update(STUB_CYCLES="7", STUB_ELF="another")
    try:
        got = _run(machine, shape="gemm_1x2048x2048_bf16", reuse=reuse)
    finally:
        os.environ.pop("STUB_CYCLES")
        os.environ.pop("STUB_ELF")
    ok = (got["rc"] == 0 and got["total_cycles"] == 3 * 7 and not any(k["reused"] for k in got["kernels"])
          and not any(k["timing_key"] in reuse for k in got["kernels"]))
    return ok, (got.get("total_cycles"), [k["reused"] for k in got["kernels"]])


def _a_kernel_timed_after_spike_is_not_reused():
    """A kernel with an indirect DMA is timed on the index snapshots Spike's run records, which its key does not hold: never reused."""
    machine = sim.plan_machine(sim.TARGET)
    said, sim.has_indirect_dma = sim.has_indirect_dma, lambda wd: os.path.basename(wd) == "01_red"
    try:
        reuse = _earlier(machine)
        got = _run(machine, shape="gemm_1x2048x2048_bf16", reuse=reuse)
    finally:
        sim.has_indirect_dma = said
    reused = {k["dir"]: k["reused"] for k in got["kernels"]}
    return got["rc"] == 0 and reused == {"01_red": False, "00_add": True}, reused


CASES = {
    "the suite's workloads and their launches": _the_suites_workloads_and_their_launches,
    "a bundle of another torch is refused": _another_torch_is_refused,
    "a workload's launches are summed": _launches_are_summed,
    "Spike runs only before an indirect DMA's timing": _spike_only_before_an_indirect_dma,
    "default target is the v6e, not the env's": _default_is_v6e,
    "--target of another machine file times on it": _another_machine_file_is_its_own_config,
    "a workload timed on another model fails": _other_model_fails,
    "results.json and the report name the machine": _results_and_report_name_it,
    "the report sets each workload against main": _report_sets_it_against_main,
    "attention CI times every shape": _attention_ci_times_every_shape,
    "layer --ci is the token cap": _layer_ci_is_the_token_cap,
    "gemm --ci caps the capped table only": _gemm_ci_caps_the_capped_table_only,
    "a shape measured in two tables is refused": _a_shape_measured_twice_is_refused,
    "shards partition the set, largest first": _shards_partition_longest_first,
    "merged shards are one run's results.json": _merged_shards_are_one_run,
    "a timeout is reported, not a failure": _a_timeout_is_reported_not_failed,
    "vector: the measured time leaves out the lead-in": _vector_measured_time_leaves_out_the_lead_in,
    "--reuse takes an unchanged kernel's cycles": _reuse_takes_an_unchanged_kernels_cycles,
    "a changed timing input is timed again": _a_changed_timing_input_is_timed_again,
    "a kernel timed after Spike is not reused": _a_kernel_timed_after_spike_is_not_reused,
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
