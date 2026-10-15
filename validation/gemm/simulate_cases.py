"""simulate_shapes generates the kernels under the tiling file, runs every shape on --target (default the suite's
tpuv6e, whatever the environment names) and records what each ran on; a stub stands in for torchsim-compile.
   python3 validation/gemm/simulate_cases.py        exit 0 iff every case holds"""
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
_HOLD = tempfile.mkdtemp(prefix="simcases-")
# Another machine file: the v6e's, copied under another name.
ELSEWHERE = os.path.join(_HOLD, "elsewhere.yml")
shutil.copyfile(sim._target.resolve(sim.TARGET), ELSEWHERE)
STUB = '''import json, os, sys, yaml
a = sys.argv; wd = a[a.index("--workdir") + 1]; os.makedirs(wd, exist_ok=True)
if a[a.index("--to") + 1] == "binary":
    path = os.environ["TORCHSIM_COMPILE_TARGET"]
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    said = dict(raw, file=path, name=os.path.splitext(os.path.basename(path))[0])
    json.dump({"target": said}, open(os.path.join(wd, "kernel.json"), "w"))
else:
    json.dump({"total_cycles": 3502, "config": os.environ["STUB_CONFIG"],
               "model": os.environ["STUB_MODEL"]}, open(os.path.join(wd, "togsim.json"), "w"))
'''
GEN = os.path.join(_HOLD, "generated")
SHAPES = {s["kernel"]: s for s in sim.generate(HERE, sim.TILING, GEN)}
SHAPE = SHAPES["tk_32x768x768_b_w32c256"]


def _run(machine, model=None):
    """One shape through the stub on `machine`; the stub reports `model` (default the planned one)."""
    tmp = tempfile.mkdtemp(prefix="simcases-")
    stub = os.path.join(tmp, "stub.py")
    with open(stub, "w") as fh:
        fh.write(STUB)
    os.makedirs(os.path.join(tmp, "logs"))
    os.environ.update(STUB_CONFIG=machine["togsim_config"], STUB_MODEL=model or machine["model"])
    driver, sim.DRIVER = sim.DRIVER, stub
    try:
        return sim.simulate_shape(SHAPE, os.path.join(tmp, "work"), os.path.join(tmp, "logs"), 60, machine,
                                  os.path.join(GEN, "kernels"))
    finally:
        sim.DRIVER = driver


def _ran_on(got):
    return (got["rc"], got["target"], got["accel"], os.path.basename(got["model"]),
            os.path.basename(got["togsim_config"]), got.get("us"))


def _default_is_v6e():
    os.environ["TORCHSIM_COMPILE_TARGET"] = ELSEWHERE
    got = _ran_on(_run(sim.plan_machine(sim.TARGET)))
    want = (0, "tpuv6e", "vcix_tpuv6e", "libtpuv6e.so", V6E_YML, 2.0)
    return sim.TARGET == "tpuv6e" and got == want, got


def _another_machine_file_is_its_own_config():
    """--target of another machine file runs on that file: the file is the TOGSim config."""
    got = _ran_on(_run(sim.plan_machine(ELSEWHERE)))
    want = (0, "elsewhere", "vcix_tpuv6e", "libtpuv6e.so", "elsewhere.yml", 2.0)
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


def _shapes():
    return list(SHAPES.values())


def _shards_partition_longest_first():
    """Every part of 2 and of 3 is fixed, disjoint from the others, covers the set, and runs largest first."""
    shapes, bad = _shapes(), []
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
    top = max(int(s["M"]) * int(s["K"]) * int(s["N"]) for s in shapes)
    first = next(s for s in shapes if s["kernel"] == order[0])
    ok = not bad and sim.cost(first, sim.KEYS) == top
    return ok, (bad, order[0], top, [len(sim.shard(shapes, k, 2, sim.KEYS)) for k in (0, 1)])


def _merged_shards_are_one_run():
    """Two shards' results.json and logs/ merge into exactly what one run of both shapes writes."""
    machine = sim.plan_machine(sim.TARGET)
    a, b = _run(machine), dict(_run(machine), M=64, kernel="tk_64x768x768_b_w32c256")

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


def _the_tiling_file_decides_the_kernels():
    """A policy file whose GEMV row for 2048x2048 says XBLOCK 1024 re-blocks that kernel alone; the policy each run
    used is kept beside its kernels."""
    with open(sim.TILING) as fh:
        text = fh.read()
    other = os.path.join(_HOLD, "other.py")
    with open(other, "w") as fh:
        fh.write(text.replace("(2048, 2048): (2048, 1024)", "(2048, 2048): (1024, 1024)"))
    got = {s["kernel"] for s in sim.generate(HERE, other, os.path.join(_HOLD, "other"))}
    moved = sorted(set(SHAPES) - got), sorted(got - set(SHAPES))
    want = (["pts_1x2048x2048_x2048r1024"], ["pts_1x2048x2048_x1024r1024"])
    kept = os.path.exists(os.path.join(_HOLD, "other", "tiling.py")) and os.path.exists(os.path.join(GEN, "tiling.py"))
    return moved == want and kept and len(SHAPES) == 62, moved


CASES = {
    "the tiling file alone decides the kernels": _the_tiling_file_decides_the_kernels,
    "default target is the v6e, not the env's": _default_is_v6e,
    "--target of another machine file times on it": _another_machine_file_is_its_own_config,
    "a shape timed on another model fails": _other_model_fails,
    "results.json and the report name the machine": _results_and_report_name_it,
    "shards partition the set, largest first": _shards_partition_longest_first,
    "merged shards are one run's results.json": _merged_shards_are_one_run,
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
