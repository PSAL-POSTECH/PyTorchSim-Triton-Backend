#!/usr/bin/env python3
"""Time the suite's workloads in PyTorchSim's kernel bundle (--bundle, default validation/pytorchsim) on --target
with TOGSim; Spike runs only before timing a kernel with an indirect DMA, whose gathers TOGSim times by Spike's index
dump. Each kernel a workload launches, largest workload first (or --shard K/N's part), its cycles summed over the
launches. results.json holds the machine and per workload what it ran on, --merge joins parts into one run's; exit 1
if any failed, but not for a workload that only ran past --timeout (the report shows it). OUT/generated/shapes.csv
lists the workloads and the bundle.json they came from (`picked` 0 for one --where leaves out). --ci leaves out the rows of CAPPED, the Qwen3-30B-A3B table, of
M over CI_M (the layer suite's token cap); every other workload of this suite is timed. --reuse takes a kernel's cycles
from an earlier run's results.json when its timing key (timing_key) is that run's, instead of timing it again."""
import argparse
import concurrent.futures
import csv
import filecmp
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402
from pytorchsim_triton_compiler.contract import vcix_model  # noqa: E402
from runtime import tog_replay  # noqa: E402
from runtime.device import config  # noqa: E402
import suite_target  # noqa: E402

DRIVER = os.path.join(REPO, "tests", "torchsim-compile")
TARGET = suite_target.NAME
RAN_ON = ("target", "accel", "model", "togsim_config")

STEPS = (("binary", ["--to", "binary"]), ("timing", ["--from", "timing", "--to", "timing"]))
#: The timing step of a kernel with an indirect DMA: TOGSim times a gather by the index dump Spike leaves.
SPIKE_THEN_TIMING = ("timing", ["--from", "spike", "--to", "timing"])
BUNDLE = os.path.join(os.path.dirname(HERE), "pytorchsim")
#: Kept per kernel in OUT/ir/<workload>/<dir>/ for the error loop's diagnosis.
IR_KEPT = ("08-transformed.mlir", "togsim.json", "triton-compile.stderr")
KEYS = ("M", "K", "N")
SHAPE_COLS = ("kernel", "kernels", "launches", "cost", "ci", "picked")
#: What times a kernel besides its own object: the harness that runs the timing step (tests/runtime, the driver,
#: this file); plan_machine adds the simulators, the model and the machine files it plans on.
HARNESS = (os.path.join(REPO, "tests", "runtime"), DRIVER, os.path.abspath(__file__))
#: The artifacts the timing step reads (gem5 the cycle ELF, TOGSim the TOG and its tile types, both the machine
#: config); the others (the functional ELF, the .ll with its workdir path) do not decide the cycles.
TIMING_READS = ("cycle_elf", "machine_config", "tile_types", "tog_so")
#: The measured table --ci caps, and its cap: the maintainer's, the tokens M a workload holds.
CAPPED = os.path.join(HERE, "tpu_v6e_qwen3moe_torchxla.csv")
CI_M = 4096


def read_machine(path):
    """What the report states about the TOGSim config a run used, read from the YAML itself."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return {"togsim_config": path, "core_freq_mhz": cfg["core_freq_mhz"],
            "num_systolic_array_per_core": cfg.get("num_systolic_array_per_core"),
            "systolic_array_split": cfg.get("systolic_array_split")}


def plan_machine(said):
    """The machine every shape must run on: the machine file `said` names, its accelerator, the
    model library that accelerator is found as, and what the file says TOGSim times on."""
    t = _target.load(said)
    model = config.VCIX_MODEL or vcix_model.find(config.VCIX_BUILD, t.accel)
    machine = {"target_file": t.file, "target": t.name, "accel": t.accel, "model": model}
    machine.update(read_machine(t.file))
    machine["timing_tools"] = digest(HARNESS + (config.GEM5, config.TOGSIM, model, t.file,
                                                vcix_model.gem5_script(t.accel)))
    return machine


def digest(paths):
    """One hash of the files at `paths`, a directory standing for every file under it (bytecode aside) by its
    path relative to it; a path that is not there counts as its name."""
    h = hashlib.sha256()
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            files = sorted(os.path.join(d, f) for d, ds, fs in os.walk(p) if "__pycache__" not in d
                           for f in fs if not f.endswith(".pyc"))
        else:
            files = [p]
        for f in files:
            h.update(os.path.relpath(f, p if os.path.isdir(p) else os.path.dirname(p)).encode() + b"\0")
            if os.path.isfile(f):
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            else:
                h.update(b"missing:" + f.encode())
            h.update(b"\0")
    return h.hexdigest()


def timing_key(wd, machine):
    """What the timing step of the kernel built in `wd` reads: its manifest, the artifacts of TIMING_READS it names,
    and the machine's timing tools. Two kernels with one key time to the same cycles."""
    path = os.path.join(wd, "kernel.json")
    with open(path) as fh:
        arts = json.load(fh).get("artifacts", {})
    files = digest([path] + [os.path.join(wd, arts[k]) for k in TIMING_READS if k in arts])
    return hashlib.sha256(f"{machine['timing_tools']} {files}".encode()).hexdigest()


def reusable(results):
    """timing_key -> (kernel, what its workload ran on) for every kernel of an earlier run's results.json that was
    timed without Spike in a workload that finished: what --reuse takes cycles from."""
    with open(results) as fh:
        shapes = json.load(fh)["shapes"]
    return {k["timing_key"]: (k, {f: s[f] for f in RAN_ON}) for s in shapes if s["rc"] == 0
            for k in s.get("kernels", ()) if k.get("timing_key") and not k.get("spike")}


def run_step(argv, fh, timeout, env):
    """One step in its own process group, so a timeout kills gem5 and TOGSim with it and the pool
    never holds more live simulators than --jobs; the exit code, or "timeout"."""
    p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        return "timeout"


def read_togsim(got, wd, machine):
    """The cycles of one finished kernel and what it ran on, from the kernel.json and togsim.json
    it left; a shape that ran on anything but `machine` fails at step "machine"."""
    with open(os.path.join(wd, "kernel.json")) as fh:
        target = json.load(fh)["target"]
    with open(os.path.join(wd, "togsim.json")) as fh:
        tog = json.load(fh)
    accel = next((e for e in target.get("isa", ()) if _target.is_accel(e)), None)
    got.update(target=target.get("name"), accel=accel,
               model=tog.get("model"), togsim_config=tog["config"], total_cycles=tog["total_cycles"])
    off = [f"{k} {got[k]} (planned {machine[k]})" for k in RAN_ON if got[k] != machine[k]]
    if off:
        got.update(rc="ran on another machine: " + "; ".join(off), failed_step="machine")
    else:
        got["us"] = tog["total_cycles"] / machine["core_freq_mhz"]


def key_of(shape, keys):
    """The columns that name a shape, a number where the column holds one."""
    return {k: int(shape[k]) if str(shape[k]).isdigit() else shape[k] for k in keys}


def label(r, keys):
    """A shape as printed: its words, then its numbers joined by x (128x768x768, softmax 8x128)."""
    words = [r[k] for k in keys if isinstance(r[k], str)]
    return " ".join(words + ["x".join(str(r[k]) for k in keys if not isinstance(r[k], str))])


def cost(shape, keys):
    """What a shape is ranked by when the pool and the shards are planned: its `cost` if it has one (attention),
    else the product of its numbers (M*N*K for a GEMM, R*C for a vector program)."""
    if shape.get("cost"):
        return int(shape["cost"])
    n = 1
    for v in key_of(shape, keys).values():
        n *= 1 if isinstance(v, str) else v
    return n


def longest_first(shapes, keys):
    """The shapes by cost, largest first, ties in key order: the order the pool is handed them."""
    return sorted(shapes, key=lambda s: (-cost(s, keys), tuple(key_of(s, keys).values())))


def shard(shapes, k, n, keys):
    """Part k of n, by longest processing time: each shape, largest first, goes to the part with the
    least cost so far (ties to the lower part), so the parts are disjoint, cover the set and are fixed."""
    load, mine = [0] * n, []
    for s in longest_first(shapes, keys):
        i = min(range(n), key=lambda j: (load[j], j))
        load[i] += cost(s, keys)
        if i == k:
            mine.append(s)
    return mine


def has_indirect_dma(wd):
    """Whether the kernel built in `wd` moves anything by an index: a transfer of its post-vcix IR with an index
    operand, read through the bindings (tog_replay.ir_transfers), which is what becomes an indirect TOGSim dma."""
    return any(t["indirect"] for t in tog_replay.ir_transfers(wd) or ())


def simulate_kernel(spec, wd, fh, timeout, env, machine=None, reuse=None):
    """One kernel through both steps, Spike before timing only for a kernel with an indirect DMA; (the step that
    failed or None, its exit code, whether Spike ran, its timing key, the `reuse` entry its timing was taken from)."""
    base = [sys.executable, DRIVER, spec, "--workdir", wd, "--tog", "--timing", "--breakdown"]
    shutil.rmtree(wd, ignore_errors=True)
    spike, key = False, None
    for step, extra in STEPS:
        if step == "timing" and has_indirect_dma(wd):
            step, extra = SPIKE_THEN_TIMING
            spike = True
        if step == "timing" and machine is not None and not spike:
            key = timing_key(wd, machine)
            if key in (reuse or {}):
                fh.write(f"[simulate_shapes] timing key {key[:16]} is an earlier run's: its cycles, not timed again\n")
                return None, 0, spike, key, reuse[key]
        fh.write(f"$ {' '.join(base + extra)}\n")
        fh.flush()
        rc = run_step(base + extra, fh, timeout, env)
        if rc == "timeout":
            fh.write(f"\n[simulate_shapes] {step} timed out after {timeout} s\n")
        if rc != 0:
            return step, rc, spike, key, None
    return None, 0, spike, key, None


def read_reused(got, wd, earlier, machine):
    """read_togsim for a kernel whose cycles come from `earlier` (kernel, ran-on): the target from its own manifest,
    the model, TOGSim config and cycles from the run it was timed in."""
    kernel, ran = earlier
    with open(os.path.join(wd, "kernel.json")) as fh:
        target = json.load(fh)["target"]
    got.update(target=target.get("name"), accel=next((e for e in target.get("isa", ()) if _target.is_accel(e)), None),
               model=ran["model"], togsim_config=ran["togsim_config"], total_cycles=kernel["total_cycles"])
    off = [f"{k} {got[k]} (planned {machine[k]})" for k in RAN_ON if got[k] != machine[k]]
    if off:
        got.update(rc="ran on another machine: " + "; ".join(off), failed_step="machine")


def keep_ir(wd, dest):
    """Copy the IR_KEPT files a kernel's workdir holds to `dest`."""
    for name in IR_KEPT:
        src = os.path.join(wd, name)
        if os.path.isfile(src):
            os.makedirs(dest, exist_ok=True)
            shutil.copyfile(src, os.path.join(dest, name))


def simulate_shape(shape, work, logs, timeout, machine, keys=KEYS, reuse=None):
    """Every kernel a workload launches through both steps on `machine`'s target, its cycles counted once per
    launch (or taken from `reuse`, see reusable); the first kernel to fail ends it, and anything raised on the way is
    that workload's failure, never lost."""
    kernel = shape["kernel"]
    log = os.path.join(logs, kernel + ".log")
    env = dict(os.environ, TORCHSIM_COMPILE_TARGET=machine["target_file"], PYTHONDONTWRITEBYTECODE="1")
    got = dict(key_of(shape, keys), kernel=kernel)
    got.update(rc=0, failed_step=None, workdir=os.path.join(work, kernel),
               log=os.path.relpath(log, os.path.dirname(logs)), kernels=[])
    t0 = time.monotonic()
    step = "binary"
    try:
        with open(log, "w") as fh:
            for launch in shape["dirs"]:
                wd = os.path.join(work, kernel, launch["dir"])
                step, rc, spike, key, earlier = simulate_kernel(launch["spec"], wd, fh, timeout, env, machine, reuse)
                keep_ir(wd, os.path.join(os.path.dirname(logs), "ir", kernel, launch["dir"]))
                if rc != 0:
                    got.update(rc=rc, failed_step=step)
                    break
                step = "machine"
                one = {}
                if earlier:
                    read_reused(one, wd, earlier, machine)
                else:
                    read_togsim(one, wd, machine)
                if one.get("failed_step"):
                    got.update(rc=one["rc"], failed_step=one["failed_step"])
                    break
                got["kernels"].append({"dir": launch["dir"], "launches": launch["launches"],
                                       "total_cycles": one["total_cycles"], "spike": spike,
                                       "timing_key": key, "reused": bool(earlier)})
                got.update({k: one[k] for k in RAN_ON})
        if got["rc"] == 0:
            got["total_cycles"] = sum(k["total_cycles"] * k["launches"] for k in got["kernels"])
            got["us"] = got["total_cycles"] / machine["core_freq_mhz"]
    except Exception as e:
        got.update(rc=f"{type(e).__name__}: {e}", failed_step=step)
    got["wall_s"] = round(time.monotonic() - t0, 1)
    return got


def write_results(path, args, machine, done, total, keys=KEYS, part=None):
    """results.json, written before the first shape and after every one, so a run of no shapes or a cancelled one
    still leaves what finished; machine is what every shape was planned on, and each shape says what it ran on."""
    shapes = sorted(done, key=lambda r: tuple(r[k] for k in keys))
    res = {"machine": machine, "jobs": args.jobs, "timeout_s": args.timeout, "expected": total, "shapes": shapes}
    if part:
        res["shard"] = part
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, path)


def merge(paths, out, keys=KEYS):
    """The results.json of shard runs, the logs/ and ir/ beside each and their generated/, as one run's in `out`: every
    part must be planned on the same machine from the same bundle, expected is the whole set's; 1 if one is missing."""
    parts = []
    for p in paths:
        with open(p) as fh:
            parts.append(json.load(fh))
    first = parts[0]
    if any(r["machine"] != first["machine"] for r in parts):
        raise SystemExit("shards were planned on different machines: " + ", ".join(paths))
    kernels = [s["kernel"] for r in parts for s in r["shapes"]]
    if len(kernels) != len(set(kernels)):
        raise SystemExit("shards overlap: " + ", ".join(sorted({k for k in kernels if kernels.count(k) > 1})))
    n = first["shard"]["of"]
    seen = sorted(r["shard"]["index"] for r in parts)
    gens = [os.path.join(os.path.dirname(os.path.abspath(p)), "generated") for p in paths]
    if any(not filecmp.cmp(os.path.join(g, "bundle.json"), os.path.join(gens[0], "bundle.json"), shallow=False)
           for g in gens):
        raise SystemExit("shards ran different bundles: " + ", ".join(paths))
    shutil.copytree(gens[0], os.path.join(out, "generated"), dirs_exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    for p in paths:
        for kept in ("logs", "ir"):
            src = os.path.join(os.path.dirname(os.path.abspath(p)), kept)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(out, kept), dirs_exist_ok=True)

    class Args:
        jobs, timeout = first["jobs"], first["timeout_s"]
    write_results(os.path.join(out, "results.json"), Args, first["machine"],
                  [s for r in parts for s in r["shapes"]], first["shard"]["total"], keys)
    if seen != list(range(n)):
        print(f"merged shards {seen} of {n}: the rest left no results", flush=True)
        return 1
    return 0


class BundleError(SystemExit):
    """The bundle cannot be timed here."""


def load_bundle(bundle, suite, keys, out, rank=None, in_ci=None, pick=None):
    """The workloads of `suite` in the bundle at `bundle`, each with the spec of every kernel it launches and how
    often; OUT/shapes.csv lists them, OUT/externs.json the calls each one's manifest says Inductor left to aten (not
    simulated) and OUT/bundle.json is the bundle's own. Refused when the bundle's kernels were built under another
    torch than this one, which loads them. `rank` costs a workload (default: its sizes' product), which orders the
    pool and the shards; `in_ci(workload)` says whether --ci times it (default: all), `pick(workload)` whether
    --where keeps it (default: all)."""
    with open(os.path.join(bundle, "bundle.json")) as fh:
        meta = json.load(fh)
    here = importlib.metadata.version("torch")
    if meta.get("torch") != here:
        raise BundleError(f"{bundle} was built under torch {meta.get('torch')}; this is torch {here}")
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(os.path.join(bundle, "bundle.json"), os.path.join(out, "bundle.json"))
    shapes, externs = [], {}
    for name in sorted(os.listdir(bundle)):
        path = os.path.join(bundle, name, "manifest.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            m = json.load(fh)
        if m["suite"] != suite:
            continue
        dirs = {}
        for la in m["launches"]:
            d = dirs.setdefault(la["dir"], {"dir": la["dir"], "launches": 0,
                                            "spec": os.path.join(bundle, name, la["dir"], la["kernel"] + "_spec.py")})
            d["launches"] += 1
        s = dict({k: m[k] for k in keys}, kernel=name, dirs=list(dirs.values()),
                 kernels=len(dirs), launches=len(m["launches"]))
        s["cost"] = rank(s) if rank else cost(s, keys)
        s["ci"] = int(in_ci is None or in_ci(s))
        s["picked"] = int(pick is None or pick(s))
        shapes.append(s)
        if m.get("externs"):
            externs[name] = [e["call"] for e in m["externs"]]
    with open(os.path.join(out, "externs.json"), "w") as fh:
        json.dump(externs, fh, indent=1, sort_keys=True)
    with open(os.path.join(out, "shapes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keys) + list(SHAPE_COLS), extrasaction="ignore")
        w.writeheader()
        w.writerows(shapes)
    return shapes


def under_cap(table=CAPPED, cap=CI_M):
    """This suite's in_ci: a workload `table` measures is timed by --ci at M <= `cap`, any other one always."""
    with open(table, newline="") as fh:
        capped = {tuple(int(r[k]) for k in KEYS) for r in csv.DictReader(fh)}
    return lambda w: tuple(w[k] for k in KEYS) not in capped or w["M"] <= cap


def where(conditions, keys):
    """--where's predicate: a workload whose column COL holds VALUE for every COL=VALUE; COL is one of `keys`."""
    want = {}
    for c in conditions or ():
        col, eq, value = c.partition("=")
        if not eq or col not in keys:
            raise SystemExit(f"--where {c}: COL=VALUE with COL one of {', '.join(keys)}")
        want.setdefault(col, set()).add(value)
    return (lambda w: all(str(w[col]) in vals for col, vals in want.items())) if want else None


def main(here=HERE, keys=KEYS, doc=__doc__, rank=None, in_ci=None):
    """Command line: the output directory, the pool size, the bundle and optionally a subset or a shard; `here`
    is the suite's directory, whose name the bundle's workloads carry, `keys` the columns that name a workload,
    and `rank` / `in_ci` what load_bundle costs and caps them by."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--out", required=True, help="results.json and logs/ go here")
    ap.add_argument("--work", help="kernel workdirs (default: OUT/work)")
    ap.add_argument("--jobs", type=int, default=8, help="kernels simulated at once")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per step of one kernel")
    ap.add_argument("--bundle", default=BUNDLE, help="PyTorchSim's kernel bundle (default validation/pytorchsim)")
    ap.add_argument("--only", nargs="*", help="workload names to run instead of all")
    ap.add_argument("--ci", action="store_true", help="only the workloads under the suite's CI cap, where it has one")
    ap.add_argument("--where", nargs="+", metavar="COL=VALUE",
                    help="only the workloads whose columns hold these values (a column given twice takes either)")
    ap.add_argument("--shard", help="K/N: time part K (0-based) of N disjoint parts of the set")
    ap.add_argument("--merge", nargs="+", metavar="RESULTS",
                    help="join these shard results.json (and their logs/) into OUT and exit")
    ap.add_argument("--reuse", metavar="RESULTS",
                    help="an earlier run's results.json: a kernel whose timing key it holds takes its cycles")
    ap.add_argument("--target", default=TARGET,
                    help=f"the machine (a name under $TOGSIM_DIR/configs or a file) every shape "
                         f"compiles for and is timed on (default: {TARGET})")
    args = ap.parse_args()
    if args.merge:
        return merge(args.merge, os.path.abspath(args.out), keys)
    machine = plan_machine(args.target)
    print(f"target {machine['target']} ({machine['accel']}), model {machine['model']}, "
          f"TOGSim {machine['togsim_config']}", flush=True)

    out = os.path.abspath(args.out)
    gen = os.path.join(out, "generated")
    shapes = load_bundle(os.path.abspath(args.bundle), os.path.basename(here), keys, gen, rank, in_ci,
                         where(args.where, keys))
    print(f"{len(shapes)} workloads in {args.bundle} -> {gen}", flush=True)
    if args.only:
        unknown = set(args.only) - {s["kernel"] for s in shapes}
        if unknown:
            raise SystemExit(f"not in the bundle: {', '.join(sorted(unknown))}")
        shapes = [s for s in shapes if s["kernel"] in args.only]
    if args.where:
        shapes = [s for s in shapes if s["picked"]]
        print(f"--where {' '.join(args.where)}: {len(shapes)} shapes", flush=True)
    if args.ci:
        skipped = [s["kernel"] for s in shapes if not s["ci"]]
        shapes = [s for s in shapes if s["ci"]]
        print(f"--ci: {len(shapes)} shapes, {len(skipped)} over the CI cap left out: {' '.join(skipped)}", flush=True)
    part = None
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= k < n:
            raise SystemExit(f"--shard {args.shard}: K must be in 0..N-1")
        part = {"index": k, "of": n, "total": len(shapes)}
        shapes = shard(shapes, k, n, keys)
        print(f"shard {k}/{n}: {len(shapes)} of {part['total']} shapes", flush=True)
    work = os.path.abspath(args.work or os.path.join(out, "work"))
    logs = os.path.join(out, "logs")
    os.makedirs(work, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    results = os.path.join(out, "results.json")

    reuse = reusable(args.reuse) if args.reuse and os.path.isfile(args.reuse) else None
    if args.reuse:
        print(f"--reuse {args.reuse}: {len(reuse or {})} kernels' timing to take where the key matches", flush=True)
    done = []
    write_results(results, args, machine, done, len(shapes), keys, part)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(simulate_shape, s, work, logs, args.timeout, machine, keys, reuse) for s in shapes]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            done.append(r)
            write_results(results, args, machine, done, len(shapes), keys, part)
            state = f"{r['us']:.3f} us" if r["rc"] == 0 else f"FAILED at {r['failed_step']} (rc {r['rc']})"
            if any(k.get("reused") for k in r["kernels"]):
                state += f" ({sum(bool(k.get('reused')) for k in r['kernels'])} of {len(r['kernels'])} kernels reused)"
            print(f"[{len(done)}/{len(shapes)}] {label(r, keys)} {r['kernel']}: {state} "
                  f"in {r['wall_s']} s", flush=True)
    return outcome(done, results)


def outcome(done, results):
    """The run's exit code: 1 if a workload failed, but not for one that only ran past --timeout. A kernel that
    slow is a finding about the kernel, which the report shows as a timeout; one that does not compile or
    run is a broken measurement."""
    slow = [r for r in done if r["rc"] == "timeout"]
    failed = [r for r in done if r["rc"] not in (0, "timeout")]
    print(f"{len(done) - len(failed) - len(slow)} of {len(done)} simulated, {len(slow)} past the timeout; "
          f"results in {results}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(in_ci=under_cap()))
