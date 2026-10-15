#!/usr/bin/env python3
"""Generate the suite's kernels under the tiling policy (--tiling, default validation/tiling.py) into OUT/generated,
then time every shape (or --shard K/N's part, largest first) on --target with TOGSim, no Spike; results.json holds
the machine and per shape what it ran on, --merge joins parts into one run's; exit 1 if any failed."""
import argparse
import concurrent.futures
import csv
import filecmp
import importlib.util
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
from runtime.device import config  # noqa: E402
import suite_target  # noqa: E402

DRIVER = os.path.join(REPO, "tests", "torchsim-compile")
TARGET = suite_target.NAME
RAN_ON = ("target", "accel", "model", "togsim_config")

STEPS = (("binary", ["--to", "binary"]), ("timing", ["--from", "timing", "--to", "timing"]))
TILING = os.path.join(os.path.dirname(HERE), "tiling.py")
KEYS = ("M", "K", "N")


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
    return machine


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
    """What a shape is ranked by when the pool and the shards are planned: the product of its numbers
    (M*N*K for a GEMM, R*C for a vector program)."""
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


def simulate_shape(shape, work, logs, timeout, machine, kernels, keys=KEYS):
    """One kernel of `kernels` through both steps on `machine`'s target; the first step to exit
    non-zero (or time out) ends it, and anything raised on the way is that shape's failure, never lost."""
    kernel = shape["kernel"]
    wd = os.path.join(work, kernel)
    log = os.path.join(logs, kernel + ".log")
    env = dict(os.environ, TORCHSIM_COMPILE_TARGET=machine["target_file"])
    base = [sys.executable, DRIVER,
            os.path.join(kernels, kernel + ".py"), "--workdir", wd,
            "--tog", "--timing", "--breakdown"]
    got = dict(key_of(shape, keys), kernel=kernel)
    got.update(rc=0, failed_step=None, workdir=wd, log=os.path.relpath(log, os.path.dirname(logs)))
    t0 = time.monotonic()
    step = "binary"
    try:
        shutil.rmtree(wd, ignore_errors=True)
        with open(log, "w") as fh:
            for step, extra in STEPS:
                fh.write(f"$ {' '.join(base + extra)}\n")
                fh.flush()
                rc = run_step(base + extra, fh, timeout, env)
                if rc == "timeout":
                    fh.write(f"\n[simulate_shapes] {step} timed out after {timeout} s\n")
                if rc != 0:
                    got.update(rc=rc, failed_step=step)
                    break
        if got["rc"] == 0:
            step = "machine"
            read_togsim(got, wd, machine)
    except Exception as e:
        got.update(rc=f"{type(e).__name__}: {e}", failed_step=step)
    got["wall_s"] = round(time.monotonic() - t0, 1)
    return got


def write_results(path, args, machine, done, total, keys=KEYS, part=None):
    """results.json, rewritten after every shape so a cancelled run still leaves what finished;
    machine is what every shape was planned on, and each shape says what it ran on."""
    shapes = sorted(done, key=lambda r: tuple(r[k] for k in keys))
    res = {"machine": machine, "jobs": args.jobs, "timeout_s": args.timeout, "expected": total, "shapes": shapes}
    if part:
        res["shard"] = part
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, path)


def merge(paths, out, keys=KEYS):
    """The results.json of shard runs, the logs/ beside each and their generated/, as one run's in `out`: every
    part must be planned on the same machine under the same tiling, expected is the whole set's; 1 if one is missing."""
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
    if any(not filecmp.cmp(os.path.join(g, "tiling.py"), os.path.join(gens[0], "tiling.py"), shallow=False)
           for g in gens):
        raise SystemExit("shards ran different tiling policies: " + ", ".join(paths))
    shutil.copytree(gens[0], os.path.join(out, "generated"), dirs_exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    for p in paths:
        logs = os.path.join(os.path.dirname(os.path.abspath(p)), "logs")
        if os.path.isdir(logs):
            shutil.copytree(logs, os.path.join(out, "logs"), dirs_exist_ok=True)

    class Args:
        jobs, timeout = first["jobs"], first["timeout_s"]
    write_results(os.path.join(out, "results.json"), Args, first["machine"],
                  [s for r in parts for s in r["shapes"]], first["shard"]["total"], keys)
    if seen != list(range(n)):
        print(f"merged shards {seen} of {n}: the rest left no results", flush=True)
        return 1
    return 0


def generate(here, tiling, out):
    """here/emit_kernels.py under the policy at `tiling`: out/kernels and out/shapes.csv, the policy copied beside."""
    spec = importlib.util.spec_from_file_location(f"emit_kernels_{os.path.basename(here)}",
                                                  os.path.join(here, "emit_kernels.py"))
    ek = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ek)
    ek.use_tiling(tiling)
    ek.generate(out)
    shutil.copyfile(tiling, os.path.join(out, "tiling.py"))
    with open(os.path.join(out, "shapes.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def main(here=HERE, keys=KEYS, doc=__doc__):
    """Command line: the output directory, the pool size, the tiling policy and optionally a subset or a shard;
    `here` holds emit_kernels.py, and `keys` are the columns that name a shape."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--out", required=True, help="results.json and logs/ go here")
    ap.add_argument("--work", help="kernel workdirs (default: OUT/work)")
    ap.add_argument("--jobs", type=int, default=8, help="kernels simulated at once")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per step of one kernel")
    ap.add_argument("--tiling", default=TILING, help="the tiling policy file (default validation/tiling.py)")
    ap.add_argument("--only", nargs="*", help="kernel names to run instead of all")
    ap.add_argument("--shard", help="K/N: time part K (0-based) of N disjoint parts of the set")
    ap.add_argument("--merge", nargs="+", metavar="RESULTS",
                    help="join these shard results.json (and their logs/) into OUT and exit")
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
    shapes = generate(here, os.path.abspath(args.tiling), gen)
    print(f"{len(shapes)} kernels under {args.tiling} -> {gen}", flush=True)
    if args.only:
        unknown = set(args.only) - {s["kernel"] for s in shapes}
        if unknown:
            raise SystemExit(f"not generated: {', '.join(sorted(unknown))}")
        shapes = [s for s in shapes if s["kernel"] in args.only]
    part = None
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= k < n:
            raise SystemExit(f"--shard {args.shard}: K must be in 0..N-1")
        part = {"index": k, "of": n, "total": len(shapes)}
        shapes = shard(shapes, k, n, keys)
        print(f"shard {k}/{n}: {len(shapes)} of {part['total']} shapes", flush=True)
    else:
        shapes = longest_first(shapes, keys)
    work = os.path.abspath(args.work or os.path.join(out, "work"))
    logs = os.path.join(out, "logs")
    os.makedirs(work, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    results = os.path.join(out, "results.json")

    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(simulate_shape, s, work, logs, args.timeout, machine,
                               os.path.join(gen, "kernels"), keys) for s in shapes]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            done.append(r)
            write_results(results, args, machine, done, len(shapes), keys, part)
            state = f"{r['us']:.3f} us" if r["rc"] == 0 else f"FAILED at {r['failed_step']} (rc {r['rc']})"
            print(f"[{len(done)}/{len(shapes)}] {label(r, keys)} {r['kernel']}: {state} "
                  f"in {r['wall_s']} s", flush=True)
    failed = [r for r in done if r["rc"] != 0]
    print(f"{len(done) - len(failed)} of {len(done)} simulated; results in {results}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
