#!/usr/bin/env python3
"""Time every shape of shapes.csv on --target (default the v6e MXU) with TOGSim, no Spike: a binary,
gem5 tile samples and a TOGSim run, --jobs kernels at a time. results.json holds the machine and per
shape the exit code, cycles and the target, model and config it ran on; exit 1 if any failed."""
import argparse
import concurrent.futures
import csv
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
from runtime.device import config  # noqa: E402

DRIVER = os.path.join(REPO, "tests", "torchsim-compile")
TARGET = os.path.join(REPO, "tests", "targets", "tpuv6e.json")
RAN_ON = ("target", "matmul_unit", "model", "togsim_config")

STEPS = (("binary", ["--to", "binary"]), ("timing", ["--from", "timing", "--to", "timing"]))


def read_machine(path):
    """What the report states about the TOGSim config a run used, read from the YAML itself."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return {"togsim_config": path, "core_freq_mhz": cfg["core_freq_mhz"],
            "num_systolic_array_per_core": cfg.get("num_systolic_array_per_core"),
            "systolic_array_split": cfg.get("systolic_array_split")}


def plan_machine(path):
    """The machine every shape must run on: the target file, its name and matmul_unit, and the
    model and TOGSim config the device layer picks for it, with what that YAML says."""
    t = _target.load(path)
    machine = {"target_file": os.path.abspath(path), "target": t.name, "matmul_unit": t.matmul_unit,
               "model": config.vcix_model(t)}
    machine.update(read_machine(config.togsim_config(t)))
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
    got.update(target=target.get("name"), matmul_unit=target.get("matmul_unit", "systolic"),
               model=tog.get("model"), togsim_config=tog["config"], total_cycles=tog["total_cycles"])
    off = [f"{k} {got[k]} (planned {machine[k]})" for k in RAN_ON if got[k] != machine[k]]
    if off:
        got.update(rc="ran on another machine: " + "; ".join(off), failed_step="machine")
    else:
        got["us"] = tog["total_cycles"] / machine["core_freq_mhz"]


def simulate_shape(shape, work, logs, timeout, machine):
    """One kernel through both steps on `machine`'s target; the first step to exit non-zero (or
    time out) ends it, and anything raised on the way is that shape's failure, never lost."""
    kernel = shape["kernel"]
    wd = os.path.join(work, kernel)
    log = os.path.join(logs, kernel + ".log")
    env = dict(os.environ, TORCHSIM_COMPILE_TARGET=machine["target_file"])
    base = [sys.executable, DRIVER,
            os.path.join(HERE, "kernels", kernel + ".py"), "--workdir", wd,
            "--tog", "--tog-grouping", "segment", "--timing", "--breakdown"]
    got = {"M": int(shape["M"]), "K": int(shape["K"]), "N": int(shape["N"]), "kernel": kernel}
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


def write_results(path, args, machine, done, total):
    """results.json, rewritten after every shape so a cancelled run still leaves what finished;
    machine is what every shape was planned on, and each shape says what it ran on."""
    shapes = sorted(done, key=lambda r: (r["M"], r["K"], r["N"]))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"machine": machine, "jobs": args.jobs, "timeout_s": args.timeout,
                   "expected": total, "shapes": shapes}, fh, indent=1)
    os.replace(tmp, path)


def main():
    """Command line: the output directory, the pool size, and optionally a subset of kernels."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="results.json and logs/ go here")
    ap.add_argument("--work", help="kernel workdirs (default: OUT/work)")
    ap.add_argument("--jobs", type=int, default=8, help="kernels simulated at once")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per step of one kernel")
    ap.add_argument("--shapes", default=os.path.join(HERE, "shapes.csv"))
    ap.add_argument("--only", nargs="*", help="kernel names to run instead of all")
    ap.add_argument("--target", default=TARGET,
                    help="the target JSON every shape compiles for and is timed on (default: "
                         "tests/targets/tpuv6e.json, the v6e on its TPUV6_MXU)")
    args = ap.parse_args()
    machine = plan_machine(args.target)
    print(f"target {machine['target']} (matmul_unit {machine['matmul_unit']}), model "
          f"{machine['model']}, TOGSim {machine['togsim_config']}", flush=True)

    with open(args.shapes, newline="") as fh:
        shapes = list(csv.DictReader(fh))
    if args.only:
        unknown = set(args.only) - {s["kernel"] for s in shapes}
        if unknown:
            raise SystemExit(f"not in {args.shapes}: {', '.join(sorted(unknown))}")
        shapes = [s for s in shapes if s["kernel"] in args.only]
    out = os.path.abspath(args.out)
    work = os.path.abspath(args.work or os.path.join(out, "work"))
    logs = os.path.join(out, "logs")
    os.makedirs(work, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    results = os.path.join(out, "results.json")

    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(simulate_shape, s, work, logs, args.timeout, machine) for s in shapes]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            done.append(r)
            write_results(results, args, machine, done, len(shapes))
            state = f"{r['us']:.3f} us" if r["rc"] == 0 else f"FAILED at {r['failed_step']} (rc {r['rc']})"
            print(f"[{len(done)}/{len(shapes)}] {r['M']}x{r['K']}x{r['N']} {r['kernel']}: {state} "
                  f"in {r['wall_s']} s", flush=True)
    failed = [r for r in done if r["rc"] != 0]
    print(f"{len(done) - len(failed)} of {len(done)} simulated; results in {results}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
