#!/usr/bin/env python3
"""The validation error loop's objective, per target group: the geomean of TOGSim time / measured time within
TARGET of 1. Reads the results.csv each suite's emit_report wrote and prints a markdown table (and --json the
same as data); the share within +-20% is shown beside the geomean, which a mix of fast and slow rows can hide.
    python3 validation/error_targets.py --gemm DIR --vector DIR --attention DIR [--json PATH]
Stdlib only: it runs on the CI host, outside the compiler image."""
import argparse
import csv
import json
import math
import os

TARGET = 0.15
SPREAD = 0.20


def gemm_group(r):
    return "GEMV (M=1)" if int(r["M"]) == 1 else "GEMM"


#: suite -> how a results.csv row names its target group.
GROUPS = {
    "gemm": gemm_group,
    "attention": lambda r: f"attention {r['kind']}",
    "vector": lambda r: f"vector {r['program']}",
}


def rows_of(directory):
    path = os.path.join(directory, "results.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def groups(dirs):
    """{group: [ratio or None per workload]} over the suites whose results.csv exists."""
    out = {}
    for suite, name_of in GROUPS.items():
        rows = rows_of(dirs[suite]) if dirs.get(suite) else None
        for r in rows or ():
            out.setdefault(name_of(r), []).append(float(r["ratio"]) if r.get("ratio") else None)
    return out


def summarize(ratios):
    done = [q for q in ratios if q is not None]
    gm = math.exp(sum(math.log(q) for q in done) / len(done)) if done else None
    return {"workloads": len(ratios), "timed": len(done), "geomean": gm,
            "within_20": sum(1 for q in done if abs(q - 1) <= SPREAD),
            "on_target": gm is not None and abs(gm - 1) <= TARGET}


def markdown(summary):
    met = sum(1 for s in summary.values() if s["on_target"])
    out = [f"**{met} / {len(summary)} groups on target** (geomean of TOGSim ÷ measured within "
           f"{1 - TARGET:.2f}..{1 + TARGET:.2f})", "",
           "| group | geomean | target | within ±20% | timed / workloads |", "|---|---:|:---:|---:|---:|"]
    for g, s in summary.items():
        gm = "–" if s["geomean"] is None else f"{s['geomean']:.3f}"
        out.append(f"| {g} | {gm} | {'✅' if s['on_target'] else '❌'} | {s['within_20']} / {s['timed']} "
                   f"| {s['timed']} / {s['workloads']} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for suite in GROUPS:
        ap.add_argument(f"--{suite}", help=f"the directory holding {suite}'s results.csv")
    ap.add_argument("--json", help="write the per-group summary here as well")
    args = ap.parse_args()
    summary = {g: summarize(r) for g, r in sorted(groups(vars(args)).items())}
    print(markdown(summary), end="")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"target": TARGET, "groups": summary}, fh, indent=1)


if __name__ == "__main__":
    main()
