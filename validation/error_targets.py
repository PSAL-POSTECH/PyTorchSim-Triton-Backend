#!/usr/bin/env python3
"""The validation error loop's objective, per target group: the geomean of TOGSim time / measured time within
`target` of 1. The groups are validation/error_targets.json's: each takes the rows of one suite's results.csv whose
columns hold the listed values (`match`), or the suite's rows no other group took (`rest`); an `exclude`d group is
shown but not judged, and a row no group takes is counted as ungrouped, so a new workload is never dropped
silently. The share within `spread` is shown beside the geomean, which a mix of fast and slow rows can hide.
    python3 validation/error_targets.py --gemm DIR --vector DIR --attention DIR [--config JSON] [--json PATH]
Stdlib only: it runs on the CI host, outside the compiler image."""
import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ("gemm", "vector", "attention")


def rows_of(directory):
    path = os.path.join(directory or "", "results.csv")
    if not directory or not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def assign(config, rows_by_suite):
    """{group name: [ratio or None]} in the config's order, and {suite: rows no group took}."""
    out = {g["name"]: [] for g in config["groups"]}
    ungrouped = {}
    for suite, rows in rows_by_suite.items():
        mine = [g for g in config["groups"] if g["suite"] == suite]
        for r in rows:
            g = next((g for g in mine if g.get("match") and
                      all(r.get(col) in vals for col, vals in g["match"].items())), None)
            g = g or next((g for g in mine if g.get("rest")), None)
            if g is None:
                ungrouped.setdefault(suite, []).append(r)
                continue
            out[g["name"]].append(float(r["ratio"]) if r.get("ratio") else None)
    return out, ungrouped


def summarize(ratios, target, spread):
    done = [q for q in ratios if q is not None]
    gm = math.exp(sum(math.log(q) for q in done) / len(done)) if done else None
    return {"workloads": len(ratios), "timed": len(done), "geomean": gm,
            "within_spread": sum(1 for q in done if abs(q - 1) <= spread),
            "on_target": gm is not None and abs(gm - 1) <= target}


def markdown(config, summary, ungrouped):
    t, sp = config["target"], config["spread"]
    judged = [g for g in config["groups"] if not g.get("exclude")]
    met = sum(1 for g in judged if summary[g["name"]]["on_target"])
    out = [f"**{met} / {len(judged)} groups on target** (geomean of TOGSim ÷ measured within {1 - t:.2f}..{1 + t:.2f})",
           "", f"| group | geomean | target | within ±{sp:.0%} | timed / workloads |", "|---|---:|:---:|---:|---:|"]
    for g in config["groups"]:
        s = summary[g["name"]]
        gm = "–" if s["geomean"] is None else f"{s['geomean']:.3f}"
        mark = "excluded" if g.get("exclude") else ("✅" if s["on_target"] else "❌")
        out.append(f"| {g['name']} | {gm} | {mark} | {s['within_spread']} / {s['timed']} | {s['timed']} / {s['workloads']} |")
    for g in config["groups"]:
        if g.get("exclude"):
            out.append(f"\nExcluded: {g['name']}" + (f" ({g['why']})" if g.get("why") else ""))
    for suite, rows in ungrouped.items():
        out.append(f"\n**Ungrouped ({suite}, {len(rows)}):** no group in validation/error_targets.json takes "
                   + ", ".join(f"`{r['kernel']}`" for r in rows[:10]) + (" ..." if len(rows) > 10 else ""))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for suite in SUITES:
        ap.add_argument(f"--{suite}", help=f"the directory holding {suite}'s results.csv")
    ap.add_argument("--config", default=os.path.join(HERE, "error_targets.json"))
    ap.add_argument("--json", help="write the per-group summary here as well")
    args = ap.parse_args()
    with open(args.config) as fh:
        config = json.load(fh)
    ratios, ungrouped = assign(config, {s: rows_of(getattr(args, s)) for s in SUITES})
    summary = {name: summarize(r, config["target"], config["spread"]) for name, r in ratios.items()}
    for g in config["groups"]:
        summary[g["name"]]["excluded"] = bool(g.get("exclude"))
    print(markdown(config, summary, ungrouped), end="")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"target": config["target"], "groups": summary,
                       "ungrouped": {s: len(r) for s, r in ungrouped.items()}}, fh, indent=1)


if __name__ == "__main__":
    main()
