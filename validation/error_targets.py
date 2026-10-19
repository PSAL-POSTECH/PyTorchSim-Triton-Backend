#!/usr/bin/env python3
"""The validation error loop's objective, per target group: the geomean of TOGSim time / measured time within
`target` of 1. The groups are validation/error_targets.json's: each takes the rows of one suite's results.csv whose
columns hold the listed values (`match`), or the suite's rows no other group took (`rest`); an `exclude`d group is
shown but not judged, and a row no group takes is counted as ungrouped, so a new workload is never dropped
silently. The share within `spread` is shown beside the geomean, which a mix of fast and slow rows can hide.
    python3 validation/error_targets.py --gemm DIR --vector DIR --attention DIR [--config JSON] [--previous JSON]
                                        [--json PATH]
--previous adds the earlier run's geomeans and the workloads that crossed `spread`. A group with no timed workload
is "not measured", never off target. Stdlib only: it runs on the CI host."""
import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ("gemm", "vector", "attention")
LISTED = 10


def rows_of(directory):
    path = os.path.join(directory or "", "results.csv")
    if not directory or not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def assign(config, rows_by_suite):
    """{group name: {workload: ratio or None}} in the config's order, and {suite: rows no group took}."""
    out = {g["name"]: {} for g in config["groups"]}
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
            out[g["name"]][r["kernel"]] = float(r["ratio"]) if r.get("ratio") else None
    return out, ungrouped


def summarize(ratios, target, spread, excluded):
    done = [q for q in ratios.values() if q is not None]
    gm = math.exp(sum(math.log(q) for q in done) / len(done)) if done else None
    within = sum(1 for q in done if abs(q - 1) <= spread)
    return {"workloads": len(ratios), "timed": len(done), "geomean": gm, "within_spread": within,
            "outside_spread": len(done) - within, "excluded": excluded,
            "on_target": None if gm is None else abs(gm - 1) <= target, "ratios": ratios}


def crossings(now, then, spread):
    """(workloads now outside spread that were within, workloads now within that were outside)."""
    def inside(q):
        return q is not None and abs(q - 1) <= spread
    left = sorted(k for k, q in now.items() if q is not None and not inside(q) and inside(then.get(k)))
    came = sorted(k for k, q in now.items() if inside(q) and then.get(k) is not None and not inside(then.get(k)))
    return left, came


def listed(names):
    more = f" and {len(names) - LISTED} more" if len(names) > LISTED else ""
    return ", ".join(f"`{n}`" for n in names[:LISTED]) + more


def markdown(config, summary, ungrouped, previous=None):
    t, sp = config["target"], config["spread"]
    judged = [g for g in config["groups"] if not g.get("exclude")]
    met = sum(1 for g in judged if summary[g["name"]]["on_target"])
    unmeasured = [g["name"] for g in judged if summary[g["name"]]["on_target"] is None]
    head = f"**{met} / {len(judged)} groups on target** (geomean of TOGSim ÷ measured within {1 - t:.2f}..{1 + t:.2f})"
    if unmeasured:
        head += f"; not measured: {', '.join(unmeasured)}"
    out = [head, "", f"| group | geomean | previous | target | within ±{sp:.0%} | timed / workloads |",
           "|---|---:|---:|:---:|---:|---:|"]
    before = (previous or {}).get("groups") or {}
    moves = []
    for g in config["groups"]:
        s = summary[g["name"]]
        gm = "–" if s["geomean"] is None else f"{s['geomean']:.3f}"
        mark = "excluded" if g.get("exclude") else {True: "✅", False: "❌", None: "not measured"}[s["on_target"]]
        p = before.get(g["name"], {})
        pm = "–" if p.get("geomean") is None else f"{p['geomean']:.3f}"
        lost = " ⚠" if s["timed"] < s["workloads"] else ""
        out.append(f"| {g['name']} | {gm} | {pm} | {mark} | {s['within_spread']} / {s['timed']} "
                   f"| {s['timed']} / {s['workloads']}{lost} |")
        if p.get("ratios") and not g.get("exclude"):
            left, came = crossings(s["ratios"], p["ratios"], sp)
            parts = []
            if left:
                parts.append(f"{len(left)} left ±{sp:.0%}: {listed(left)}")
            if came:
                parts.append(f"{len(came)} came within: {listed(came)}")
            if parts:
                moves.append(f"- **{g['name']}**: " + "; ".join(parts))
    if moves:
        out += ["", f"Workloads that crossed ±{sp:.0%} since the previous column:"] + moves
    if any(s["timed"] < s["workloads"] for s in summary.values()):
        out.append("\n⚠ some workloads were not timed (failed, timed out or over a CI cap): a geomean is of the rest.")
    for g in config["groups"]:
        if g.get("exclude"):
            out.append(f"\nExcluded: {g['name']}" + (f" ({g['why']})" if g.get("why") else ""))
    for suite, rows in ungrouped.items():
        out.append(f"\n**Ungrouped ({suite}, {len(rows)}):** no group in validation/error_targets.json takes "
                   + listed([r["kernel"] for r in rows]))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for suite in SUITES:
        ap.add_argument(f"--{suite}", help=f"the directory holding {suite}'s results.csv")
    ap.add_argument("--config", default=os.path.join(HERE, "error_targets.json"))
    ap.add_argument("--previous", help="the JSON an earlier run wrote, to set each group and workload against")
    ap.add_argument("--json", help="write the per-group summary here as well")
    args = ap.parse_args()
    with open(args.config) as fh:
        config = json.load(fh)
    ratios, ungrouped = assign(config, {s: rows_of(getattr(args, s)) for s in SUITES})
    summary = {g["name"]: summarize(ratios[g["name"]], config["target"], config["spread"], bool(g.get("exclude")))
               for g in config["groups"]}
    previous = None
    if args.previous and os.path.isfile(args.previous):
        with open(args.previous) as fh:
            previous = json.load(fh)
    print(markdown(config, summary, ungrouped, previous), end="")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"target": config["target"], "spread": config["spread"], "groups": summary,
                       "ungrouped": {s: len(r) for s, r in ungrouped.items()}}, fh, indent=1)


if __name__ == "__main__":
    main()
