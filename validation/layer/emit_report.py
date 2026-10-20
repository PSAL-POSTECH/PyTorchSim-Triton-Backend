#!/usr/bin/env python3
"""results.json + shapes.csv + the measured v6e Qwen3-30B-A3B layer rows -> report.html (self-contained: vanilla JS),
summary.md and results.csv. Raw simulated time against raw measured time, nothing fitted; a failed kernel is shown,
not dropped, a workload left out of the run (over the CI cap) is shown as such, and the measured row's status (ok or
warn, with its reason) is shown beside it. A workload whose manifest lists calls Inductor left to aten (externs.json
beside shapes.csv) is incomplete: its kernels are timed and shown, the calls named, and it is in no verdict. Each layer is also set against the sum of its ops at its T on both sides;
with --baseline (main's results.json) each workload is set against main's simulated time."""
import argparse
import csv
import datetime
import html
import importlib.util
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORST = 10
KEYS = ("section", "name", "T", "S")
NAMES = ("prefill", "decode")
#: How often the layer runs each measured op: PyTorchSim's layer program (scripts/ci/bundle.py `_layer` there)
#: normalizes before attention and before the MoE (rmsnorm) and adds both residuals (residual).
LAYER_OPS = {"rmsnorm": 2, "q_norm": 1, "k_norm": 1, "rope_q": 1, "rope_k": 1, "residual": 2, "router_post": 1,
             "dispatch": 1, "gmm_gate_up": 1, "silu_mul": 1, "gmm_down": 1, "combine": 1}
#: In the layer, not in the measured op set (PSAL-POSTECH/tpu-perf-archive#5 for the projections and the gate).
UNCOVERED = "q/k/v/o_proj, the router gate and attention"
#: Where PyTorchSim tracks a kernel route for each call it leaves to aten.
EXTERN_ISSUE = "PSAL-POSTECH/PyTorchSim#397"


def gemm_report():
    """validation/gemm/emit_report.py under its own module name: the machine line and the baseline are said one way."""
    path = os.path.join(os.path.dirname(HERE), "gemm", "emit_report.py")
    spec = importlib.util.spec_from_file_location("gemm_emit_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def suite_cap():
    """validation/layer/simulate_shapes.py's under_cap, the one definition of this suite's CI cap."""
    spec = importlib.util.spec_from_file_location("layer_simulate_shapes", os.path.join(HERE, "simulate_shapes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.under_cap


def incomplete(calls):
    """Why a workload with these extern calls is in no verdict, or "" when it has none."""
    if not calls:
        return ""
    n = {c: calls.count(c) for c in calls}
    return "left to aten, not simulated: " + ", ".join(f"{c} ×{k}" if k > 1 else c for c, k in n.items())


def key(r):
    return (r["section"], r["name"], int(r["T"]), int(r["S"]))


def measured(layer_csv):
    """(section, name, T, S) -> the archive row measured for it."""
    with open(layer_csv, newline="") as fh:
        return {key(r): r for r in csv.DictReader(fh)}


def join_rows(shapes_csv, real, results, externs=None):
    """One row per workload: its kernels, its measured time (with the spread and status) and its simulated time;
    `skipped` when the run left it out because shapes.csv marks it over the CI cap, `incomplete` (why) when
    `externs` lists calls Inductor left to aten for it."""
    externs = externs or {}
    sim = {r["kernel"]: r for r in results["shapes"]}
    rows = []
    with open(shapes_csv, newline="") as fh:
        for s in csv.DictReader(fh):
            k = key(s)
            if k not in real:
                raise SystemExit(f"no measurement for {k}")
            t, got = real[k], sim.get(s["kernel"], {})
            real_us = int(t["latency_ns"]) / 1000
            sim_us = got.get("us") if got.get("rc") == 0 else None
            skipped = not got and s.get("ci") == "0"
            rows.append({
                "section": k[0], "name": k[1], "T": k[2], "S": k[3], "kernel": s["kernel"],
                "kernels": int(s["kernels"]), "launches": int(s["launches"]), "ci": s.get("ci") != "0",
                "skipped": skipped, "real_us": real_us, "p10_us": int(t["p10_ns"]) / 1000,
                "p90_us": int(t["p90_ns"]) / 1000, "sim_us": sim_us,
                "ratio": None if sim_us is None else sim_us / real_us,
                "status": t["status"], "status_reason": t["status_reason"], "run_id": t["run_id"],
                "total_cycles": got.get("total_cycles"), "wall_s": got.get("wall_s"),
                "rc": got.get("rc", "over the CI cap" if skipped else "not run"),
                "externs": externs.get(s["kernel"], []), "incomplete": incomplete(externs.get(s["kernel"])),
                "failed_step": got.get("failed_step"), "log": got.get("log"), "target": got.get("target"),
                "accel": got.get("accel"), "model": got.get("model"), "togsim_config": got.get("togsim_config")})
    return rows


def judged(r):
    """Whether a row's ratio counts in a verdict: it ran and the bundle holds every call of it."""
    return not r["skipped"] and not r["incomplete"]


def geomean(qs):
    qs = [q for q in qs if q is not None]
    return math.exp(sum(math.log(q) for q in qs) / len(qs)) if qs else None


def decompose(rows, real):
    """Per measured layer: its time against LAYER_OPS' ops at its T, measured (from the measured table, whether or
    not the bundle holds the workload) and simulated, and what is left on each side (layer - sum of ops). A side's
    sum is None unless every op it counts has a time; an incomplete layer or op has no simulated time here."""
    sim = {(r["section"], r["name"], r["T"], r["S"]): r for r in rows}
    out = []
    for k in sorted((k for k in real if k[0] == "layer"), key=lambda k: (NAMES.index(k[1]), k[3])):
        _, name, T, S = k
        layer, real_us = sim.get(k, {}), int(real[k]["latency_ns"]) / 1000
        took = [(real.get(("op", op, T, 0)), n) for op, n in LAYER_OPS.items()]
        real_ops = (sum(int(t["latency_ns"]) / 1000 * n for t, n in took)
                    if all(t is not None for t, _ in took) else None)
        timed = [(sim[("op", op, T, 0)], n) for op, n in LAYER_OPS.items()
                 if sim.get(("op", op, T, 0), {}).get("sim_us") is not None and judged(sim[("op", op, T, 0)])]
        sim_ops = sum(o["sim_us"] * n for o, n in timed) if len(timed) == len(LAYER_OPS) else None
        whole = layer and judged(layer)
        d = {"name": name, "T": T, "S": S, "layer_real_us": real_us,
             "layer_sim_us": layer.get("sim_us") if whole else None, "layer_incomplete": layer.get("incomplete", ""),
             "ops_real_us": real_ops, "ops_sim_us": sim_ops, "ops_timed": len(timed), "ops": len(LAYER_OPS),
             "layer_ratio": layer.get("ratio") if whole else None, "status": real[k]["status"]}
        d["rest_real_us"] = None if real_ops is None else real_us - real_ops
        d["rest_sim_us"] = None if sim_ops is None or d["layer_sim_us"] is None else d["layer_sim_us"] - sim_ops
        d["ops_ratio"] = None if real_ops is None or sim_ops is None else sim_ops / real_ops
        d["rest_ratio"] = (None if d["rest_real_us"] is None or d["rest_sim_us"] is None or d["rest_real_us"] <= 0
                           else d["rest_sim_us"] / d["rest_real_us"])
        out.append(d)
    return out


def unbundled(rows, real, under_cap=None):
    """The measured rows the bundle holds no workload for, as printed: (under the CI cap, over it). PyTorchSim
    bundles this suite's rows under the cap only, so a missing row over it is expected."""
    have = {(r["section"], r["name"], r["T"], r["S"]) for r in rows}
    gone = [k for k in real if k not in have]
    said = {k: f"{k[0]} {k[1]} T{k[2]}" + (f" S{k[3]}" if k[0] == "layer" else "") for k in gone}
    over = [k for k in gone if under_cap and not under_cap(dict(zip(KEYS, k)))]
    return [said[k] for k in gone if k not in over], [said[k] for k in over]


def count_chips(rows, missing, over_cap=()):
    """The headline counts: workloads run, within ±10% / ±20% and the ratio geomean of each layer kind and of the ops,
    over the judged ones (complete), the incomplete, the failures and the workloads left out by the CI cap; each a
    (value, label, alert) triple."""
    run = [r for r in rows if not r["skipped"]]
    whole = [r for r in run if judged(r)]

    def within(tol):
        return sum(1 for r in whole if r["ratio"] is not None and abs(r["ratio"] - 1) <= tol)
    chips = [(f"{len(run)} / {len(rows)}", "workloads run", False),
             (f"{within(.10)} / {len(whole)}", "complete, within ±10%", False),
             (f"{within(.20)} / {len(whole)}", "complete, within ±20%", False)]
    for label, rs in [(f"layer {n}", [r for r in whole if r["section"] == "layer" and r["name"] == n]) for n in NAMES] \
            + [("ops", [r for r in whole if r["section"] == "op"])]:
        gm = geomean(r["ratio"] for r in rs)
        chips.append(("–" if gm is None else f"{gm:.3f}", f"{label} ratio geomean", False))
    part = sum(1 for r in run if r["incomplete"])
    if part:
        chips.append((f"{part} / {len(run)}", "incomplete (calls left to aten), not judged", True))
    failed = sum(1 for r in run if r["ratio"] is None)
    if failed:
        chips.append((f"{failed} / {len(run)}", "simulation failed", True))
    if len(rows) > len(run):
        chips.append((f"{len(rows) - len(run)}", "over the CI cap, not run", False))
    if over_cap:
        chips.append((f"{len(over_cap)}", "measured, over the CI cap, not bundled", False))
    if missing:
        chips.append((f"{len(missing)}", "measured, not in the bundle", True))
    return chips


def cell(r):
    """A ratio as the summary prints it: 'cap' when not run, 'failed', 'extern' when incomplete, else the ratio; ⚠
    for a measured warn."""
    v = ("–" if r is None else "cap" if r["skipped"] else "failed" if r["ratio"] is None
         else "extern" if r["incomplete"] else f"{r['ratio']:.2f}")
    return v + (" ⚠" if r is not None and r["status"] != "ok" else "")


def us(v):
    return "–" if v is None else f"{v:,.1f}"


def emit_summary(rows, real, chips, meta, under_cap=None):
    """Markdown for a job summary: the chips, the layers, the ops' ratios per T, the decomposition, failures,
    workloads not run, the worst outliers and the baseline."""
    out = [f"### Layer validation: TOGSim vs TPU v6e, Qwen3-30B-A3B ({len(rows)} workloads) @ `{meta['commit'][:12]}`",
           "", f"Ran on {meta['ran_on']}. Kernels are PyTorchSim's for the decoder layer as one program and for each "
           f"op alone; CI times the workloads of at most 4096 tokens (a layer's context S, an op's T; "
           f"validation/layer/simulate_shapes.py). ⚠ marks a measured row whose status is warn; `extern` a workload with "
           f"calls Inductor left to aten, which no ratio, geomean or verdict counts.", "",
           "| " + " | ".join(label for _, label, _ in chips) + " |", "|" + "---|" * len(chips),
           "| " + " | ".join(f"**{value}**" for value, _, _ in chips) + " |", "",
           "| layer | T | S | measured µs | TOGSim µs | ratio | measured status |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for r in sorted((r for r in rows if r["section"] == "layer"), key=lambda r: (NAMES.index(r["name"]), r["S"])):
        out.append(f"| {r['name']} | {r['T']} | {r['S']} | {us(r['real_us'])} | {us(r['sim_us'])} | {cell(r)} "
                   f"| {r['status']}{': ' + r['status_reason'] if r['status_reason'] else ''} |")
    ops = {(r["name"], r["T"]): r for r in rows if r["section"] == "op"}
    Ts = sorted({t for _, t in ops})
    if ops:
        out += ["", "TOGSim ÷ measured per op and T:", "", "| op | " + " | ".join(str(t) for t in Ts) + " |",
                "|---|" + "---:|" * len(Ts)]
    for op in [o for o in LAYER_OPS if any(n == o for n, _ in ops)] + sorted({n for n, _ in ops} - set(LAYER_OPS)):
        out.append(f"| {op} | " + " | ".join(cell(ops.get((op, t))) for t in Ts) + " |")
    warn = [r for r in rows if r["section"] == "op" and r["status"] != "ok"]
    if warn:
        out += ["", f"<details><summary>{len(warn)} op rows measured with status warn</summary>", ""]
        out += [f"- {r['name']} T{r['T']}: {r['status_reason']}" for r in warn]
        out += ["", "</details>"]
    counted = ", ".join(f"{n}×{o}" if n > 1 else o for o, n in LAYER_OPS.items())
    out += ["", f"Layer against the sum of its ops at its T ({counted}). "
            f"**The measured op set lacks {UNCOVERED}** (PSAL-POSTECH/tpu-perf-archive#5 for the projections and the "
            f"gate), so Σ ops does not cover the layer: the rest (layer − Σ ops) is that uncovered part plus whatever "
            f"fusion or overlap saves, on each side.", "",
            "| layer | T | S | measured layer | Σ ops | rest | TOGSim layer | Σ ops | rest | Σ ops ratio | rest ratio "
            "| layer ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in decompose(rows, real):
        sim_ops = us(d["ops_sim_us"]) if d["ops_sim_us"] is not None else f"{d['ops_timed']}/{d['ops']} timed"
        out.append(f"| {d['name']} | {d['T']} | {d['S']} | {us(d['layer_real_us'])} | {us(d['ops_real_us'])} "
                   f"| {us(d['rest_real_us'])} | {us(d['layer_sim_us'])} | {sim_ops} | {us(d['rest_sim_us'])} "
                   f"| {fmt(d['ops_ratio'])} | {fmt(d['rest_ratio'])} | {fmt(d['layer_ratio'])} |")
    out.append("")
    part = [r for r in rows if r["incomplete"] and not r["skipped"]]
    if part:
        out += [f"**Incomplete ({len(part)})**: the bundle holds no kernel for these calls, which run as aten ops and "
                f"are not simulated (routes tracked in {EXTERN_ISSUE}); the kernels are timed and shown in the report, "
                f"left out of every ratio, geomean and verdict:", ""]
        out += [f"- `{r['kernel']}`: {r['incomplete']}" for r in part]
        out.append("")
    failed = [r for r in rows if r["ratio"] is None and not r["skipped"]]
    if failed:
        out += [f"**Failed ({len(failed)})**, shown in the report, logs in the artifact:", ""]
        out += [f"- `{r['kernel']}`: " + (f"{r['failed_step']} step, rc {r['rc']}, `{r['log']}`"
                                         if r["failed_step"] else "not run") for r in failed]
        out.append("")
    skipped = [r["kernel"] for r in rows if r["skipped"]]
    if skipped:
        out += [f"Not run, over the CI cap ({len(skipped)}): " + ", ".join(f"`{k}`" for k in skipped), ""]
    missing, over = unbundled(rows, real, under_cap)
    if over:
        out += [f"Measured, over the CI cap, not bundled ({len(over)}): " + ", ".join(f"`{k}`" for k in over), ""]
    if missing:
        out += [f"**Measured, not in the bundle ({len(missing)})**: " + ", ".join(f"`{k}`" for k in missing), ""]
    worst = sorted((r for r in rows if r["ratio"] is not None and judged(r)),
                   key=lambda r: -abs(math.log(r["ratio"])))[:WORST]
    if worst:
        out += [f"Largest deviations (TOGSim ÷ measured, {len(worst)} of {len(rows)}):", "",
                "| workload | kernels | measured µs | TOGSim µs | ratio | measured status |",
                "|---|---:|---:|---:|---:|---|"]
        out += [f"| {label(r)} | {r['kernels']} | {r['real_us']:.3f} | {r['sim_us']:.3f} | {r['ratio']:.3f} "
                f"| {r['status']} |" for r in worst]
        out.append("")
    out += gemm_report().baseline_summary(rows, label, meta.get("baseline_commit"))
    if meta.get("artifact"):
        out.append(f"Full report: `report.html` in artifact `{meta['artifact']}`.")
    return "\n".join(out) + "\n"


def fmt(q):
    return "–" if q is None else f"{q:.3f}"


def label(r):
    return f"{r['name']} T{r['T']}" + (f" S{r['S']}" if r["section"] == "layer" else "")


def emit_csv(path, rows):
    """The joined table, one line per workload."""
    cols = ["section", "name", "T", "S", "kernel", "kernels", "launches", "base_us", "vs_main", "ci", "real_us",
            "p10_us", "p90_us", "sim_us", "ratio", "incomplete", "status", "status_reason", "run_id", "total_cycles",
            "wall_s", "rc",
            "failed_step", "log", "target", "accel", "model", "togsim_config"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def emit_html(rows, real, chips, meta, under_cap=None):
    """The page: header, chips, failures, the layers, the op table, the decomposition, raw table, notes."""
    sha = html.escape(meta["commit"][:7])
    commit = f'<a href="{html.escape(meta["commit_url"])}">{sha}</a>' if meta.get("commit_url") else sha
    run = f' · <a href="{html.escape(meta["run_url"])}">CI run</a>' if meta.get("run_url") else ""
    ref = f' ({html.escape(meta["ref"])})' if meta.get("ref") else ""
    runs = ", ".join(sorted({r["run_id"] for r in rows}))
    head = (f'{meta["date"]} · PTB {commit}{ref}{run} · {html.escape(meta["ran_on"])} · measured torchxla v6e '
            f'({html.escape(runs)}, tpu-perf-archive {html.escape(meta["archive_commit"])})')
    missing, over = unbundled(rows, real, under_cap)
    data = {"rows": rows, "chips": chips, "dec": decompose(rows, real), "missing": missing, "over": over,
            "ops": list(LAYER_OPS), "mult": LAYER_OPS}
    return (PAGE.replace("@@N@@", str(len(rows))).replace("@@HEAD@@", head)
            .replace("@@UNCOVERED@@", html.escape(UNCOVERED)).replace("@@EXTERN_ISSUE@@", html.escape(EXTERN_ISSUE))
            .replace("@@DATA@@", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))


def main():
    """Command line: where results.json is, where the page goes, and what this run was."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="report.html, summary.md and results.csv go here")
    ap.add_argument("--shapes", required=True, help="the shapes.csv simulate_shapes generated (OUT/generated)")
    ap.add_argument("--externs", help="the externs.json simulate_shapes generated (default: beside --shapes)")
    ap.add_argument("--layer", default=os.path.join(HERE, "tpu_v6e_qwen3moe_layer_torchxla.csv"))
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--commit-url")
    ap.add_argument("--ref")
    ap.add_argument("--run-url")
    ap.add_argument("--artifact")
    ap.add_argument("--baseline", help="main's results.json, to set each workload against")
    ap.add_argument("--baseline-commit", help="the main commit --baseline was run at")
    args = ap.parse_args()

    with open(args.results) as fh:
        results = json.load(fh)
    real = measured(args.layer)
    externs_json = args.externs or os.path.join(os.path.dirname(args.shapes), "externs.json")
    externs = {}
    if os.path.isfile(externs_json):
        with open(externs_json) as fh:
            externs = json.load(fh)
    rows = join_rows(args.shapes, real, results, externs)
    gemm_report().attach_baseline(rows, args.baseline, KEYS)
    under_cap = suite_cap()
    chips = count_chips(rows, *unbundled(rows, real, under_cap))
    meta = {"commit": args.commit, "commit_url": args.commit_url, "ref": args.ref, "run_url": args.run_url,
            "baseline_commit": args.baseline and (args.baseline_commit or "unknown"),
            "artifact": args.artifact, "archive_commit": next(iter(real.values()))["archive_commit"],
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "ran_on": gemm_report().ran_on(results["machine"])}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.html"), "w") as fh:
        fh.write(emit_html(rows, real, chips, meta, under_cap))
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(emit_summary(rows, real, chips, meta, under_cap))
    emit_csv(os.path.join(args.out, "results.csv"), rows)
    print(f"{len(rows)} workloads, {sum(r['skipped'] for r in rows)} over the CI cap, "
          f"{sum(bool(r['incomplete']) for r in rows)} incomplete, "
          f"{sum(r['ratio'] is None and not r['skipped'] for r in rows)} failed -> "
          f"{os.path.join(args.out, 'report.html')}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TOGSim vs TPU v6e layer</title>
<style>
:root{--bg:#f4f5f3;--panel:#fff;--fg:#1c2321;--muted:#5d6864;--line:#d6dcd9;--accent:#2f6b5e;--warn:#b5562a;
--ok:#e3efe9;--mid:#f5ecd6;--bad:#f3dcd6;--fast:#dfe7f3;--fail:#e2e2e2;--skip:#efefec;--ext:#ece4f2;
--sans:system-ui,-apple-system,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;--warn:#e59a6c;
--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--skip:#232a28;--ext:#2f2638;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;--warn:#e59a6c;
--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--skip:#232a28;--ext:#2f2638;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.6;margin:0}
a{color:var(--accent)}
.wrap{max-width:1240px;margin:0 auto;padding-inline:16px;padding-block:28px 56px;display:grid;gap:22px}
h1{font-size:24px;margin:0;text-wrap:balance}h2{font-size:17px;margin:0 0 10px;text-wrap:balance}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted)}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.chips{display:flex;flex-wrap:wrap;gap:10px}
.chip{border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--panel)}
.chip b{font-family:var(--mono);font-size:20px;display:block}.chip span{font-size:12.5px;color:var(--muted)}
.chip.alert{border-color:var(--warn)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px;width:100%}
th,td{border:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}
th{font-weight:500;color:var(--muted);font-size:12px;background:var(--panel);text-align:center}
td.l{text-align:left}td.r{font-family:var(--mono);text-align:center}
td.ok{background:var(--ok)}td.mid{background:var(--mid)}td.bad{background:var(--bad)}td.fast{background:var(--fast)}td.fail{background:var(--fail)}td.skip{background:var(--skip);color:var(--muted)}td.ext{background:var(--ext)}
.w{color:var(--warn);font-weight:600}
#raw th{cursor:pointer;user-select:none}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12.5px;color:var(--muted);margin-top:8px;align-items:center}
.sw{display:inline-block;width:12px;height:12px;border:1px solid var(--line);vertical-align:-2px;margin-right:4px}
p,li{max-width:80ch}ul{margin:6px 0;padding-left:20px}
</style></head><body>
<div class="wrap">
<header style="display:grid;gap:6px">
<div class="meta">@@HEAD@@</div>
<h1>TOGSim vs TPU v6e: @@N@@ Qwen3-30B-A3B layer workloads (the decoder layer · each op alone)</h1>
<p class="meta" style="margin:0">Every workload runs the kernels PyTorchSim launches for it (the bundle.json this run used is generated/bundle.json in the artifact): the decoder layer as one program (prefill over T = S tokens, decode one token over a cache of S), and each of its ops alone at T tokens; a workload of several kernels is timed as the sum of its launches.</p>
</header>
<section class="chips" id="chips"></section>
<section class="box" id="failbox" hidden><h2>Simulation failures</h2>
<p class="meta">These kernels exited non-zero in compilation or simulation. Their logs are under logs/ in the artifact.</p>
<div class="scroll"><table id="fails"></table></div></section>
<section class="box" id="extbox" hidden><h2>Incomplete: calls left to aten</h2>
<p class="meta">PyTorchSim's bundle holds no kernel for these calls: Inductor leaves them to aten (extern kernels and fallbacks), so they run unsimulated and the workload's TOGSim time covers only its other kernels. Each is shown with that time and left out of every ratio chip, geomean, the decomposition and the error targets. Kernel routes for them are tracked in @@EXTERN_ISSUE@@.</p>
<div class="scroll"><table id="exts"></table></div></section>
<section class="box" id="missbox" hidden><h2>Measured, not in the bundle</h2>
<p class="meta">The measured table has these workloads and PyTorchSim's bundle has no kernels for them: they are not timed.</p>
<p class="meta" id="miss"></p></section>
<section class="box" id="overbox" hidden><h2>Measured, over the CI cap, not bundled</h2>
<p class="meta">PyTorchSim bundles this suite's workloads under the CI cap only (validation/layer/simulate_shapes.py under_cap).</p>
<p class="meta" id="over"></p></section>
<section class="box"><h2>The layer (TOGSim ÷ measured)</h2>
 <div class="scroll"><table id="layers"></table></div></section>
<section class="box"><h2>Each op alone, per T (TOGSim ÷ measured)</h2>
 <div class="scroll"><table id="ops"></table></div>
 <div class="legend"><span><i class="sw" style="background:var(--ok)"></i>within ±10%</span><span><i class="sw" style="background:var(--mid)"></i>±10~20%</span><span><i class="sw" style="background:var(--bad)"></i>&gt;20% slower</span><span><i class="sw" style="background:var(--fast)"></i>&gt;20% faster</span><span><i class="sw" style="background:var(--fail)"></i>failed</span><span><i class="sw" style="background:var(--skip)"></i>over the CI cap, not run</span><span><i class="sw" style="background:var(--ext)"></i>extern: calls left to aten, not judged</span><span><span class="w">⚠</span> measured status warn (hover for its reason)</span></div></section>
<section class="box"><h2>The layer against the sum of its ops</h2>
 <p><b>The measured op set lacks @@UNCOVERED@@</b> (PSAL-POSTECH/tpu-perf-archive#5 for the projections and the router gate), so Σ ops does not cover the layer yet. The rest (layer − Σ ops) is that uncovered part plus whatever fusion or overlap saves, on each side; it is shown, not hidden.</p>
 <div class="scroll"><table id="dec"></table></div>
 <p class="meta" id="decnote"></p></section>
<section class="box"><h2>All values</h2>
 <p class="meta">Click a column header to sort. p10/p90 is the measured spread; "sim wall s" is how long the workload took to compile and time.</p>
 <div class="scroll"><table id="raw"></table></div></section>
<section class="box"><h2>Reading notes</h2><ul>
 <li>Every workload is compiled for and timed on the target in the header line; a workload that ran on anything else is a failure, not a number. Output values are not checked here (Spike runs only before timing a kernel with an indirect DMA, for the index dump TOGSim times its gathers by; nothing is verified), and no constant is fitted: the ratio is the simulated time over the measured <code>latency_ns</code> as is.</li>
 <li>A measured row of status warn (⚠) is kept and shown with its reason: the grouped matmuls' p10–p90 reaches 60–104% of the median, and layers past S 1024 had no CPU check.</li>
 <li>Σ ops ratio near the layer ratio says the layer's error is the ops' own; a rest ratio far from 1 says the two sides differ in the uncovered part or in what fusion and overlap save. Neither side's rest is a measurement of fusion alone while the op set lacks the uncovered ops.</li>
 <li>Workloads over the CI cap (more than 4096 tokens: a layer's S, an op's T; simulate_shapes.CI_TOKENS) are in shapes.csv and this page but not timed in CI; run them with simulate_shapes.py without --ci.</li>
 <li>A workload is <b>incomplete</b> when PyTorchSim's manifest lists calls Inductor left to aten for it (extern kernels and fallbacks: no kernel in the bundle, nothing simulated for them). Its TOGSim time is of its other kernels only; the raw table shows that ratio marked (extern), and no chip, geomean, decomposition or error target counts it.</li>
 <li>vs main is this run's TOGSim time divided by the one main's last report recorded for the same workload; above 1 means this run is slower.</li>
</ul></section>
</div>
<script>
const D=@@DATA@@,R=D.rows;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const gcls=q=>q===null||q===undefined?'fail':Math.abs(q-1)<=.1?'ok':Math.abs(q-1)<=.2?'mid':q>1?'bad':'fast';
const cls=r=>r.skipped?'skip':r.ratio!==null&&r.incomplete?'ext':gcls(r.ratio);
const show=r=>r.skipped?'cap':r.ratio===null?'failed':r.incomplete?'extern':r.ratio.toFixed(2);
const fx=(v,d)=>v===null||v===undefined?'–':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const warn=r=>r.status!=='ok'?' <span class="w">⚠</span>':'';
const say=r=>(r.skipped?'over the CI cap, not run':r.ratio===null?`failed: ${r.failed_step||'not run'} (rc ${r.rc})`:`measured ${r.real_us.toFixed(3)} µs · TOGSim ${r.sim_us.toFixed(3)} µs`)+(r.incomplete?` · incomplete, ${r.incomplete}`:'')+` · ${r.kernels} kernel(s) · ${r.kernel} · measured ${r.status}${r.status_reason?': '+r.status_reason:''}`;
const cell=r=>r?`<td class="r ${cls(r)}" title="${esc(say(r))}">${show(r)}${warn(r)}</td>`:'<td>–</td>';
document.getElementById('chips').innerHTML=D.chips.map(([b,s,a])=>`<div class="chip${a?' alert':''}"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join('');
if(D.missing.length){document.getElementById('missbox').hidden=false;document.getElementById('miss').textContent=D.missing.join(' · ')}
if(D.over.length){document.getElementById('overbox').hidden=false;document.getElementById('over').textContent=D.over.join(' · ')}
const ext=R.filter(r=>r.incomplete&&!r.skipped);
if(ext.length){document.getElementById('extbox').hidden=false;
 document.getElementById('exts').innerHTML='<tr><th>workload</th><th>kernels timed</th><th>TOGSim µs (kernels only)</th><th>measured µs</th><th>calls left to aten</th></tr>'+ext.map(r=>
 `<tr><td class="l">${esc(r.section+' '+r.name+' T'+r.T+(r.section==='layer'?' S'+r.S:''))}</td><td>${r.kernels}</td><td>${r.sim_us===null?'failed':fx(r.sim_us,1)}</td><td>${fx(r.real_us,1)}</td><td class="l">${esc(r.externs.join(', '))}</td></tr>`).join('')}
const failed=R.filter(r=>r.ratio===null&&!r.skipped);
if(failed.length){document.getElementById('failbox').hidden=false;
 document.getElementById('fails').innerHTML='<tr><th>workload</th><th>kernel</th><th>step</th><th>rc</th><th>log</th></tr>'+failed.map(r=>
 `<tr><td class="l">${esc(r.section+' '+r.name+' T'+r.T+' S'+r.S)}</td><td class="l">${esc(r.kernel)}</td><td>${esc(r.failed_step||'not run')}</td><td>${esc(r.rc)}</td><td>${r.log?`<a href="${esc(r.log)}">${esc(r.log)}</a>`:'–'}</td></tr>`).join('')}
(function(){const L=R.filter(r=>r.section==='layer').sort((a,b)=>(a.name<b.name?1:a.name>b.name?-1:0)||a.S-b.S);
 document.getElementById('layers').innerHTML='<tr><th>layer</th><th>T</th><th>S</th><th>measured µs</th><th>TOGSim µs</th><th>ratio</th><th>measured status</th></tr>'+
 L.map(r=>`<tr><td class="l">${r.name}</td><td>${r.T}</td><td>${r.S}</td><td>${fx(r.real_us,1)}</td><td>${r.sim_us===null?(r.skipped?'not run':'failed'):fx(r.sim_us,1)+(r.incomplete?' (kernels only)':'')}</td>${cell(r)}<td class="l">${esc(r.status)}${r.status_reason?': '+esc(r.status_reason):''}</td></tr>`).join('')})();
(function(){const O=R.filter(r=>r.section==='op'),Ts=[...new Set(O.map(r=>r.T))].sort((a,b)=>a-b),names=[...D.ops.filter(o=>O.some(r=>r.name===o)),...[...new Set(O.map(r=>r.name))].filter(o=>!D.ops.includes(o)).sort()];
 let h='<tr><th>op \\ T</th>'+Ts.map(t=>`<th>${t}</th>`).join('')+'</tr>';
 for(const o of names)h+=`<tr><th>${o}</th>`+Ts.map(t=>cell(O.find(r=>r.name===o&&r.T===t))).join('')+'</tr>';
 document.getElementById('ops').innerHTML=O.length?h:'<tr><td>no op workloads</td></tr>'})();
(function(){document.getElementById('dec').innerHTML='<tr><th rowspan="2">layer</th><th rowspan="2">T</th><th rowspan="2">S</th><th colspan="3">measured µs</th><th colspan="3">TOGSim µs</th><th colspan="3">TOGSim ÷ measured</th></tr><tr><th>layer</th><th>Σ ops</th><th>rest</th><th>layer</th><th>Σ ops</th><th>rest</th><th>Σ ops</th><th>rest</th><th>layer</th></tr>'+
 D.dec.map(d=>`<tr><td class="l">${d.name}</td><td>${d.T}</td><td>${d.S}</td><td>${fx(d.layer_real_us,1)}</td><td>${fx(d.ops_real_us,1)}</td><td>${fx(d.rest_real_us,1)}</td><td>${d.layer_incomplete?'extern':fx(d.layer_sim_us,1)}</td><td>${d.ops_sim_us===null?`${d.ops_timed}/${d.ops} timed`:fx(d.ops_sim_us,1)}</td><td>${fx(d.rest_sim_us,1)}</td><td class="r ${d.ops_ratio===null?'skip':gcls(d.ops_ratio)}">${fx(d.ops_ratio,3)}</td><td class="r">${fx(d.rest_ratio,3)}</td><td class="r ${d.layer_ratio===null?'skip':gcls(d.layer_ratio)}">${fx(d.layer_ratio,3)}</td></tr>`).join('');
 document.getElementById('decnote').textContent='Σ ops counts each op as often as the layer runs it: '+Object.entries(D.mult).map(([o,n])=>n>1?`${n}× ${o}`:o).join(', ')+'. A side whose ops are not all timed (an incomplete op counts as not timed) has no sum.'})();
const cols=[['section',r=>r.section,'s'],['name',r=>r.name,'s'],['T',r=>r.T,0],['S',r=>r.S,0],['kernels',r=>r.kernels,0],['launches',r=>r.launches,0],
 ['measured µs',r=>r.real_us,3],['p10 µs',r=>r.p10_us,3],['p90 µs',r=>r.p90_us,3],['TOGSim µs',r=>r.sim_us,3],['ratio',r=>r.ratio,3],['extern',r=>r.externs.join(', '),'s'],
 ['TOGSim cycles',r=>r.total_cycles,0],['sim wall s',r=>r.wall_s,1],['vs main',r=>r.vs_main,3],['measured status',r=>r.status,'s'],['status reason',r=>r.status_reason,'s']];
let sortK=null,sortD=1;
function raw(){const a=[...R];if(sortK!==null)a.sort((p,o)=>{const f=cols[sortK][1],u=f(p),v=f(o);if(u===null||v===null||u===undefined||v===undefined)return ((u??null)===null)-((v??null)===null);return (u>v?1:u<v?-1:0)*sortD});
 let t='<tr>'+cols.map((c,i)=>`<th data-i="${i}">${c[0]}${sortK===i?(sortD>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>';
 for(const r of a)t+='<tr>'+cols.map(([n,f,d])=>{const v=f(r);if(d==='s')return `<td class="l" title="${esc(r.kernel)}">${esc(v)}</td>`;
  if(n==='ratio')return `<td class="${cls(r)}">${r.skipped?'cap':v===null?'failed':v.toFixed(d)+(r.incomplete?' (extern)':'')}</td>`;
  if(n==='TOGSim µs'&&v===null)return r.skipped?'<td class="skip">not run</td>':`<td class="fail">${r.log?`<a href="${esc(r.log)}">failed</a>`:'failed'}</td>`;
  return `<td>${fx(v,d)}</td>`}).join('')+'</tr>';
 const el=document.getElementById('raw');el.innerHTML=t;el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const i=+th.dataset.i;sortD=(sortK===i)?-sortD:1;sortK=i;raw()})}
raw();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
