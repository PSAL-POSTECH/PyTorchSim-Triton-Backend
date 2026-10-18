#!/usr/bin/env python3
"""results.json + shapes.csv + the measured v6e attention rows -> report.html (self-contained: inline SVG and vanilla
JS), summary.md and results.csv. Raw simulated time against raw measured time, nothing fitted; a failed kernel is shown,
not dropped, and a shape left out of the run (over CI_CAP) is shown as such, not as a failure. The kernels are the ones
PyTorchSim launches for SDPA (validation/pytorchsim); with --baseline (main's results.json) each workload is also set
against main's simulated time."""
import argparse
import csv
import datetime
import html
import importlib.util
import json
import math
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WORST = 10
KINDS = ("prefill", "decode")


def gemm_report():
    """validation/gemm/emit_report.py under its own module name: the machine line is said one way."""
    path = os.path.join(os.path.dirname(HERE), "gemm", "emit_report.py")
    spec = importlib.util.spec_from_file_location("gemm_emit_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measured(attention_csv):
    """(kind, Hq, Hkv, S) -> the archive row measured for it."""
    with open(attention_csv, newline="") as fh:
        return {(r["kind"], int(r["Hq"]), int(r["Hkv"]), int(r["S"])): r for r in csv.DictReader(fh)}


def join_rows(shapes_csv, real, results):
    """One row per shape: its kernel and blocks, its measured and its simulated time, the archive's FLOP and bytes;
    `skipped` when the run left it out because shapes.csv marks it over the CI cap."""
    sim = {r["kernel"]: r for r in results["shapes"]}
    rows = []
    with open(shapes_csv, newline="") as fh:
        for s in csv.DictReader(fh):
            key = (s["kind"], int(s["Hq"]), int(s["Hkv"]), int(s["S"]))
            if key not in real:
                raise SystemExit(f"no measurement for {key}")
            t, got = real[key], sim.get(s["kernel"], {})
            flops, nbytes = int(t["flops"]), int(t["bytes"])
            real_us = int(t["latency_ns"]) / 1000
            sim_us = got.get("us") if got.get("rc") == 0 else None
            skipped = not got and s.get("ci") == "0"
            rows.append({
                "kind": key[0], "Hq": key[1], "Hkv": key[2], "S": key[3], "kernel": s["kernel"],
                "kernels": int(s["kernels"]), "launches": int(s["launches"]), "ci": s.get("ci") != "0",
                "skipped": skipped, "flops": flops, "bytes": nbytes, "ai": flops / nbytes,
                "real_us": real_us, "real_kernel_us": float(t["kernel_us"]), "sim_us": sim_us,
                "ratio": None if sim_us is None else sim_us / real_us,
                "real_tf": flops / real_us / 1e6, "sim_tf": None if sim_us is None else flops / sim_us / 1e6,
                "real_gbps": nbytes / real_us / 1e3, "sim_gbps": None if sim_us is None else nbytes / sim_us / 1e3,
                "status": t["status"], "run_id": t["run_id"], "total_cycles": got.get("total_cycles"),
                "wall_s": got.get("wall_s"), "rc": got.get("rc", "over the CI cap" if skipped else "not run"),
                "failed_step": got.get("failed_step"), "log": got.get("log"), "target": got.get("target"),
                "accel": got.get("accel"), "model": got.get("model"), "togsim_config": got.get("togsim_config")})
    return rows


def stats(rs):
    """n, simulated, within ±10% / ±20%, median |error| and the ratio geomean of the rows of rs that were run."""
    cap = sum(r["skipped"] for r in rs)
    rs = [r for r in rs if not r["skipped"]]
    ok = sorted(abs(r["ratio"] - 1) for r in rs if r["ratio"] is not None)
    if not ok:
        return {"n": len(rs), "done": 0, "cap": cap}
    gm = math.exp(sum(math.log(r["ratio"]) for r in rs if r["ratio"] is not None) / len(ok))
    return {"n": len(rs), "done": len(ok), "w10": sum(e <= .1 for e in ok), "w20": sum(e <= .2 for e in ok),
            "med": ok[len(ok) // 2], "gm": gm, "cap": cap}


def breakdown(rows):
    """stats per kind, and per kind and S."""
    out = []
    for k in KINDS:
        rs = [r for r in rows if r["kind"] == k]
        if rs:
            out.append(dict(stats(rs), kind=k, S="all"))
        for S in sorted({r["S"] for r in rs}):
            out.append(dict(stats([r for r in rs if r["S"] == S]), kind=k, S=S))
    return out


def count_chips(rows):
    """The headline counts: shapes simulated, within ±10% / ±20% of those, each kind's ratio geomean, the failures
    and the shapes left out by the CI cap; each a (value, label, alert) triple."""
    run = [r for r in rows if not r["skipped"]]

    def within(tol):
        return sum(1 for r in run if r["ratio"] is not None and abs(r["ratio"] - 1) <= tol)
    chips = [(f"{len(run)} / {len(rows)}", "shapes run", False),
             (f"{within(.10)} / {len(run)}", "run, within ±10%", False),
             (f"{within(.20)} / {len(run)}", "run, within ±20%", False)]
    for k in KINDS:
        s = stats([r for r in rows if r["kind"] == k])
        chips.append(("–" if not s["done"] else f"{s['gm']:.3f}", f"{k} ratio geomean", False))
    failed = sum(1 for r in run if r["ratio"] is None)
    if failed:
        chips.append((f"{failed} / {len(run)}", "simulation failed", True))
    skipped = len(rows) - len(run)
    if skipped:
        chips.append((f"{skipped}", "over the CI cap, not run", False))
    return chips


def emit_summary(rows, chips, meta):
    """Markdown for a job summary: the chips, the per-kind table, failures, shapes not run, the worst outliers."""
    out = [f"### Attention validation: TOGSim vs TPU v6e ({len(rows)} shapes) @ `{meta['commit'][:12]}`", "",
           f"Ran on {meta['ran_on']}. Kernels are PyTorchSim's for SDPA (causal prefill, one-token decode); CI times "
           f"the shapes of at most 2^26 score elements (validation/attention/simulate_shapes.py).", "",
           "| " + " | ".join(label for _, label, _ in chips) + " |", "|" + "---|" * len(chips),
           "| " + " | ".join(f"**{value}**" for value, _, _ in chips) + " |", "",
           "| kind | S | run | ±10% | ±20% | median abs error | ratio geomean |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in breakdown(rows):
        run = f"{s['done']} / {s['n']}" + (f" ({s['cap']} over the cap)" if s["cap"] else "")
        if s["done"]:
            out.append(f"| {s['kind']} | {s['S']} | {run} | {s['w10']} | {s['w20']} | "
                       f"{100 * s['med']:.1f}% | {s['gm']:.3f} |")
        else:
            out.append(f"| {s['kind']} | {s['S']} | {run} | – | – | – | – |")
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
    worst = sorted((r for r in rows if r["ratio"] is not None), key=lambda r: -abs(math.log(r["ratio"])))[:WORST]
    if worst:
        out += [f"Largest deviations (TOGSim ÷ measured, {len(worst)} of {len(rows)}):", "",
                "| kind | Hq/Hkv | S | kernels | measured µs | TOGSim µs | ratio |", "|---|---|---:|---:|---:|---:|---:|"]
        out += [f"| {r['kind']} | {r['Hq']}/{r['Hkv']} | {r['S']} | {r['kernels']} | {r['real_us']:.3f} "
                f"| {r['sim_us']:.3f} | {r['ratio']:.3f} |" for r in worst]
        out.append("")
    out += gemm_report().baseline_summary(rows, lambda r: f"{r['kind']} {r['Hq']}/{r['Hkv']} S{r['S']}",
                                          meta.get("baseline_commit"))
    if meta.get("artifact"):
        out.append(f"Full report: `report.html` in artifact `{meta['artifact']}`.")
    return "\n".join(out) + "\n"


def emit_csv(path, rows):
    """The joined table, one line per shape."""
    cols = ["kind", "Hq", "Hkv", "S", "kernel", "kernels", "launches", "base_us", "vs_main", "ci", "flops", "bytes", "ai", "real_us",
            "real_kernel_us", "sim_us", "ratio", "real_tf", "sim_tf", "real_gbps", "sim_gbps", "status", "run_id",
            "total_cycles", "wall_s", "rc", "failed_step", "log", "target", "accel", "model", "togsim_config"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def emit_html(rows, chips, meta, device):
    """The page: header, chips, failures, two error tables, roofline, per-kind table, raw table, notes."""
    peak, bw = float(device["peak_bf16_tflops"]), float(device["hbm_gbps"])
    sha = html.escape(meta["commit"][:7])
    commit = f'<a href="{html.escape(meta["commit_url"])}">{sha}</a>' if meta.get("commit_url") else sha
    run = f' · <a href="{html.escape(meta["run_url"])}">CI run</a>' if meta.get("run_url") else ""
    ref = f' ({html.escape(meta["ref"])})' if meta.get("ref") else ""
    runs = ", ".join(sorted({r["run_id"] for r in rows}))
    head = (f'{meta["date"]} · PTB {commit}{ref}{run} · {html.escape(meta["ran_on"])} · measured torchxla v6e '
            f'({html.escape(runs)}, tpu-perf-archive {html.escape(meta["archive_commit"])})')
    data = {"rows": rows, "chips": chips, "peak": peak, "bw": bw, "kinds": [k for k in KINDS
                                                                           if any(r["kind"] == k for r in rows)]}
    return (PAGE.replace("@@N@@", str(len(rows))).replace("@@HEAD@@", head)
            .replace("@@PEAK@@", f"{peak:,.0f}").replace("@@BW@@", f"{bw:,.1f}")
            .replace("@@KNEE@@", f"{peak * 1000 / bw:,.0f}")
            .replace("@@DATA@@", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))


def main():
    """Command line: where results.json is, where the page goes, and what this run was."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="report.html, summary.md and results.csv go here")
    ap.add_argument("--shapes", required=True, help="the shapes.csv simulate_shapes generated (OUT/generated)")
    ap.add_argument("--attention", default=os.path.join(HERE, "tpu_v6e_attention_torchxla.csv"))
    ap.add_argument("--device", default=os.path.join(os.path.dirname(HERE), "gemm", "tpu_v6e.yaml"))
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
    with open(args.device) as fh:
        device = yaml.safe_load(fh)
    real = measured(args.attention)
    rows = join_rows(args.shapes, real, results)
    gemm_report().attach_baseline(rows, args.baseline, ("kind", "Hq", "Hkv", "S"))
    chips = count_chips(rows)
    meta = {"commit": args.commit, "commit_url": args.commit_url, "ref": args.ref, "run_url": args.run_url,
            "baseline_commit": args.baseline and (args.baseline_commit or "unknown"),
            "artifact": args.artifact, "archive_commit": next(iter(real.values()))["archive_commit"],
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "ran_on": gemm_report().ran_on(results["machine"])}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.html"), "w") as fh:
        fh.write(emit_html(rows, chips, meta, device))
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(emit_summary(rows, chips, meta))
    emit_csv(os.path.join(args.out, "results.csv"), rows)
    print(f"{len(rows)} shapes, {sum(r['skipped'] for r in rows)} over the CI cap, "
          f"{sum(r['ratio'] is None and not r['skipped'] for r in rows)} failed -> "
          f"{os.path.join(args.out, 'report.html')}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TOGSim vs TPU v6e attention</title>
<style>
:root{--bg:#f4f5f3;--panel:#fff;--fg:#1c2321;--muted:#5d6864;--line:#d6dcd9;--accent:#2f6b5e;
--roof:#1c2321;--ok:#e3efe9;--mid:#f5ecd6;--bad:#f3dcd6;--fast:#dfe7f3;--fail:#e2e2e2;--skip:#efefec;
--p0:#b5562a;--p1:#4d5fa8;
--sans:system-ui,-apple-system,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;
--roof:#e3e9e6;--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--skip:#232a28;--p0:#e59a6c;--p1:#93a5ef;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;
--roof:#e3e9e6;--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--skip:#232a28;--p0:#e59a6c;--p1:#93a5ef;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.6;margin:0}
a{color:var(--accent)}
.wrap{max-width:1240px;margin:0 auto;padding-inline:16px;padding-block:28px 56px;display:grid;gap:22px}
h1{font-size:24px;margin:0;text-wrap:balance}h2{font-size:17px;margin:0 0 10px;text-wrap:balance}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted)}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.chips{display:flex;flex-wrap:wrap;gap:10px}
.chip{border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--panel)}
.chip b{font-family:var(--mono);font-size:20px;display:block}.chip span{font-size:12.5px;color:var(--muted)}
.chip.alert{border-color:var(--p0)}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:16px}
@media (max-width:960px){.grid2{grid-template-columns:minmax(0,1fr)}}
.stack{display:grid;gap:16px;min-width:0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px;width:100%}
th,td{border:1px solid var(--line);padding:4px 6px;text-align:center;white-space:nowrap}
th{font-weight:500;color:var(--muted);font-size:12px;background:var(--panel)}
td.r{font-family:var(--mono)}
td.ok{background:var(--ok)}td.mid{background:var(--mid)}td.bad{background:var(--bad)}td.fast{background:var(--fast)}td.fail{background:var(--fail)}td.skip{background:var(--skip);color:var(--muted)}
#raw td{text-align:right}#raw td.l{text-align:left}#raw th{cursor:pointer;user-select:none}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12.5px;color:var(--muted);margin-top:8px;align-items:center}
.sw{display:inline-block;width:12px;height:12px;border:1px solid var(--line);vertical-align:-2px;margin-right:4px}
.chart{position:relative}.chart svg{display:block;width:100%;height:auto;font-family:var(--sans)}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;max-width:320px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none}
.ctrl{display:grid;gap:6px;margin-bottom:8px}
.crow{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.clab{font-size:12px;color:var(--muted);width:5.2em}
.tg{font:inherit;font-size:12.5px;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:999px;padding:2px 10px;cursor:pointer}
.tg[aria-pressed="false"]{opacity:.45;text-decoration:line-through}
.tg:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.tg .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:0}
.tg.small{padding:2px 8px;color:var(--muted)}
p,li{max-width:80ch}ul{margin:6px 0;padding-left:20px}
</style></head><body>
<div class="wrap">
<header style="display:grid;gap:6px">
<div class="meta">@@HEAD@@</div>
<h1>TOGSim vs TPU v6e: @@N@@ attention shapes (causal prefill · paged decode)</h1>
<p class="meta" style="margin:0">Every workload runs the kernels PyTorchSim launches for scaled_dot_product_attention at that shape (validation/pytorchsim; the bundle.json this run used is generated/bundle.json in the artifact): prefill causal over S queries, decode one query over S keys, head_dim 128, GQA; a workload of several kernels is timed as the sum of its launches.</p>
</header>
<section class="chips" id="chips"></section>
<section class="box" id="failbox" hidden><h2>Simulation failures</h2>
<p class="meta">These kernels exited non-zero in compilation or simulation. Their logs are under logs/ in the artifact.</p>
<div class="scroll"><table id="fails"></table></div></section>
<section class="grid2">
<div class="stack">
 <div class="box"><h2>prefill error table (TOGSim ÷ measured)</h2>
  <div class="scroll"><table id="heat-prefill"></table></div></div>
 <div class="box"><h2>decode error table (TOGSim ÷ measured)</h2>
  <div class="scroll"><table id="heat-decode"></table></div>
  <p class="meta" style="margin-top:8px">Rows are Hq/Hkv, columns are S (prefill: Sq = Skv; decode: the KV length). Above 1 means TOGSim is slower.</p>
  <div class="legend"><span><i class="sw" style="background:var(--ok)"></i>within ±10%</span><span><i class="sw" style="background:var(--mid)"></i>±10~20%</span><span><i class="sw" style="background:var(--bad)"></i>&gt;20% slower</span><span><i class="sw" style="background:var(--fast)"></i>&gt;20% faster</span><span><i class="sw" style="background:var(--fail)"></i>failed</span><span><i class="sw" style="background:var(--skip)"></i>over the CI cap, not run</span></div></div>
</div>
<div class="box"><h2>Roofline (v6e: MXU @@PEAK@@ TFLOP/s, HBM @@BW@@ GB/s)</h2>
 <div class="ctrl">
  <div class="crow"><span class="clab">Series</span><span id="c-series"></span></div>
  <div class="crow"><span class="clab">Kind</span><span id="c-kind"></span></div>
 </div>
 <div class="chart" id="chart-roof"><svg id="roof" viewBox="0 0 640 460" role="img" aria-label="roofline"></svg><div class="tip"></div></div>
 <div class="legend" id="leg-roof"></div>
 <p class="meta">x is arithmetic intensity (the archive's <code>flops</code> ÷ <code>bytes</code>: prefill 4·Hq·D·S(S+1)/2 over 2·D·S·(2Hq + 2Hkv), decode 4·Hq·D·S over 2·D·(2Hq + 2Hkv·S)), y is achieved TFLOP/s. The knee is at @@KNEE@@ FLOP/B: decode is on the HBM side, prefill from S 512 on the MXU side. Measured is a filled circle (●), TOGSim a hollow triangle (△). Double-click a button to show only that one.</p>
</div>
</section>
<section class="box"><h2>Per kind and S</h2>
 <div class="scroll"><table id="bins"></table></div>
 <p class="meta" style="margin-top:8px">run of n shapes (a shape over the CI cap is not counted); error is TOGSim ÷ measured − 1.</p>
</section>
<section class="box"><h2>All values</h2>
 <p class="meta">Click a column header to sort. "kernel µs" is the archive's <code>kernel_us</code>, the Pallas call alone; the ratio is against the whole module's <code>latency_ns</code>. "sim wall s" is how long the shape took to compile and time.</p>
 <div class="scroll"><table id="raw"></table></div></section>
<section class="box"><h2>Reading notes</h2><ul>
 <li>Every shape is compiled for and timed on the target in the header line; a shape that ran on anything else is a failure, not a number. Output values are not checked here (no Spike), and no constant is fitted: the ratio is the simulated time over the measured <code>latency_ns</code> as is.</li>
 <li>The measured time is the whole XLA module: besides the kernel, prefill multiplies q by 1/√D in a fusion and decode prepares RPA's metadata (<code>other_ops_us</code>, 12% and 16% at the median). The simulated kernel scales the scores in place and has no metadata.</li>
 <li>The kernels and their blocks are PyTorchSim's fused SDPA kernels (pytorchsim/frontend/attention.py there), or Inductor's own kernels where PyTorchSim declines the fused route; the kernels column counts them.</li>
 <li>Shapes whose score elements exceed the CI cap (simulate_shapes.CI_CAP) are in shapes.csv and this page but not timed in CI; run them with simulate_shapes.py without --ci.</li>
 <li>vs main is this run's TOGSim time divided by the one main's last report recorded for the same workload; above 1 means this run is slower.</li>
</ul></section>
</div>
<script>
const D=@@DATA@@,R=D.rows,KINDS=D.kinds;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=r=>r.skipped?'skip':r.ratio===null?'fail':Math.abs(r.ratio-1)<=.1?'ok':Math.abs(r.ratio-1)<=.2?'mid':r.ratio>1?'bad':'fast';
const gcls=q=>q===null?'fail':Math.abs(q-1)<=.1?'ok':Math.abs(q-1)<=.2?'mid':q>1?'bad':'fast';
const pcol=k=>`var(--p${KINDS.indexOf(k)})`;
const fx=(v,d)=>v===null||v===undefined?'–':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
document.getElementById('chips').innerHTML=D.chips.map(([b,s,a])=>`<div class="chip${a?' alert':''}"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join('');
const failed=R.filter(r=>r.ratio===null&&!r.skipped);
if(failed.length){document.getElementById('failbox').hidden=false;
 document.getElementById('fails').innerHTML='<tr><th>kind</th><th>Hq/Hkv</th><th>S</th><th>kernel</th><th>step</th><th>rc</th><th>log</th></tr>'+failed.map(r=>
 `<tr><td>${r.kind}</td><td>${r.Hq}/${r.Hkv}</td><td>${r.S}</td><td>${esc(r.kernel)}</td><td>${esc(r.failed_step||'not run')}</td><td>${esc(r.rc)}</td><td>${r.log?`<a href="${esc(r.log)}">${esc(r.log)}</a>`:'–'}</td></tr>`).join('')}
const cell=r=>{if(!r)return '<td>–</td>';const t=r.skipped?`over the CI cap, not run · ${r.kernel}`:r.ratio===null?`failed: ${r.failed_step||'not run'} (rc ${r.rc}) · ${r.kernel}`:
 `measured ${r.real_us.toFixed(3)} µs · TOGSim ${r.sim_us.toFixed(3)} µs · ${r.kernels} kernel(s) · ${r.kernel}`;
 return `<td class="r ${cls(r)}" title="${esc(t)}">${r.skipped?'cap':r.ratio===null?'failed':r.ratio.toFixed(2)}</td>`};
for(const k of ['prefill','decode']){const rs=R.filter(r=>r.kind===k),Hs=[...new Set(rs.map(r=>r.Hq+'/'+r.Hkv))],Ss=[...new Set(rs.map(r=>r.S))].sort((a,b)=>a-b);
 let h='<tr><th>Hq/Hkv \\ S</th>'+Ss.map(s=>`<th>${s}</th>`).join('')+'</tr>';
 for(const hh of Hs)h+=`<tr><th>${hh}</th>`+Ss.map(s=>cell(rs.find(x=>x.Hq+'/'+x.Hkv===hh&&x.S===s))).join('')+'</tr>';
 document.getElementById('heat-'+k).innerHTML=rs.length?h:`<tr><td>no ${k} shapes</td></tr>`}
(function(){const st=rs=>{const cap=rs.filter(r=>r.skipped).length;rs=rs.filter(r=>!r.skipped);const ok=rs.filter(r=>r.ratio!==null),e=ok.map(r=>Math.abs(r.ratio-1)).sort((a,b)=>a-b);
  return {cap,n:rs.length,d:ok.length,w10:e.filter(x=>x<=.1).length,w20:e.filter(x=>x<=.2).length,med:e.length?e[Math.floor(e.length/2)]:null,
   gm:ok.length?Math.exp(ok.reduce((s,r)=>s+Math.log(r.ratio),0)/ok.length):null}};
 let h='<tr><th>kind</th><th>S</th><th>run</th><th>±10%</th><th>±20%</th><th>median |error|</th><th>ratio geomean</th></tr>';
 const pct=v=>v===null?'–':(100*v).toFixed(1)+'%';
 for(const k of KINDS){const rs=R.filter(r=>r.kind===k);for(const S of ['all',...new Set(rs.map(r=>r.S))]){const sub=S==='all'?rs:rs.filter(r=>r.S===S);
  const s=st(sub);h+=`<tr><th>${k}</th><td>${S}</td><td class="r">${s.d} / ${s.n}${s.cap?` (${s.cap} over the cap)`:''}</td><td class="r">${s.w10}</td><td class="r">${s.w20}</td><td class="r">${pct(s.med)}</td><td class="r ${s.gm===null?'skip':gcls(s.gm)}">${s.gm===null?'–':s.gm.toFixed(3)}</td></tr>`}}
 document.getElementById('bins').innerHTML=h;})();
const cols=[['kind',r=>r.kind,'s'],['Hq',r=>r.Hq,0],['Hkv',r=>r.Hkv,0],['S',r=>r.S,0],['kernels',r=>r.kernels,0],['FLOP/B',r=>r.ai,1],
 ['measured µs',r=>r.real_us,3],['kernel µs',r=>r.real_kernel_us,3],['TOGSim µs',r=>r.sim_us,3],['ratio',r=>r.ratio,3],['measured TFLOP/s',r=>r.real_tf,2],['TOGSim TFLOP/s',r=>r.sim_tf,2],
 ['TOGSim cycles',r=>r.total_cycles,0],['sim wall s',r=>r.wall_s,1],['vs main',r=>r.vs_main,3],['measured status',r=>r.status,'s']];
let sortK=null,sortD=1;
function raw(){const a=[...R];if(sortK!==null)a.sort((p,o)=>{const f=cols[sortK][1],u=f(p),v=f(o);if(u===null||v===null||u===undefined||v===undefined)return ((u??null)===null)-((v??null)===null);return (u>v?1:u<v?-1:0)*sortD});
 let t='<tr>'+cols.map((c,i)=>`<th data-i="${i}">${c[0]}${sortK===i?(sortD>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>';
 for(const r of a)t+='<tr>'+cols.map(([n,f,d])=>{const v=f(r);if(d==='s')return `<td class="l" title="${esc(r.kernel)}">${esc(v)}</td>`;
  if(n==='ratio')return `<td class="${cls(r)}">${r.skipped?'cap':v===null?'failed':v.toFixed(d)}</td>`;
  if(n==='TOGSim µs'&&v===null)return r.skipped?'<td class="skip">not run</td>':`<td class="fail">${r.log?`<a href="${esc(r.log)}">failed</a>`:'failed'}</td>`;
  return `<td>${fx(v,d)}</td>`}).join('')+'</tr>';
 const el=document.getElementById('raw');el.innerHTML=t;el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const i=+th.dataset.i;sortD=(sortK===i)?-sortD:1;sortK=i;raw()})}
raw();
const ST={real:true,sim:true,link:true,k:new Set(KINDS)};
function mkBtn(label,on,toggle,only,dot){const b=document.createElement('button');b.className='tg';b.type='button';
 b.innerHTML=(dot?`<i class="dot" style="background:${dot}"></i>`:'')+esc(label);
 let t=null;b.addEventListener('click',()=>{if(t){clearTimeout(t);t=null;only();sync();return}t=setTimeout(()=>{t=null;toggle();sync()},230)});
 b._sync=()=>b.setAttribute('aria-pressed',on());b._sync();return b}
const btns=[];
function group(id,items,all,none){const el=document.getElementById(id);for(const it of items){const b=mkBtn(it.label,it.on,it.toggle,it.only,it.dot);btns.push(b);el.appendChild(b)}
 for(const[txt,fn]of[['all',all],['none',none]]){const x=document.createElement('button');x.className='tg small';x.type='button';x.textContent=txt;x.onclick=()=>{fn();sync()};el.appendChild(x)}}
group('c-series',[{label:'measured TPU',on:()=>ST.real,toggle:()=>ST.real=!ST.real,only:()=>{ST.real=true;ST.sim=false}},
 {label:'TOGSim',on:()=>ST.sim,toggle:()=>ST.sim=!ST.sim,only:()=>{ST.sim=true;ST.real=false}},
 {label:'links',on:()=>ST.link,toggle:()=>ST.link=!ST.link,only:()=>{ST.link=!ST.link}}],()=>{ST.real=ST.sim=ST.link=true},()=>{ST.real=ST.sim=ST.link=false});
group('c-kind',KINDS.map(k=>({label:k,dot:pcol(k),on:()=>ST.k.has(k),toggle:()=>ST.k.has(k)?ST.k.delete(k):ST.k.add(k),only:()=>{ST.k=new Set([k])}})),()=>{ST.k=new Set(KINDS)},()=>{ST.k=new Set()});
function sync(){btns.forEach(b=>b._sync());draw()}
const NS='http://www.w3.org/2000/svg',lg=Math.log10;
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
const tick=v=>v>=1?v.toLocaleString('en-US'):String(+v.toPrecision(1));
function hover(chart,node,text){const tip=chart.querySelector('.tip');node.addEventListener('mouseenter',()=>{tip.textContent=text;tip.style.display='block'});
 node.addEventListener('mousemove',e=>{const b=chart.getBoundingClientRect();let x=e.clientX-b.left+12;if(x+tip.offsetWidth>b.width)x=e.clientX-b.left-tip.offsetWidth-12;tip.style.left=Math.max(0,x)+'px';tip.style.top=(e.clientY-b.top+12)+'px'});
 node.addEventListener('mouseleave',()=>{tip.style.display='none'})}
function tri(x,y,s){return `${x},${y-s} ${x-s*.9},${y+s*.6} ${x+s*.9},${y+s*.6}`}
const span=(vs,lo,hi)=>{const v=vs.filter(x=>x!==null&&x>0);return [Math.floor(lg(Math.min(...v)/lo)),Math.ceil(lg(Math.max(...v)*hi))]};
function draw(){const svg=document.getElementById('roof'),chart=svg.parentNode,W=640,H=460,ML=58,MR=14,MT=12,MB=46;svg.replaceChildren();
 const knee=D.peak*1000/D.bw,[x0,x1]=span([...R.map(r=>r.ai),knee],2,3),[y0,y1]=span([...R.map(r=>r.real_tf),...R.map(r=>r.sim_tf),D.peak],2,1.5);
 const sx=x=>ML+(lg(x)-x0)/(x1-x0)*(W-ML-MR),sy=y=>H-MB-(lg(y)-y0)/(y1-y0)*(H-MT-MB);
 const cp=el('clipPath',{id:'roof-clip'},el('defs',{},svg));el('rect',{x:ML,y:MT,width:W-ML-MR,height:H-MT-MB},cp);
 const ax={style:'stroke:var(--line)','stroke-width':1},txt={style:'fill:var(--muted)','font-size':11};
 for(let e=x0;e<=x1;e++){const x=sx(10**e);el('line',{x1:x,x2:x,y1:MT,y2:H-MB,...ax},svg);el('text',{x,y:H-MB+16,'text-anchor':'middle',...txt},svg).textContent=tick(10**e)}
 for(let e=y0;e<=y1;e++){const y=sy(10**e);el('line',{x1:ML,x2:W-MR,y1:y,y2:y,...ax},svg);el('text',{x:ML-6,y:y+4,'text-anchor':'end',...txt},svg).textContent=tick(10**e)}
 el('rect',{x:ML,y:MT,width:W-ML-MR,height:H-MT-MB,fill:'none',...ax},svg);
 el('text',{x:(ML+W-MR)/2,y:H-8,'text-anchor':'middle',...txt,'font-size':12},svg).textContent='arithmetic intensity (FLOP/byte)';
 el('text',{x:14,y:(MT+H-MB)/2,'text-anchor':'middle',transform:`rotate(-90 14 ${(MT+H-MB)/2})`,...txt,'font-size':12},svg).textContent='TFLOP/s';
 const g=el('g',{'clip-path':'url(#roof-clip)'},svg),xa=10**x0,xb=10**x1;
 el('polyline',{points:`${sx(xa)},${sy(xa*D.bw/1000)} ${sx(knee)},${sy(D.peak)} ${sx(xb)},${sy(D.peak)}`,fill:'none',style:'stroke:var(--roof)','stroke-width':1.5},g);
 const vis=R.filter(r=>ST.k.has(r.kind));
 const say=(w,r,v)=>`${w} ${r.kind} ${r.Hq}/${r.Hkv} S${r.S}: ${v.toFixed(2)} TFLOP/s · measured ${r.real_us.toFixed(3)} µs · TOGSim ${r.sim_us===null?(r.skipped?'not run':'failed'):r.sim_us.toFixed(3)+' µs'} · ratio ${r.ratio===null?'–':r.ratio.toFixed(2)}`;
 if(ST.link&&ST.real&&ST.sim)for(const r of vis)if(r.sim_tf!==null)el('line',{x1:sx(r.ai),y1:sy(r.real_tf),x2:sx(r.ai),y2:sy(r.sim_tf),style:`stroke:${pcol(r.kind)}`,'stroke-width':1,'stroke-dasharray':'3 3'},g);
 if(ST.real)for(const r of vis)hover(chart,el('circle',{cx:sx(r.ai),cy:sy(r.real_tf),r:4.5,style:`fill:${pcol(r.kind)};stroke:${pcol(r.kind)}`},g),say('measured',r,r.real_tf));
 if(ST.sim)for(const r of vis)if(r.sim_tf!==null)hover(chart,el('polygon',{points:tri(sx(r.ai),sy(r.sim_tf),5.5),style:`fill:none;stroke:${pcol(r.kind)}`,'stroke-width':1.6},g),say('TOGSim',r,r.sim_tf));
 const L=[`<span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" style="stroke:var(--roof)" stroke-width="2"/></svg> roof (HBM ${D.bw.toLocaleString('en-US')} GB/s · MXU ${D.peak.toLocaleString('en-US')} TFLOP/s)</span>`];
 for(const k of KINDS)L.push(`<span><i class="sw" style="background:${pcol(k)}"></i>${k}</span>`);
 if(ST.real)L.push('<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" style="fill:var(--muted)"/></svg> measured TPU v6e (●)</span>');
 if(ST.sim)L.push('<span><svg width="14" height="12"><polygon points="7,1 1,10 13,10" style="fill:none;stroke:var(--muted)" stroke-width="1.6"/></svg> TOGSim (△)</span>');
 document.getElementById('leg-roof').innerHTML=L.join('')}
draw();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
