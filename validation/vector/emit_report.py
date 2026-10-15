#!/usr/bin/env python3
"""results.json + shapes.csv + the measured v6e softmax and load rows -> report.html (self-contained: inline SVG and
vanilla JS), summary.md and results.csv. Raw simulated time against raw measured time, nothing fitted; a failed
kernel is shown, not dropped. The kernels are blocked by validation/tiling.py (XLA's v6e windows), not tuned for TOGSim."""
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
PROGRAMS = ("softmax", "max", "sum", "add1", "slice")
FLOPS_PER_ELEMENT = {"softmax": 5, "max": 1, "sum": 1, "add1": 1}
VALU_SLOTS, VREG_ELEMENTS = 4, 8 * 128
BINS = (("< 64K", 0, 1 << 16), ("64K–1M", 1 << 16, 1 << 20), ("1M–16M", 1 << 20, 1 << 24), ("≥ 16M", 1 << 24, 1 << 62))


def gemm_report():
    """validation/gemm/emit_report.py under its own module name: the machine line is said one way."""
    path = os.path.join(os.path.dirname(HERE), "gemm", "emit_report.py")
    spec = importlib.util.spec_from_file_location("gemm_emit_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flops_of(program, R, C):
    """softmax 5 per element (max, subtract, exp, sum, divide); max, sum, add1 1 per element; slice 1."""
    return 1 if program == "slice" else FLOPS_PER_ELEMENT[program] * R * C


def size_bin(R, C):
    return next(name for name, lo, hi in BINS if lo <= R * C < hi)


def measured(softmax_csv, load_csv):
    """(program, R, C) -> the archive row measured for it."""
    real = {}
    with open(softmax_csv, newline="") as fh:
        real.update({("softmax", int(r["R"]), int(r["C"])): r for r in csv.DictReader(fh)})
    with open(load_csv, newline="") as fh:
        real.update({(r["program"], int(r["R"]), int(r["C"])): r for r in csv.DictReader(fh)})
    return real


def join_rows(shapes_csv, real, results):
    """One row per shape: its kernel and form, its measured and its simulated time, bytes and FLOP."""
    sim = {r["kernel"]: r for r in results["shapes"]}
    rows = []
    with open(shapes_csv, newline="") as fh:
        for s in csv.DictReader(fh):
            key = (s["program"], int(s["R"]), int(s["C"]))
            if key not in real:
                raise SystemExit(f"no measurement for {key}")
            t, got = real[key], sim.get(s["kernel"], {})
            nbytes, flops = int(t["bytes"]), flops_of(*key)
            real_us = int(t["latency_ns"]) / 1000
            sim_us = got.get("us") if got.get("rc") == 0 else None
            rows.append({
                "program": key[0], "R": key[1], "C": key[2], "kernel": s["kernel"], "form": s["form"],
                "BM": int(s["BM"]), "bin": size_bin(key[1], key[2]), "bytes": nbytes, "flops": flops,
                "ai": flops / nbytes, "real_us": real_us, "sim_us": sim_us,
                "ratio": None if sim_us is None else sim_us / real_us,
                "real_gf": flops / real_us / 1e3, "sim_gf": None if sim_us is None else flops / sim_us / 1e3,
                "real_gbps": nbytes / real_us / 1e3, "sim_gbps": None if sim_us is None else nbytes / sim_us / 1e3,
                "status": t["status"], "run_id": t["run_id"], "total_cycles": got.get("total_cycles"),
                "rc": got.get("rc", "not run"), "failed_step": got.get("failed_step"), "log": got.get("log"),
                "target": got.get("target"), "accel": got.get("accel"), "model": got.get("model"),
                "togsim_config": got.get("togsim_config")})
    return rows


def stats(rs):
    """n, within ±10% / ±20%, median |error| and the ratio geomean of the simulated rows of rs."""
    ok = sorted(abs(r["ratio"] - 1) for r in rs if r["ratio"] is not None)
    if not ok:
        return {"n": len(rs), "done": 0}
    gm = math.exp(sum(math.log(r["ratio"]) for r in rs if r["ratio"] is not None) / len(ok))
    return {"n": len(rs), "done": len(ok), "w10": sum(e <= .1 for e in ok), "w20": sum(e <= .2 for e in ok),
            "med": ok[len(ok) // 2], "gm": gm}


def breakdown(rows):
    """stats per program, and per program and size bin."""
    out = []
    for p in PROGRAMS:
        rs = [r for r in rows if r["program"] == p]
        if rs:
            out.append(dict(stats(rs), program=p, bin="all"))
        for name, _, _ in BINS:
            sub = [r for r in rs if r["bin"] == name]
            if sub and p != "slice":
                out.append(dict(stats(sub), program=p, bin=name))
    return out


def count_chips(rows):
    """The headline counts: within ±10% / ±20% overall and from 1M elements on, each program's
    ratio geomean, and the failures; each a (value, label, alert) triple."""
    def within(rs, tol):
        return sum(1 for r in rs if r["ratio"] is not None and abs(r["ratio"] - 1) <= tol)
    big = [r for r in rows if r["R"] * r["C"] >= 1 << 20 and r["program"] != "slice"]
    chips = [(f"{within(rows, .10)} / {len(rows)}", "all, within ±10%", False),
             (f"{within(rows, .20)} / {len(rows)}", "all, within ±20%", False),
             (f"{within(big, .20)} / {len(big)}", "≥ 1M elements, within ±20%", False)]
    for p in PROGRAMS[:4]:
        s = stats([r for r in rows if r["program"] == p])
        chips.append(("–" if not s["done"] else f"{s['gm']:.3f}", f"{p} ratio geomean", False))
    failed = sum(1 for r in rows if r["ratio"] is None)
    if failed:
        chips.append((f"{failed} / {len(rows)}", "simulation failed", True))
    return chips


def emit_summary(rows, chips, meta):
    """Markdown for a job summary: the chips, the per-program table, the failures, the worst outliers."""
    out = [f"### Vector validation: TOGSim vs TPU v6e ({len(rows)} shapes) @ `{meta['commit'][:12]}`", "",
           f"Ran on {meta['ran_on']}. Kernels are blocked as XLA blocks them "
           f"(≤ 2^20 elements per program, validation/tiling.py).", "",
           "| " + " | ".join(label for _, label, _ in chips) + " |", "|" + "---|" * len(chips),
           "| " + " | ".join(f"**{value}**" for value, _, _ in chips) + " |", "",
           "| program | elements | n | ±10% | ±20% | median abs error | ratio geomean |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in breakdown(rows):
        if s["done"]:
            out.append(f"| {s['program']} | {s['bin']} | {s['n']} | {s['w10']} | {s['w20']} | "
                       f"{100 * s['med']:.1f}% | {s['gm']:.3f} |")
        else:
            out.append(f"| {s['program']} | {s['bin']} | {s['n']} | – | – | – | – |")
    out.append("")
    failed = [r for r in rows if r["ratio"] is None]
    if failed:
        out += [f"**Failed ({len(failed)})**, shown in the report, logs in the artifact:", ""]
        out += [f"- `{r['kernel']}`: " + (f"{r['failed_step']} step, rc {r['rc']}, `{r['log']}`"
                                         if r["failed_step"] else "not run") for r in failed]
        out.append("")
    worst = sorted((r for r in rows if r["ratio"] is not None), key=lambda r: -abs(math.log(r["ratio"])))[:WORST]
    if worst:
        out += [f"Largest deviations (TOGSim ÷ measured, {len(worst)} of {len(rows)}):", "",
                "| program | R×C | form | measured µs | TOGSim µs | ratio |", "|---|---|---|---:|---:|---:|"]
        out += [f"| {r['program']} | {r['R']}×{r['C']} | {r['form']} BM={r['BM']} | {r['real_us']:.3f} "
                f"| {r['sim_us']:.3f} | {r['ratio']:.3f} |" for r in worst]
        out.append("")
    if meta.get("artifact"):
        out.append(f"Full report: `report.html` in artifact `{meta['artifact']}`.")
    return "\n".join(out) + "\n"


def emit_csv(path, rows):
    """The joined table, one line per shape."""
    cols = ["program", "R", "C", "kernel", "form", "BM", "bytes", "flops", "ai", "real_us", "sim_us", "ratio",
            "real_gbps", "sim_gbps", "real_gf", "sim_gf", "status", "run_id", "total_cycles", "rc", "failed_step",
            "log", "target", "accel", "model", "togsim_config"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def roofs(device, machine):
    """HBM GB/s from the device file; the VALU roof in GFLOP/s = slots x vreg elements x the TOGSim clock."""
    mhz = float((machine or {}).get("core_freq_mhz") or 0)
    return float(device["hbm_gbps"]), VALU_SLOTS * VREG_ELEMENTS * mhz / 1e3, mhz


def emit_html(rows, chips, meta, device, machine):
    """The page: header, chips, failures, two error tables, roofline and bandwidth charts, raw table, notes."""
    bw, valu, mhz = roofs(device, machine)
    sha = html.escape(meta["commit"][:7])
    commit = f'<a href="{html.escape(meta["commit_url"])}">{sha}</a>' if meta.get("commit_url") else sha
    run = f' · <a href="{html.escape(meta["run_url"])}">CI run</a>' if meta.get("run_url") else ""
    ref = f' ({html.escape(meta["ref"])})' if meta.get("ref") else ""
    runs = ", ".join(sorted({r["run_id"] for r in rows}))
    head = (f'{meta["date"]} · PTB {commit}{ref}{run} · {html.escape(meta["ran_on"])} · measured torchxla v6e '
            f'({html.escape(runs)}, tpu-perf-archive {html.escape(meta["archive_commit"])})')
    data = {"rows": rows, "chips": chips, "bw": bw, "valu": valu, "bins": [b[0] for b in BINS],
            "programs": [p for p in PROGRAMS if any(r["program"] == p for r in rows)]}
    return (PAGE.replace("@@N@@", str(len(rows))).replace("@@HEAD@@", head)
            .replace("@@BW@@", f"{bw:,.0f}").replace("@@VALU@@", f"{valu / 1e3:,.2f}")
            .replace("@@MHZ@@", f"{mhz:,.0f}").replace("@@KNEE@@", f"{valu / bw:,.1f}")
            .replace("@@DATA@@", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))


def main():
    """Command line: where results.json is, where the page goes, and what this run was."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="report.html, summary.md and results.csv go here")
    ap.add_argument("--shapes", required=True, help="the shapes.csv simulate_shapes generated (OUT/generated)")
    ap.add_argument("--softmax", default=os.path.join(HERE, "tpu_v6e_softmax_torchxla.csv"))
    ap.add_argument("--load", default=os.path.join(HERE, "tpu_v6e_load_torchxla.csv"))
    ap.add_argument("--device", default=os.path.join(os.path.dirname(HERE), "gemm", "tpu_v6e.yaml"))
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--commit-url")
    ap.add_argument("--ref")
    ap.add_argument("--run-url")
    ap.add_argument("--artifact")
    args = ap.parse_args()

    with open(args.results) as fh:
        results = json.load(fh)
    with open(args.device) as fh:
        device = yaml.safe_load(fh)
    real = measured(args.softmax, args.load)
    rows = join_rows(args.shapes, real, results)
    chips = count_chips(rows)
    meta = {"commit": args.commit, "commit_url": args.commit_url, "ref": args.ref, "run_url": args.run_url,
            "artifact": args.artifact, "archive_commit": next(iter(real.values()))["archive_commit"],
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "ran_on": gemm_report().ran_on(results["machine"])}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.html"), "w") as fh:
        fh.write(emit_html(rows, chips, meta, device, results["machine"]))
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(emit_summary(rows, chips, meta))
    emit_csv(os.path.join(args.out, "results.csv"), rows)
    print(f"{len(rows)} shapes, {sum(r['ratio'] is None for r in rows)} without a simulated time -> "
          f"{os.path.join(args.out, 'report.html')}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TOGSim vs TPU v6e vector</title>
<style>
:root{--bg:#f4f5f3;--panel:#fff;--fg:#1c2321;--muted:#5d6864;--line:#d6dcd9;--accent:#2f6b5e;
--roof:#1c2321;--ok:#e3efe9;--mid:#f5ecd6;--bad:#f3dcd6;--fast:#dfe7f3;--fail:#e2e2e2;
--p0:#b5562a;--p1:#4d5fa8;--p2:#9b3f78;--p3:#2f6b5e;--p4:#5c5c5c;
--sans:system-ui,-apple-system,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;
--roof:#e3e9e6;--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--p0:#e59a6c;--p1:#93a5ef;--p2:#e08cc4;--p3:#74c2ad;--p4:#b8b8b8;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;
--roof:#e3e9e6;--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--p0:#e59a6c;--p1:#93a5ef;--p2:#e08cc4;--p3:#74c2ad;--p4:#b8b8b8;color-scheme:dark}
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
td.ok{background:var(--ok)}td.mid{background:var(--mid)}td.bad{background:var(--bad)}td.fast{background:var(--fast)}td.fail{background:var(--fail)}
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
<h1>TOGSim vs TPU v6e: @@N@@ vector shapes (softmax · max · sum · add1 · slice)</h1>
<p class="meta" style="margin:0">Every kernel is blocked as XLA blocks the same program on v6e: at most 2^20 elements per program (XLA's fusion window), a softmax C that is not a power of two split into two power-of-two pieces, R=1 kept as one row. See validation/vector/emit_kernels.py.</p>
</header>
<section class="chips" id="chips"></section>
<section class="box" id="failbox" hidden><h2>Simulation failures</h2>
<p class="meta">These kernels exited non-zero in compilation or simulation. Their logs are under logs/ in the artifact.</p>
<div class="scroll"><table id="fails"></table></div></section>
<section class="grid2">
<div class="stack">
 <div class="box"><h2>softmax error table (TOGSim ÷ measured)</h2>
  <div class="scroll"><table id="heat-sm"></table></div>
  <p class="meta" style="margin-top:8px">Rows are R, columns are C (the softmax axis). Above 1 means TOGSim is slower.</p></div>
 <div class="box"><h2>load error table (TOGSim ÷ measured)</h2>
  <div class="scroll"><table id="heat-ld"></table></div>
  <p class="meta" style="margin-top:8px">Rows are x[R, C] bf16, columns are the program. slice is measured at three sizes only.</p>
  <div class="legend"><span><i class="sw" style="background:var(--ok)"></i>within ±10%</span><span><i class="sw" style="background:var(--mid)"></i>±10~20%</span><span><i class="sw" style="background:var(--bad)"></i>&gt;20% slower</span><span><i class="sw" style="background:var(--fast)"></i>&gt;20% faster</span><span><i class="sw" style="background:var(--fail)"></i>failed</span></div></div>
</div>
<div class="box"><h2>Roofline (v6e vector: VALU @@VALU@@ TOP/s, HBM @@BW@@ GB/s)</h2>
 <div class="ctrl">
  <div class="crow"><span class="clab">Series</span><span id="c-series"></span></div>
  <div class="crow"><span class="clab">Program</span><span id="c-prog"></span></div>
  <div class="crow"><span class="clab">Elements</span><span id="c-bin"></span></div>
 </div>
 <div class="chart" id="chart-roof"><svg id="roof" viewBox="0 0 640 460" role="img" aria-label="roofline"></svg><div class="tip"></div></div>
 <div class="legend" id="leg-roof"></div>
 <p class="meta">x is arithmetic intensity (FLOP ÷ the archive's <code>bytes</code>: softmax and add1 4·R·C, max and sum 2·R·C, slice 4), y is achieved GFLOP/s. FLOP are counted per element: softmax 5 (max, subtract, exp, sum, divide), max, sum and add1 1; slice is 1 FLOP in all. The flat roof is the VALU: @@VALU@@ TOP/s = 4 VALU slots × an 8×128 vreg × @@MHZ@@ MHz (the slot count from v6e's LLO bundles, the vreg and the clock from the TOGSim machine). The knee is at @@KNEE@@ FLOP/B, so every program here is on the HBM side. One program's points share an intensity, so they stand in a column, larger shapes higher; max/sum and add1/slice share one and are drawn a little apart. Measured is a filled circle (●), TOGSim a hollow triangle (△). Double-click a button to show only that one.</p>
</div>
</section>
<section class="box"><h2>Achieved bandwidth by size</h2>
 <div class="chart" id="chart-bw"><svg id="bw" viewBox="0 0 1000 380" role="img" aria-label="bandwidth by bytes"></svg><div class="tip"></div></div>
 <div class="legend" id="leg-bw"></div>
 <p class="meta">The same points over bytes moved (the buttons above apply). The flat line is HBM @@BW@@ GB/s.</p>
</section>
<section class="box"><h2>Per program and size</h2>
 <div class="scroll"><table id="bins"></table></div>
 <p class="meta" style="margin-top:8px">n simulated of n shapes; error is TOGSim ÷ measured − 1.</p>
</section>
<section class="box"><h2>All values</h2>
 <p class="meta">Click a column header to sort.</p>
 <div class="scroll"><table id="raw"></table></div></section>
<section class="box"><h2>Reading notes</h2><ul>
 <li>Every shape is compiled for and timed on the target in the header line; a shape that ran on anything else is a failure, not a number. Output values are not checked (no Spike), and no constant is fitted: the ratio is the simulated time over the measured <code>latency_ns</code> as is.</li>
 <li>The measured time is the whole XLA module's device time. XLA runs softmax as three ops (reduce, fusion, normalize); TOGSim runs one Triton kernel.</li>
 <li>Blocking follows XLA, not TOGSim: rows ride the lanes with as many rows per program as fit 2^20 elements (XLA's compiled add1 3072×1024 has output_window_bounds [128, 8] of (8,128) tiles = 1024×1024, iteration_bounds [3, 1]). Below 128 rows a softmax spreads C over the lanes in one program; max and sum walk the blocks in one program.</li>
 <li>max, sum and slice store their scalar as f32, not bf16: a scalar f32 → bf16 store lowers to <code>__truncsfbf2</code>, which the RISC-V link does not define (tests/cases/mechanism/ops/ops_scalar_bf16_store.py, blocked).</li>
 <li>Not compared: the softmax table's model-only shapes (not on the grid) and the load table's gemm8, aspect, subtile, ragged and edge series.</li>
</ul></section>
</div>
<script>
const D=@@DATA@@,R=D.rows,PROGS=D.programs,BINS=D.bins;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=q=>q===null?'fail':Math.abs(q-1)<=.1?'ok':Math.abs(q-1)<=.2?'mid':q>1?'bad':'fast';
const pcol=p=>`var(--p${['softmax','max','sum','add1','slice'].indexOf(p)})`;
const fx=(v,d)=>v===null||v===undefined?'–':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
document.getElementById('chips').innerHTML=D.chips.map(([b,s,a])=>`<div class="chip${a?' alert':''}"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join('');
const failed=R.filter(r=>r.ratio===null);
if(failed.length){document.getElementById('failbox').hidden=false;
 document.getElementById('fails').innerHTML='<tr><th>program</th><th>R×C</th><th>kernel</th><th>step</th><th>rc</th><th>log</th></tr>'+failed.map(r=>
 `<tr><td>${r.program}</td><td>${r.R}×${r.C}</td><td>${esc(r.kernel)}</td><td>${esc(r.failed_step||'not run')}</td><td>${esc(r.rc)}</td><td>${r.log?`<a href="${esc(r.log)}">${esc(r.log)}</a>`:'–'}</td></tr>`).join('')}
const cell=r=>{if(!r)return '<td>–</td>';const q=r.ratio,t=q===null?`failed: ${r.failed_step||'not run'} (rc ${r.rc}) · ${r.kernel}`:
 `measured ${r.real_us.toFixed(3)} µs · TOGSim ${r.sim_us.toFixed(3)} µs · ${r.form} BM=${r.BM} · ${r.kernel}`;
 return `<td class="r ${cls(q)}" title="${esc(t)}">${q===null?'failed':q.toFixed(2)}</td>`};
(function(){const sm=R.filter(r=>r.program==='softmax'),Rs=[...new Set(sm.map(r=>r.R))].sort((a,b)=>a-b),Cs=[...new Set(sm.map(r=>r.C))].sort((a,b)=>a-b);
 let h='<tr><th>R \\ C</th>'+Cs.map(c=>`<th>${c}</th>`).join('')+'</tr>';
 for(const rr of Rs)h+=`<tr><th>${rr}</th>`+Cs.map(c=>cell(sm.find(x=>x.R===rr&&x.C===c))).join('')+'</tr>';
 document.getElementById('heat-sm').innerHTML=sm.length?h:'<tr><td>no softmax shapes</td></tr>';
 const ld=R.filter(r=>r.program!=='softmax'),P=PROGS.filter(p=>p!=='softmax');
 const keys=[...new Map(ld.map(r=>[r.R+'x'+r.C,[r.R,r.C]])).values()].sort((a,b)=>a[0]*a[1]-b[0]*b[1]||a[0]-b[0]);
 const kb=n=>n>=2**20?(n/2**20)+' MiB':(n/1024)+' KiB';
 let g='<tr><th>R×C</th><th>x bytes</th>'+P.map(p=>`<th>${p}</th>`).join('')+'</tr>';
 for(const[a,b]of keys)g+=`<tr><th>${a}×${b}</th><th>${kb(2*a*b)}</th>`+P.map(p=>cell(ld.find(x=>x.program===p&&x.R===a&&x.C===b))).join('')+'</tr>';
 document.getElementById('heat-ld').innerHTML=ld.length?g:'<tr><td>no load shapes</td></tr>';})();
(function(){const st=rs=>{const ok=rs.filter(r=>r.ratio!==null),e=ok.map(r=>Math.abs(r.ratio-1)).sort((a,b)=>a-b);
  return {n:rs.length,d:ok.length,w10:e.filter(x=>x<=.1).length,w20:e.filter(x=>x<=.2).length,med:e.length?e[Math.floor(e.length/2)]:null,
   gm:ok.length?Math.exp(ok.reduce((s,r)=>s+Math.log(r.ratio),0)/ok.length):null,
   rr:rs.length?rs.map(r=>r.real_gbps/D.bw).sort((a,b)=>a-b)[Math.floor(rs.length/2)]:null,
   sr:ok.length?ok.map(r=>r.sim_gbps/D.bw).sort((a,b)=>a-b)[Math.floor(ok.length/2)]:null}};
 let h='<tr><th>program</th><th>elements</th><th>simulated</th><th>±10%</th><th>±20%</th><th>median |error|</th><th>ratio geomean</th><th>measured % of HBM (median)</th><th>TOGSim % of HBM (median)</th></tr>';
 const pct=v=>v===null?'–':(100*v).toFixed(1)+'%';
 for(const p of PROGS){const rs=R.filter(r=>r.program===p);for(const b of ['all',...BINS]){const sub=b==='all'?rs:rs.filter(r=>r.bin===b);if(!sub.length||(p==='slice'&&b!=='all'))continue;
  const s=st(sub);h+=`<tr><th>${p}</th><td>${b}</td><td class="r">${s.d} / ${s.n}</td><td class="r">${s.w10}</td><td class="r">${s.w20}</td><td class="r">${pct(s.med)}</td><td class="r ${s.gm===null?'fail':cls(s.gm)}">${s.gm===null?'–':s.gm.toFixed(3)}</td><td class="r">${pct(s.rr)}</td><td class="r">${pct(s.sr)}</td></tr>`}}
 document.getElementById('bins').innerHTML=h;})();
const cols=[['program',r=>r.program,'s'],['R',r=>r.R,0],['C',r=>r.C,0],['form',r=>r.form+' BM='+r.BM,'s'],['bytes',r=>r.bytes,0],['FLOP/B',r=>r.ai,2],
 ['measured µs',r=>r.real_us,3],['TOGSim µs',r=>r.sim_us,3],['ratio',r=>r.ratio,3],['measured GB/s',r=>r.real_gbps,1],['TOGSim GB/s',r=>r.sim_gbps,1],
 ['TOGSim cycles',r=>r.total_cycles,0],['measured status',r=>r.status,'s']];
let sortK=null,sortD=1;
function raw(){const a=[...R];if(sortK!==null)a.sort((p,o)=>{const f=cols[sortK][1],u=f(p),v=f(o);if(u===null||v===null)return (u===null)-(v===null);return (u>v?1:u<v?-1:0)*sortD});
 let t='<tr>'+cols.map((c,i)=>`<th data-i="${i}">${c[0]}${sortK===i?(sortD>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>';
 for(const r of a)t+='<tr>'+cols.map(([n,f,d])=>{const v=f(r);if(d==='s')return `<td class="l" title="${esc(r.kernel)}">${esc(v)}</td>`;
  if(n==='ratio')return `<td class="${cls(v)}">${v===null?'failed':v.toFixed(d)}</td>`;
  if(n==='TOGSim µs'&&v===null)return `<td class="fail">${r.log?`<a href="${esc(r.log)}">failed</a>`:'failed'}</td>`;
  return `<td>${fx(v,d)}</td>`}).join('')+'</tr>';
 const el=document.getElementById('raw');el.innerHTML=t;el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const i=+th.dataset.i;sortD=(sortK===i)?-sortD:1;sortK=i;raw()})}
raw();
const ST={real:true,sim:true,link:true,p:new Set(PROGS),b:new Set(BINS)};
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
group('c-prog',PROGS.map(p=>({label:p,dot:pcol(p),on:()=>ST.p.has(p),toggle:()=>ST.p.has(p)?ST.p.delete(p):ST.p.add(p),only:()=>{ST.p=new Set([p])}})),()=>{ST.p=new Set(PROGS)},()=>{ST.p=new Set()});
group('c-bin',BINS.map(b=>({label:b,on:()=>ST.b.has(b),toggle:()=>ST.b.has(b)?ST.b.delete(b):ST.b.add(b),only:()=>{ST.b=new Set([b])}})),()=>{ST.b=new Set(BINS)},()=>{ST.b=new Set()});
function sync(){btns.forEach(b=>b._sync());draw()}
const NS='http://www.w3.org/2000/svg',lg=Math.log10;
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
const tick=v=>v>=1?v.toLocaleString('en-US'):String(+v.toPrecision(1));
function hover(chart,node,text){const tip=chart.querySelector('.tip');node.addEventListener('mouseenter',()=>{tip.textContent=text;tip.style.display='block'});
 node.addEventListener('mousemove',e=>{const b=chart.getBoundingClientRect();let x=e.clientX-b.left+12;if(x+tip.offsetWidth>b.width)x=e.clientX-b.left-tip.offsetWidth-12;tip.style.left=Math.max(0,x)+'px';tip.style.top=(e.clientY-b.top+12)+'px'});
 node.addEventListener('mouseleave',()=>{tip.style.display='none'})}
function tri(x,y,s){return `${x},${y-s} ${x-s*.9},${y+s*.6} ${x+s*.9},${y+s*.6}`}
function plot(o){const svg=document.getElementById(o.id),chart=svg.parentNode,[,,W,H]=svg.getAttribute('viewBox').split(' ').map(Number),ML=58,MR=14,MT=12,MB=46;
 svg.replaceChildren();const sx=x=>ML+(lg(x)-o.x0)/(o.x1-o.x0)*(W-ML-MR),sy=y=>H-MB-(lg(y)-o.y0)/(o.y1-o.y0)*(H-MT-MB);
 const cp=el('clipPath',{id:o.id+'-clip'},el('defs',{},svg));el('rect',{x:ML,y:MT,width:W-ML-MR,height:H-MT-MB},cp);
 const ax={style:'stroke:var(--line)','stroke-width':1},txt={style:'fill:var(--muted)','font-size':11};
 for(let e=Math.ceil(o.x0);e<=o.x1;e++){const x=sx(10**e);el('line',{x1:x,x2:x,y1:MT,y2:H-MB,...ax},svg);el('text',{x,y:H-MB+16,'text-anchor':'middle',...txt},svg).textContent=tick(10**e)}
 for(let e=Math.ceil(o.y0);e<=o.y1;e++){const y=sy(10**e);el('line',{x1:ML,x2:W-MR,y1:y,y2:y,...ax},svg);el('text',{x:ML-6,y:y+4,'text-anchor':'end',...txt},svg).textContent=tick(10**e)}
 el('rect',{x:ML,y:MT,width:W-ML-MR,height:H-MT-MB,fill:'none',...ax},svg);
 el('text',{x:(ML+W-MR)/2,y:H-8,'text-anchor':'middle',...txt,'font-size':12},svg).textContent=o.xl;
 el('text',{x:14,y:(MT+H-MB)/2,'text-anchor':'middle',transform:`rotate(-90 14 ${(MT+H-MB)/2})`,...txt,'font-size':12},svg).textContent=o.yl;
 const g=el('g',{'clip-path':`url(#${o.id}-clip)`},svg);
 el('polyline',{points:o.roof.map(([x,y])=>`${sx(x)},${sy(y)}`).join(' '),fill:'none',style:'stroke:var(--roof)','stroke-width':1.5},g);
 const vis=R.filter(r=>ST.p.has(r.program)&&ST.b.has(r.bin)),X=r=>o.X(r)*o.jit(r);
 const say=(k,r,v)=>`${k} ${r.program} ${r.R}×${r.C}: ${v} · measured ${r.real_us.toFixed(3)} µs · TOGSim ${r.sim_us===null?'failed':r.sim_us.toFixed(3)+' µs'} · ratio ${r.ratio===null?'–':r.ratio.toFixed(2)}`;
 if(ST.link&&ST.real&&ST.sim)for(const r of vis)if(o.Ys(r)!==null)el('line',{x1:sx(X(r)),y1:sy(o.Yr(r)),x2:sx(X(r)),y2:sy(o.Ys(r)),style:`stroke:${pcol(r.program)}`,'stroke-width':1,'stroke-dasharray':'3 3'},g);
 if(ST.real)for(const r of vis)hover(chart,el('circle',{cx:sx(X(r)),cy:sy(o.Yr(r)),r:4.5,style:`fill:${pcol(r.program)};stroke:${pcol(r.program)}`},g),say('measured',r,o.fmt(o.Yr(r),r)));
 if(ST.sim)for(const r of vis)if(o.Ys(r)!==null)hover(chart,el('polygon',{points:tri(sx(X(r)),sy(o.Ys(r)),5.5),style:`fill:none;stroke:${pcol(r.program)}`,'stroke-width':1.6},g),say('TOGSim',r,o.fmt(o.Ys(r),r)));
 const L=[`<span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" style="stroke:var(--roof)" stroke-width="2"/></svg> ${o.roofName}</span>`];
 if(ST.real)L.push('<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" style="fill:var(--muted)"/></svg> measured TPU v6e (●, color = program)</span>');
 if(ST.sim)L.push('<span><svg width="14" height="12"><polygon points="7,1 1,10 13,10" style="fill:none;stroke:var(--muted)" stroke-width="1.6"/></svg> TOGSim (△)</span>');
 if(ST.link&&ST.real&&ST.sim)L.push('<span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" style="stroke:var(--muted)" stroke-dasharray="3 3"/></svg> same-shape link</span>');
 document.getElementById(o.leg).innerHTML=L.join('')}
const span=(vs,lo,hi)=>{const v=vs.filter(x=>x!==null&&x>0);return [Math.floor(lg(Math.min(...v)/lo)),Math.ceil(lg(Math.max(...v)*hi))]};
const JIT={softmax:0,max:-0.03,sum:0.03,add1:-0.03,slice:0.03};
function draw(){const knee=D.valu/D.bw,[x0,x1]=span([...R.map(r=>r.ai),knee],2,3),[y0,y1]=span([...R.map(r=>r.real_gf),...R.map(r=>r.sim_gf),D.valu],2,1.5);
 plot({id:'roof',leg:'leg-roof',x0,x1,y0,y1,xl:'arithmetic intensity (FLOP/byte)',yl:'GFLOP/s',roofName:`roof (HBM ${Math.round(D.bw).toLocaleString('en-US')} GB/s · VALU ${(D.valu/1e3).toFixed(2)} TOP/s)`,
  roof:[[10**x0,10**x0*D.bw],[knee,D.valu],[10**x1,D.valu]],X:r=>r.ai,jit:r=>10**JIT[r.program],Yr:r=>r.real_gf,Ys:r=>r.sim_gf,
  fmt:(v,r)=>`${v.toFixed(2)} GFLOP/s (${(100*v/Math.min(D.valu,r.ai*D.bw)).toFixed(1)}% of roof)`});
 const[b0,b1]=span(R.map(r=>r.bytes),1,1),[g0,g1]=span([...R.map(r=>r.real_gbps),...R.map(r=>r.sim_gbps),D.bw],2,1.5);
 plot({id:'bw',leg:'leg-bw',x0:b0,x1:b1,y0:g0,y1:g1,xl:'bytes moved (archive bytes)',yl:'GB/s',roofName:`HBM ${Math.round(D.bw).toLocaleString('en-US')} GB/s`,
  roof:[[10**b0,D.bw],[10**b1,D.bw]],X:r=>r.bytes,jit:()=>1,Yr:r=>r.real_gbps,Ys:r=>r.sim_gbps,fmt:v=>`${v.toFixed(1)} GB/s (${(100*v/D.bw).toFixed(1)}% of HBM)`})}
draw();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
