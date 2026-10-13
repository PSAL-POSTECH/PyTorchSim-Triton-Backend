#!/usr/bin/env python3
"""results.json + shapes.csv + the real TPU v6e rows -> report.html (self-contained: inline SVG
and vanilla JS, nothing fetched), summary.md for the job summary, and results.csv. Raw simulated
time against raw measured time; no constant is fitted, and a failed kernel is shown, not dropped."""
import argparse
import csv
import datetime
import html
import json
import math
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WORST = 8


def join_rows(shapes_csv, real_csv, results):
    """One row per shape: the kernel that stands for it, its measured and its simulated time."""
    with open(real_csv, newline="") as fh:
        real = {(int(r["M"]), int(r["K"]), int(r["N"])): r for r in csv.DictReader(fh)}
    sim = {r["kernel"]: r for r in results["shapes"]}
    rows = []
    with open(shapes_csv, newline="") as fh:
        for s in csv.DictReader(fh):
            key = (int(s["M"]), int(s["K"]), int(s["N"]))
            if key not in real:
                raise SystemExit(f"{real_csv} has no measurement for {key}")
            t, got = real[key], sim.get(s["kernel"], {})
            flops, nbytes = int(t["flops"]), int(t["bytes"])
            real_us = int(t["latency_ns"]) / 1000
            sim_us = got.get("us") if got.get("rc") == 0 else None
            rows.append({
                "M": key[0], "K": key[1], "N": key[2], "kernel": s["kernel"], "form": s["form"],
                "label": s["label"], "mark": s["mark"], "flops": flops, "bytes": nbytes,
                "ai": flops / nbytes, "real_us": real_us, "real_tf": flops / real_us / 1e6,
                "sim_us": sim_us, "sim_tf": None if sim_us is None else flops / sim_us / 1e6,
                "ratio": None if sim_us is None else sim_us / real_us,
                "total_cycles": got.get("total_cycles"),
                "rc": got.get("rc", "not run"), "failed_step": got.get("failed_step"),
                "log": got.get("log")})
    return rows


def count_chips(rows):
    """The headline counts: shapes within ±10% / ±20%, overall and for GEMM (M > 1), the GEMM
    geometric-mean ratio, and the failures; each a (value, label, alert) triple."""
    def within(rs, tol):
        return sum(1 for r in rs if r["ratio"] is not None and abs(r["ratio"] - 1) <= tol)
    gemm = [r for r in rows if r["M"] > 1]
    done = [r for r in gemm if r["ratio"] is not None]
    m_min = min((r["M"] for r in gemm), default=2)
    chips = [(f"{within(rows, .10)} / {len(rows)}", "전체, ±10% 이내", False),
             (f"{within(rows, .20)} / {len(rows)}", "전체, ±20% 이내", False),
             (f"{within(gemm, .10)} / {len(gemm)}", f"GEMM(M≥{m_min}), ±10%", False),
             (f"{within(gemm, .20)} / {len(gemm)}", f"GEMM(M≥{m_min}), ±20%", False),
             ("–" if not done else f"{math.exp(sum(math.log(r['ratio']) for r in done) / len(done)):.3f}",
              "GEMM 비율 기하평균", False)]
    failed = sum(1 for r in rows if r["ratio"] is None)
    if failed:
        chips.append((f"{failed} / {len(rows)}", "시뮬레이션 실패", True))
    return chips


def emit_summary(rows, chips, meta):
    """Markdown for $GITHUB_STEP_SUMMARY: the chips, the failures, the worst outliers."""
    out = [f"### GEMM validation: TOGSim vs TPU v6e ({len(rows)} shapes) @ `{meta['commit'][:12]}`", "",
           "| " + " | ".join(label for _, label, _ in chips) + " |",
           "|" + "---|" * len(chips),
           "| " + " | ".join(f"**{value}**" for value, _, _ in chips) + " |", ""]
    failed = [r for r in rows if r["ratio"] is None]
    if failed:
        out += [f"**Failed ({len(failed)})** — shown in the report, logs in the artifact:", ""]
        out += [f"- `{r['M']}x{r['K']}x{r['N']}` `{r['kernel']}`: "
                + (f"{r['failed_step']} step, rc {r['rc']}, `{r['log']}`" if r["failed_step"] else "not run")
                for r in failed]
        out.append("")
    worst = sorted((r for r in rows if r["ratio"] is not None),
                   key=lambda r: -abs(math.log(r["ratio"])))[:WORST]
    if worst:
        out += [f"Largest deviations (TOGSim ÷ measured, {len(worst)} of {len(rows)}):", "",
                "| M×K×N | kernel form | measured µs | TOGSim µs | ratio |", "|---|---|---:|---:|---:|"]
        out += [f"| {r['M']}×{r['K']}×{r['N']} | {r['form']}{' ' + r['mark'] if r['mark'] else ''} "
                f"| {r['real_us']:.3f} | {r['sim_us']:.3f} | {r['ratio']:.3f} |" for r in worst]
        out.append("")
    if meta.get("artifact"):
        out.append(f"Full report: `report.html` in artifact `{meta['artifact']}`.")
    return "\n".join(out) + "\n"


def emit_csv(path, rows):
    """The joined table, one line per shape, for anyone who wants the numbers without the page."""
    cols = ["M", "K", "N", "kernel", "form", "mark", "flops", "bytes", "ai", "real_us", "sim_us",
            "ratio", "real_tf", "sim_tf", "total_cycles", "rc", "failed_step", "log"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def emit_html(rows, chips, meta, device, machine):
    """The page: header, chips, failures, ratio heatmap, roofline, raw table, notes."""
    peak, bw = float(device["peak_bf16_tflops"]), float(device["hbm_gbps"])
    sha = html.escape(meta["commit"][:7])
    commit = (f'<a href="{html.escape(meta["commit_url"])}">{sha}</a>' if meta.get("commit_url") else sha)
    run = f' · <a href="{html.escape(meta["run_url"])}">CI 실행</a>' if meta.get("run_url") else ""
    ref = f' ({html.escape(meta["ref"])})' if meta.get("ref") else ""
    machine = machine or {}
    arrays = machine.get("num_systolic_array_per_core")
    togsim = html.escape(os.path.basename(machine["togsim_config"])) if machine else "실행 안 됨"
    real = meta["real"]
    head = (f'{meta["date"]} · PTB {commit}{ref}{run} · TOGSim {togsim}'
            f'{f" · 배열 {arrays}개" if arrays else ""} · 실측 {html.escape(real["framework"])} '
            f'{html.escape(real["device"])} ({html.escape(real["run_id"])}, tpu-perf-archive '
            f'{html.escape(real["archive_commit"])})')
    data = {"rows": rows, "chips": chips, "peak": peak, "bw": bw}
    page = (PAGE.replace("@@N@@", str(len(rows)))
            .replace("@@HEAD@@", head)
            .replace("@@DEVICE@@", html.escape(str(device["name"])))
            .replace("@@PEAK@@", f"{peak:,.0f}")
            .replace("@@BW@@", f"{bw:,.0f}")
            .replace("@@KNEE@@", f"{peak * 1000 / bw:,.0f}")
            .replace("@@DATA@@", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    return page


def main():
    """Command line: where results.json is, where the page goes, and what this run was."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="report.html, summary.md and results.csv go here")
    ap.add_argument("--shapes", default=os.path.join(HERE, "shapes.csv"))
    ap.add_argument("--real", default=os.path.join(HERE, "tpu_v6e_torchxla.csv"))
    ap.add_argument("--device", default=os.path.join(HERE, "tpu_v6e.yaml"))
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
    with open(args.real, newline="") as fh:
        first = next(csv.DictReader(fh))
    rows = join_rows(args.shapes, args.real, results)
    chips = count_chips(rows)
    meta = {"commit": args.commit, "commit_url": args.commit_url, "ref": args.ref,
            "run_url": args.run_url, "artifact": args.artifact, "real": first,
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.html"), "w") as fh:
        fh.write(emit_html(rows, chips, meta, device, results["machine"]))
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(emit_summary(rows, chips, meta))
    emit_csv(os.path.join(args.out, "results.csv"), rows)
    print(f"{len(rows)} shapes, {sum(r['ratio'] is None for r in rows)} without a simulated time -> "
          f"{os.path.join(args.out, 'report.html')}")


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TOGSim 대 TPU @@DEVICE@@ 정확도</title>
<style>
:root{--bg:#f4f5f3;--panel:#fff;--fg:#1c2321;--muted:#5d6864;--line:#d6dcd9;--accent:#2f6b5e;
--roof:#1c2321;--ok:#e3efe9;--mid:#f5ecd6;--bad:#f3dcd6;--fast:#dfe7f3;--fail:#e2e2e2;
--k0:#2f6b5e;--k1:#b5562a;--k2:#4d5fa8;--k3:#8a6a12;--k4:#9b3f78;--k5:#3f8a9a;--k6:#5c5c5c;
--sans:system-ui,-apple-system,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;
--roof:#e3e9e6;--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--k0:#74c2ad;--k1:#e59a6c;--k2:#93a5ef;--k3:#d9b551;--k4:#e08cc4;--k5:#7fcfdc;--k6:#b8b8b8;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#131917;--panel:#1b2220;--fg:#e3e9e6;--muted:#9aa5a1;--line:#2f3936;--accent:#74c2ad;
--roof:#e3e9e6;--ok:#1f3a31;--mid:#3a3420;--bad:#46271f;--fast:#1f2b3d;--fail:#30302e;--k0:#74c2ad;--k1:#e59a6c;--k2:#93a5ef;--k3:#d9b551;--k4:#e08cc4;--k5:#7fcfdc;--k6:#b8b8b8;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.6;margin:0}
a{color:var(--accent)}
.wrap{max-width:1180px;margin:0 auto;padding-inline:16px;padding-block:28px 56px;display:grid;gap:22px}
h1{font-size:24px;margin:0;text-wrap:balance}h2{font-size:17px;margin:0 0 10px;text-wrap:balance}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted)}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.chips{display:flex;flex-wrap:wrap;gap:10px}
.chip{border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--panel)}
.chip b{font-family:var(--mono);font-size:20px;display:block}
.chip span{font-size:12.5px;color:var(--muted)}
.chip.alert{border-color:var(--k1)}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:16px}
@media (max-width:900px){.grid2{grid-template-columns:minmax(0,1fr)}}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13.5px;width:100%}
th,td{border:1px solid var(--line);padding:5px 7px;text-align:center;white-space:nowrap}
th{font-weight:500;color:var(--muted);font-size:12.5px;background:var(--panel)}
td.r{font-family:var(--mono)}
#raw td{text-align:right}#raw td.l{text-align:left;font-family:var(--sans);white-space:normal;min-width:16ch}#raw th{cursor:pointer;user-select:none}
td.ok{background:var(--ok)}td.mid{background:var(--mid)}td.bad{background:var(--bad)}td.fast{background:var(--fast)}td.fail{background:var(--fail)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12.5px;color:var(--muted);margin-top:8px;align-items:center}
.sw{display:inline-block;width:12px;height:12px;border:1px solid var(--line);vertical-align:-2px;margin-right:4px}
.chart{position:relative}
.chart svg{display:block;width:100%;height:auto;font-family:var(--sans)}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;max-width:300px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none}
.ctrl{display:grid;gap:6px;margin-bottom:8px}
.crow{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.clab{font-size:12px;color:var(--muted);width:3.2em}
.tg{font:inherit;font-size:12.5px;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:999px;padding:2px 10px;cursor:pointer}
.tg[aria-pressed="false"]{opacity:.45;text-decoration:line-through}
.tg:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.tg .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:0}
.tg.small{padding:2px 8px;color:var(--muted)}
p,li{max-width:76ch}ul{margin:6px 0;padding-left:20px}
.tag{font-size:10px;color:var(--muted);margin-left:2px}
</style></head><body>
<div class="wrap">
<header style="display:grid;gap:6px">
<div class="meta">@@HEAD@@</div>
<h1>TOGSim 대 TPU @@DEVICE@@: GEMM·GEMV @@N@@개 크기</h1>
</header>
<section class="chips" id="chips"></section>
<section class="box" id="failbox" hidden><h2>시뮬레이션 실패</h2>
<p class="meta">아래 커널은 컴파일 또는 시뮬레이션이 0이 아닌 코드로 끝났습니다. 로그는 artifact의 logs/ 아래에 있습니다.</p>
<div class="scroll"><table id="fails"></table></div></section>
<section class="grid2">
<div class="box"><h2>오차 표 (TOGSim 시간 ÷ 실측 시간)</h2>
<div class="scroll"><table id="heat"></table></div>
<div class="legend"><span><i class="sw" style="background:var(--ok)"></i>±10% 이내</span><span><i class="sw" style="background:var(--mid)"></i>±10~20%</span><span><i class="sw" style="background:var(--bad)"></i>20% 넘게 느림</span><span><i class="sw" style="background:var(--fast)"></i>20% 넘게 빠름</span><span><i class="sw" style="background:var(--fail)"></i>실패</span><span>T: 조각 방식 커널로 대신함 · Kp: K 조각 형태</span></div>
<p class="meta" style="margin-top:8px">행은 M, 열은 K×N입니다. M=1은 GEMV입니다. 1보다 크면 TOGSim이 느립니다.</p>
</div>
<div class="box"><h2>Roofline (@@DEVICE@@: @@PEAK@@ TFLOP/s, @@BW@@ GB/s)</h2>
<div class="ctrl">
 <div class="crow"><span class="clab">계열</span><span id="c-series"></span></div>
 <div class="crow"><span class="clab">K×N</span><span id="c-kn"></span></div>
 <div class="crow"><span class="clab">M</span><span id="c-m"></span></div>
</div>
<div class="chart" id="chart"><svg id="roof" viewBox="0 0 640 440" role="img" aria-label="roofline"></svg><div class="tip" id="tip"></div></div>
<div class="legend" id="leg"></div>
<p class="meta">가로축은 연산 강도(FLOP ÷ 최소 메모리 바이트, bf16 입출력), 세로축은 달성 TFLOP/s입니다. 색은 K×N, 실측은 채운 원(●), TOGSim은 빈 삼각형(△)입니다. 같은 크기의 두 점은 선으로 잇습니다. 위 버튼으로 계열과 workload를 골라 보세요(버튼을 두 번 연달아 누르면 그것만 보입니다). 꺾인 선이 지붕입니다(꺾이는 지점 @@KNEE@@ FLOP/B).</p>
</div>
</section>
<section class="box"><h2>전체 값</h2>
<p class="meta">열 제목을 누르면 정렬됩니다.</p>
<div class="scroll"><table id="raw"></table></div>
</section>
<section class="box"><h2>읽을 때 주의할 점</h2><ul>
<li>TOGSim 시간은 Spike 없이 잰 값입니다. 커널을 binary까지 컴파일한 뒤 gem5 타일 표본과 TOGSim으로 시간만 재고, 결과 값은 검사하지 않습니다. 맞춘 상수는 없습니다: 비율은 시뮬레이션 시간을 실측 지연(latency_ns)으로 나눈 그대로입니다.</li>
<li>K가 2의 거듭제곱이 아닌 칸(384, 768, 3072)은 K를 한 번에 처리하는 커널로 쟀습니다. 이 형태는 0으로 채운 행까지 DRAM에서 읽어 일부 칸이 실제보다 1.0에 가깝게 보일 수 있습니다.</li>
<li>384×1152의 M≥2048은 실물 XLA의 128폭 출력 창을 그대로 따라 계산 장치를 절반만 씁니다.</li>
<li>T 표시 칸은 입력을 scratchpad에 다 올릴 수 없어 조각 방식 커널로 대신했습니다.</li>
<li>작은 M(32~256)에서 실물의 계산 구간은 크기와 관계없이 약 1.3 µs 아래로 내려가지 않습니다. 원인은 아직 확인하지 않았습니다.</li>
<li>GEMV(M=1)는 validation/gemm/kernels/xs_*.py 커널입니다.</li>
</ul></section>
</div>
<script>
const D=@@DATA@@;
const R=D.rows;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=q=>q===null?'fail':Math.abs(q-1)<=.1?'ok':Math.abs(q-1)<=.2?'mid':q>1?'bad':'fast';
const fx=(v,d)=>v===null?'–':v.toFixed(d);
document.getElementById('chips').innerHTML=D.chips.map(([b,s,a])=>`<div class="chip${a?' alert':''}"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join('');
const failed=R.filter(r=>r.ratio===null);
if(failed.length){document.getElementById('failbox').hidden=false;
 document.getElementById('fails').innerHTML='<tr><th>M×K×N</th><th>커널</th><th>단계</th><th>rc</th><th>로그</th></tr>'+failed.map(r=>
 `<tr><td>${r.M}×${r.K}×${r.N}</td><td>${esc(r.kernel)}</td><td>${esc(r.failed_step||'실행 안 됨')}</td><td>${esc(r.rc)}</td><td>${r.log?`<a href="${esc(r.log)}">${esc(r.log)}</a>`:'–'}</td></tr>`).join('')}
const KN=[...new Map(R.map(r=>[r.K+'x'+r.N,[r.K,r.N]])).values()].sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
const Ms=[...new Set(R.map(r=>r.M))].sort((a,b)=>a-b);
const knKey=r=>r.K+'x'+r.N, knIdx=k=>KN.findIndex(([a,b])=>a+'x'+b===k), kcol=r=>`var(--k${knIdx(knKey(r))%7})`;
let h='<tr><th>M \\ K×N</th>'+KN.map(([k,n])=>`<th>${k}×${n}</th>`).join('')+'</tr>';
for(const m of Ms){h+=`<tr><th>${m}</th>`;for(const [k,n] of KN){const r=R.find(x=>x.M==m&&x.K==k&&x.N==n);
 if(!r){h+='<td>–</td>';continue}const q=r.ratio,t=r.mark?`<span class="tag">${esc(r.mark)}</span>`:'';
 const tip=q===null?`실패: ${r.failed_step||'실행 안 됨'} (rc ${r.rc}) · ${r.kernel}`:`실측 ${r.real_us.toFixed(2)} µs · TOGSim ${r.sim_us.toFixed(2)} µs · ${r.kernel}`;
 h+=`<td class="r ${cls(q)}" title="${esc(tip)}">${q===null?'실패':q.toFixed(2)}${t}</td>`}h+='</tr>'}
document.getElementById('heat').innerHTML=h;
const cols=[['M',r=>r.M,0],['K',r=>r.K,0],['N',r=>r.N,0],['커널 형태',r=>r.label+(r.mark?` (${r.mark})`:''),'s'],['연산 강도',r=>r.ai,1],['실측 µs',r=>r.real_us,2],['TOGSim µs',r=>r.sim_us,2],['비율',r=>r.ratio,3],['실측 TFLOP/s',r=>r.real_tf,1],['TOGSim TFLOP/s',r=>r.sim_tf,1]];
let sortK=null,sortD=1;
function raw(){const a=[...R];if(sortK!==null)a.sort((p,q)=>{const f=cols[sortK][1],u=f(p),v=f(q);if(u===null||v===null)return (u===null)-(v===null);return (u>v?1:u<v?-1:0)*sortD});
 let t='<tr>'+cols.map((c,i)=>`<th data-i="${i}">${c[0]}${sortK===i?(sortD>0?' ▲':' ▼'):''}</th>`).join('')+'</tr>';
 for(const r of a){t+='<tr>'+cols.map(([n,f,d])=>{const v=f(r);if(d==='s')return `<td class="l" title="${esc(r.kernel)}">${esc(v)}</td>`;
  if(n==='비율')return `<td class="${cls(v)}">${v===null?'실패':v.toFixed(d)}</td>`;
  if(n==='TOGSim µs'&&v===null)return `<td class="fail">${r.log?`<a href="${esc(r.log)}">실패</a>`:'실패'}</td>`;
  return `<td>${fx(v,d)}</td>`}).join('')+'</tr>'}
 const el=document.getElementById('raw');el.innerHTML=t;el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const i=+th.dataset.i;sortD=(sortK===i)?-sortD:1;sortK=i;raw()})}
raw();
const ST={real:true,sim:true,link:true,kn:new Set(KN.map(([k,n])=>k+'x'+n)),m:new Set(Ms)};
function mkBtn(label,on,toggle,only,dot){const b=document.createElement('button');b.className='tg';b.type='button';b.setAttribute('aria-pressed',on());
 b.innerHTML=(dot?`<i class="dot" style="background:${dot}"></i>`:'')+esc(label);
 let t=null;b.addEventListener('click',()=>{if(t){clearTimeout(t);t=null;only();sync();return}t=setTimeout(()=>{t=null;toggle();sync()},230)});
 b._sync=()=>b.setAttribute('aria-pressed',on());return b}
const btns=[];
function addGroup(id,items,all,none){const el=document.getElementById(id);
 for(const it of items){const b=mkBtn(it.label,it.on,it.toggle,it.only,it.dot);btns.push(b);el.appendChild(b)}
 for(const [txt,fn] of [['전체',all],['해제',none]]){const b=document.createElement('button');b.className='tg small';b.type='button';b.textContent=txt;b.onclick=()=>{fn();sync()};el.appendChild(b)}}
addGroup('c-series',[
 {label:'실측 TPU',on:()=>ST.real,toggle:()=>ST.real=!ST.real,only:()=>{ST.real=true;ST.sim=false}},
 {label:'TOGSim',on:()=>ST.sim,toggle:()=>ST.sim=!ST.sim,only:()=>{ST.sim=true;ST.real=false}},
 {label:'잇는 선',on:()=>ST.link,toggle:()=>ST.link=!ST.link,only:()=>{ST.link=!ST.link}}],()=>{ST.real=ST.sim=ST.link=true},()=>{ST.real=ST.sim=ST.link=false});
addGroup('c-kn',KN.map(([k,n],i)=>{const key=k+'x'+n;return {label:k+'×'+n,dot:`var(--k${i%7})`,on:()=>ST.kn.has(key),
 toggle:()=>ST.kn.has(key)?ST.kn.delete(key):ST.kn.add(key),only:()=>{ST.kn=new Set([key])}}}),()=>{ST.kn=new Set(KN.map(([k,n])=>k+'x'+n))},()=>{ST.kn=new Set()});
addGroup('c-m',Ms.map(m=>({label:m===1?'1 (GEMV)':String(m),on:()=>ST.m.has(m),toggle:()=>ST.m.has(m)?ST.m.delete(m):ST.m.add(m),only:()=>{ST.m=new Set([m])}})),()=>{ST.m=new Set(Ms)},()=>{ST.m=new Set()});
function sync(){btns.forEach(b=>b._sync());draw()}
const NS='http://www.w3.org/2000/svg',W=640,H=440,ML=58,MR=14,MT=12,MB=46;
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
const lg=Math.log10,vals=[...R.map(r=>r.real_tf),...R.map(r=>r.sim_tf)].filter(v=>v!==null);
const X0=lg(Math.min(...R.map(r=>r.ai))/2),X1=lg(Math.max(...R.map(r=>r.ai),D.peak*1000/D.bw)*3);
const Y0=lg(Math.min(...vals)/2),Y1=lg(D.peak*1.3);
const sx=x=>ML+(lg(x)-X0)/(X1-X0)*(W-ML-MR),sy=y=>H-MB-(lg(y)-Y0)/(Y1-Y0)*(H-MT-MB);
const tick=v=>v>=1?v.toLocaleString('en-US'):String(v);
const tip=document.getElementById('tip'),chart=document.getElementById('chart');
function hover(node,text){node.addEventListener('mouseenter',e=>{tip.textContent=text;tip.style.display='block'});
 node.addEventListener('mousemove',e=>{const b=chart.getBoundingClientRect();let x=e.clientX-b.left+12;if(x+tip.offsetWidth>b.width)x=e.clientX-b.left-tip.offsetWidth-12;tip.style.left=Math.max(0,x)+'px';tip.style.top=(e.clientY-b.top+12)+'px'});
 node.addEventListener('mouseleave',()=>{tip.style.display='none'})}
function tri(x,y,s){return `${x},${y-s} ${x-s*.9},${y+s*.6} ${x+s*.9},${y+s*.6}`}
function draw(){const svg=document.getElementById('roof');svg.replaceChildren();
 const defs=el('defs',{},svg),cp=el('clipPath',{id:'plot'},defs);el('rect',{x:ML,y:MT,width:W-ML-MR,height:H-MT-MB},cp);
 const ax={style:'stroke:var(--line)','stroke-width':1},txt={style:'fill:var(--muted)','font-size':11};
 for(let e=Math.ceil(X0);e<=X1;e++){const x=sx(10**e);el('line',{x1:x,x2:x,y1:MT,y2:H-MB,...ax},svg);const t=el('text',{x,y:H-MB+16,'text-anchor':'middle',...txt},svg);t.textContent=tick(10**e)}
 for(let e=Math.ceil(Y0);e<=Y1;e++){const y=sy(10**e);el('line',{x1:ML,x2:W-MR,y1:y,y2:y,...ax},svg);const t=el('text',{x:ML-6,y:y+4,'text-anchor':'end',...txt},svg);t.textContent=tick(10**e)}
 el('rect',{x:ML,y:MT,width:W-ML-MR,height:H-MT-MB,fill:'none',...ax},svg);
 const xt=el('text',{x:(ML+W-MR)/2,y:H-8,'text-anchor':'middle',...txt,'font-size':12},svg);xt.textContent='연산 강도 (FLOP/byte)';
 const yt=el('text',{x:14,y:(MT+H-MB)/2,'text-anchor':'middle',transform:`rotate(-90 14 ${(MT+H-MB)/2})`,...txt,'font-size':12},svg);yt.textContent='TFLOP/s';
 const g=el('g',{'clip-path':'url(#plot)'},svg),knee=D.peak*1000/D.bw,xa=10**X0,xb=10**X1;
 el('polyline',{points:`${sx(xa)},${sy(xa*D.bw/1000)} ${sx(knee)},${sy(D.peak)} ${sx(xb)},${sy(D.peak)}`,fill:'none',style:'stroke:var(--roof)','stroke-width':1.5},g);
 const vis=R.filter(r=>ST.kn.has(knKey(r))&&ST.m.has(r.M));
 const say=(k,r,y)=>`${k} ${r.M}×${r.K}×${r.N}: ${y.toFixed(1)} TFLOP/s (실측 ${r.real_us.toFixed(2)} µs · TOGSim ${r.sim_us===null?'실패':r.sim_us.toFixed(2)+' µs'} · 비율 ${r.ratio===null?'–':r.ratio.toFixed(2)})`;
 if(ST.link&&ST.real&&ST.sim)for(const r of vis)if(r.sim_tf!==null)el('line',{x1:sx(r.ai),y1:sy(r.real_tf),x2:sx(r.ai),y2:sy(r.sim_tf),style:`stroke:${kcol(r)}`,'stroke-width':1,'stroke-dasharray':'3 3'},g);
 if(ST.real)for(const r of vis)hover(el('circle',{cx:sx(r.ai),cy:sy(r.real_tf),r:5,style:`fill:${kcol(r)};stroke:${kcol(r)}`},g),say('실측',r,r.real_tf));
 if(ST.sim)for(const r of vis)if(r.sim_tf!==null)hover(el('polygon',{points:tri(sx(r.ai),sy(r.sim_tf),6),style:`fill:none;stroke:${kcol(r)}`,'stroke-width':1.6},g),say('TOGSim',r,r.sim_tf));
 const L=['<span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" style="stroke:var(--roof)" stroke-width="2"/></svg> 지붕</span>'];
 if(ST.real)L.push('<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" style="fill:var(--muted)"/></svg> 실측 TPU (●, 색 = K×N)</span>');
 if(ST.sim)L.push('<span><svg width="14" height="12"><polygon points="7,1 1,10 13,10" style="fill:none;stroke:var(--muted)" stroke-width="1.6"/></svg> TOGSim (△, 색 = K×N)</span>');
 if(ST.link&&ST.real&&ST.sim)L.push('<span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" style="stroke:var(--muted)" stroke-dasharray="3 3"/></svg> 같은 크기 잇는 선</span>');
 document.getElementById('leg').innerHTML=L.join('')}
draw();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
