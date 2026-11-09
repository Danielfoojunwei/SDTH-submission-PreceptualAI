#!/usr/bin/env python3
"""Render a local, read-only explorer of the sealed spatial-control diagnostic.

Usage: python artifacts/spatial_control_review/render_results.py --browser
Only the four sibling presentation artifacts are written. No project scientific
module is imported, no model is executed, and no raw scientific array is decoded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import statistics
import threading


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MANIFEST = "artifacts/spatial_control/verification/manifest.json"
MANIFEST_SHA = "f652af5c0c1671a978ad0b2ebefcf35d7e5a7474dc36b373d63b4f60aa41d28f"
AUDIT = "artifacts/spatial_control_review/result.json"
AUDIT_SHA = "5f79df8dd60f4e45eb3ac300813a00d2c5a8fd8f724c4d380ce16fa2b2461df3"
REPORT = "artifacts/spatial_control/report.json"
REPORT_SHA = "5ba06b3d7a09b5cfc135ea3ea3ab6076982de99aacede1fbab9fa5c19e73cf29"
PROTOCOL = "artifacts/spatial_control/protocol.json"
ARMS = ["radiomap", "radiomap_recovery", "regression", "regression_support",
        "regression_recovery", "regression_both"]
SEEDS = [90511001, 90511002, 90511003, 90511004]
CONTRASTS = [("regression", "radiomap"), ("regression_support", "regression"),
             ("regression_recovery", "regression"), ("regression_both", "regression"),
             ("radiomap_recovery", "radiomap"), ("regression_both", "radiomap_recovery")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(name):
    target = (ROOT / name).resolve()
    require(target.is_relative_to(ROOT), "Source outside repository: " + name)
    return target


def sha(target):
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(name):
    return json.loads(path(name).read_text())


def verify_inputs():
    for name, digest in ((MANIFEST, MANIFEST_SHA), (AUDIT, AUDIT_SHA), (REPORT, REPORT_SHA)):
        require(sha(path(name)) == digest, "Pinned identity mismatch: " + name)
    manifest, audit, report = read(MANIFEST), read(AUDIT), read(REPORT)
    require(audit["passes"] is True and audit["source_evidence_unchanged"] is True,
            "Independent review did not pass with unchanged inputs")
    require(audit["study_manifest_sha256"] == MANIFEST_SHA, "Audit is for a different study")
    require(audit["SERVICE_opened"] is False and audit["production_promotion"] is False,
            "Unexpected qualification change")
    require(audit["scientific_calls"] == {"RT_solves": 0, "fits": 0, "forwards": 0,
                                          "model_constructors": 0, "packet_simulations": 0},
            "Review performed unexpected scientific calls")
    require(manifest["evidence_sha256"][REPORT] == REPORT_SHA, "Report is not sealed evidence")
    verified = {}
    for field in ("source_sha256", "evidence_sha256"):
        count, total = 0, 0
        for name, digest in manifest[field].items():
            target = path(name)
            require(sha(target) == digest, "Sealed byte mismatch: " + name)
            count += 1
            total += target.stat().st_size
        verified[field] = {"files": count, "bytes": total, "all_match": True}
    require(report["complete"] is True and report["fresh_simulations"] == 48
            and report["new_fits"] == 0 and report["default_changed"] is False
            and report["production_qualified"] is False, "Unexpected report scope")
    require(report["arm_means"] == audit["arm_means"]
            and report["contrasts"] == audit["contrasts"], "Report / independent review mismatch")
    protocol = read(PROTOCOL)
    require(protocol["arms"] == ARMS and protocol["seeds"] == SEEDS
            and protocol["robots"] == 100 and protocol["duration_seconds"] == 20,
            "Unexpected episode design")
    expected = {(h, arm, seed) for h in (.03, 4.) for arm in ARMS for seed in SEEDS}
    actual = {(r["handover_seconds"], r["arm"], r["seed"]) for r in report["rows"]}
    require(len(report["rows"]) == 48 and actual == expected, "Incomplete / duplicated episode rows")
    for row in report["rows"]:
        require(all(isinstance(v, (int, float)) and math.isfinite(v)
                    for v in row["metrics"].values()), "Non-finite metric")
        require(row["metrics"]["control_sent"] == 40000, "Unexpected release denominator")
    require(len(report["arm_means"]) == 12 and len(report["contrasts"]) == 12,
            "Unexpected summary membership")
    for h in (.03, 4.):
        for arm in ARMS:
            rows = [r["metrics"] for r in report["rows"]
                    if r["handover_seconds"] == h and r["arm"] == arm]
            summaries = [r for r in report["arm_means"]
                         if r["handover_seconds"] == h and r["arm"] == arm]
            require(len(summaries) == 1 and summaries[0]["episodes"] == 4, "Invalid arm summary")
            for key, value in summaries[0]["mean"].items():
                require(math.isclose(statistics.mean(r[key] for r in rows), value,
                                     rel_tol=1e-11, abs_tol=1e-11), "Episode mean mismatch: " + key)
        for candidate, baseline in CONTRASTS:
            contrasts = [r for r in report["contrasts"] if r["handover_seconds"] == h
                         and r["candidate"] == candidate and r["baseline"] == baseline]
            require(len(contrasts) == 1, "Invalid contrast membership")
            pairs = []
            for seed in SEEDS:
                c = next(r["metrics"] for r in report["rows"] if r["handover_seconds"] == h
                         and r["arm"] == candidate and r["seed"] == seed)
                b = next(r["metrics"] for r in report["rows"] if r["handover_seconds"] == h
                         and r["arm"] == baseline and r["seed"] == seed)
                pairs.append({k: c[k] - b[k] for k in c})
            contrast = contrasts[0]
            for key, value in contrast["mean_delta"].items():
                require(math.isclose(statistics.mean(r[key] for r in pairs), value,
                                     rel_tol=1e-11, abs_tol=1e-11), "Contrast mean mismatch: " + key)
            require(contrast["episodes_with_more_misses"] == sum(p["control_misses"] > 0 for p in pairs)
                    and contrast["episodes_with_larger_max_gap"] == sum(p["maximum_silence_ms"] > 0 for p in pairs),
                    "Paired harm count mismatch")
    return report, protocol, verified


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Spatial control · sealed result explorer</title>
<style>
:root{--bg:#0b1220;--panel:#131e30;--line:#2b3a50;--ink:#edf3fc;--muted:#a6b6cd;--mint:#7bdfbe;--orange:#ffb27f;--red:#ff909e;--blue:#8db9ff;--violet:#c5a1ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 85% 0,#19324a 0,transparent 40%),var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1400px;margin:auto;padding:35px 38px 28px}h1{font-size:clamp(29px,4vw,44px);letter-spacing:-1.7px;line-height:1.12;margin:9px 0 13px}h2{font-size:18px;letter-spacing:-.3px;margin:0 0 6px}h3{font-size:14px;margin:0}.eyebrow{color:var(--mint);text-transform:uppercase;font-size:11px;letter-spacing:2px;font-weight:750}.muted,small{color:var(--muted)}.intro{max-width:940px;font-size:15px;margin:0 0 18px}.banner{border:1px solid #536b5f;background:#172b2a;padding:11px 15px;border-radius:10px;font-size:13px}.ribbons{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:17px 0 20px}.ribbon{border-left:3px solid var(--mint);padding:10px 15px;background:#152437;border-radius:0 10px 10px 0}.ribbon.negative{border-color:var(--red)}.ribbon b{display:block;font-size:15px;margin:3px 0}.ribbon p{margin:0;font-size:12px;color:var(--muted)}.ribbon .tag{font-size:10px;letter-spacing:1.3px;text-transform:uppercase;color:var(--mint)}.negative .tag{color:var(--red)}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:20px;margin:24px 0 16px}.switch{display:flex;gap:5px;padding:5px;background:#080e19;border:1px solid var(--line);border-radius:12px}.switch button{font:inherit;border:0;color:var(--muted);background:transparent;border-radius:8px;padding:10px 18px;cursor:pointer}.switch button[aria-pressed=true]{color:#081b1c;background:var(--mint);font-weight:750}.selected-note{margin:0 0 15px;font-size:13px;color:var(--muted)}.charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{min-width:0;border:1px solid var(--line);background:linear-gradient(150deg,#172337,#111b2b);border-radius:15px;padding:18px}.card .sub{font-size:11px;color:var(--muted);margin:0 0 12px;min-height:32px}.plot{width:100%;height:auto;display:block}.plot text{fill:var(--muted);font:10px system-ui,sans-serif}.plot .value{fill:var(--ink);font-weight:650;font-size:11px}.plot .label{fill:var(--ink);font-size:10px}.plot .axis{stroke:#38475e;stroke-width:1}.plot .grid{stroke:#2a3a4d;stroke-dasharray:2 4}.metric-note{font-size:11px;color:var(--muted);margin:10px 0 0}.section{margin-top:24px}.two{display:grid;grid-template-columns:1fr 1.12fr;gap:16px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px}th{text-align:left;color:var(--muted);font-weight:500;font-size:10px;line-height:1.35;padding:11px 8px;border-bottom:1px solid var(--line)}td{padding:10px 8px;border-bottom:1px solid #243247;white-space:nowrap}td:first-child,th:first-child{padding-left:0}tbody tr:last-child td{border:0}.armname{font-weight:600}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px}.bad{color:var(--red)}.good{color:var(--mint)}.neutral{color:var(--muted)}.sectionhead{display:flex;align-items:baseline;justify-content:space-between;gap:15px}.sectionhead p{font-size:11px;color:var(--muted);margin:0}details{margin-top:12px;border-top:1px solid var(--line);padding-top:11px}summary{cursor:pointer;color:var(--blue);font-size:12px}.details-body{padding-top:12px;color:var(--muted);font-size:12px}.details-body p{margin:0 0 9px}.arch{width:100%;height:auto;margin-top:12px}.arch text{font-family:system-ui,sans-serif;fill:var(--ink);font-size:12px}.arch .caption{font-size:10px;fill:var(--muted)}.arch .title{font-size:11px;font-weight:700;letter-spacing:1px;fill:var(--mint)}.architecture-note{font-size:12px;color:var(--muted);margin:7px 0 0}.pill{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:2px 6px;font-size:10px;color:var(--muted);margin-left:5px}.legend{display:flex;flex-wrap:wrap;gap:15px;font-size:11px;color:var(--muted);margin-top:10px}.fine{font-size:11px;color:var(--muted)}footer{font-size:10px;color:var(--muted);margin-top:22px;overflow-wrap:anywhere}a{color:var(--blue)}.qualifications{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;font-size:12px;color:var(--muted)}.qualifications b{color:var(--ink);display:block;margin-bottom:4px}@media(max-width:1000px){main{padding:24px 18px}.charts{grid-template-columns:1fr}.plot{max-height:330px}.two{grid-template-columns:1fr}.qualifications{grid-template-columns:1fr}.ribbons{grid-template-columns:1fr}.toolbar{align-items:flex-start;flex-direction:column}.arch{min-width:850px}.archwrap{overflow-x:auto}}@media(prefers-reduced-motion:no-preference){button{transition:background .15s}}
</style></head><body><main>
<div class="eyebrow">WirelessJEV · warehouse connectivity · verified simulation</div>
<h1>Spatial support. Local recovery.<br>Different failure modes.</h1>
<p class="intro">A frozen regression forecast is tested with two separate additions: delivered server-side spatial evidence, and an on-device blackout watchdog. All six arms are shown, including the cases where the additions hurt.</p>
<div class="banner"><b>RadioMap remains the default.</b> Four simulated episode seeds per handover condition · 100 robots · 20 seconds after shared warm history · no production qualification or JEPA / SOTA claim.</div>
<div class="ribbons"><div class="ribbon"><span class="tag">30 ms handover · RadioMap leads</span><b id="fast-headline"></b><p id="fast-detail"></p></div><div class="ribbon negative"><span class="tag">4 s handover · recovery is harmful</span><b id="slow-headline"></b><p id="slow-detail"></p></div></div>
<div class="toolbar"><div><h2>Compare the handover cost</h2><span class="muted">Equal weight per episode. Lower is better for all three charts.</span></div><div class="switch" role="group" aria-label="Handover duration"><button type="button" data-handover="0.03" aria-pressed="true">30 ms · 0.03 s</button><button type="button" data-handover="4" aria-pressed="false">4 seconds</button></div></div>
<p class="selected-note" id="selected-note" aria-live="polite"></p>
<div class="charts"><section class="card"><h2>Command loss</h2><p class="sub">Missed command releases / 40,000 releases per episode (%)</p><svg class="plot" id="loss-chart" viewBox="0 0 390 278" role="img" aria-label="Command loss for all six arms"></svg></section><section class="card"><h2>Worst-decile command loss</h2><p class="sub">Mean loss of the 10 worst clients out of 100, then mean of episodes (%)</p><svg class="plot" id="tail-chart" viewBox="0 0 390 278" role="img" aria-label="Worst decile loss for all six arms"></svg></section><section class="card"><h2>Mean episode maximum silence</h2><p class="sub">Find the longest client silence in each episode, then average four maxima (s)</p><svg class="plot" id="gap-chart" viewBox="0 0 390 278" role="img" aria-label="Mean episode maximum silence for all six arms"></svg></section></div>
<p class="metric-note">Axes start at zero and stay fixed when switching handover conditions. Maximum silence is not a percentile, a fleet average, or the largest gap over all four episodes. No confidence / significance claim is made.</p>
<div class="two section"><section class="card"><div class="sectionhead"><h2>Paired episode harms</h2><span>Candidate − baseline</span></div><p class="fine">Counts of episodes with strictly more misses or a larger maximum silence; denominator 4. A smaller mean does not imply every episode benefits.</p><div class="table-wrap"><table id="harms-table"><thead><tr><th>Candidate vs baseline</th><th>Δ loss<br>percentage points</th><th>More<br>misses</th><th>Larger<br>max silence</th></tr></thead><tbody></tbody></table></div></section><section class="card"><h2>Compute and sensing costs</h2><p class="fine">Mean episode totals across 100 clients. Timings are elapsed wall time on CPU, not process CPU usage. Scan interruption is summed client-radio time, not episode duration.</p><div class="table-wrap"><table id="costs-table"><thead><tr><th>Arm</th><th>Forecast<br>elapsed (s)</th><th>Paid compute<br>allowance (s)</th><th>Scans<br>started</th><th>Scan interruption<br>client-seconds</th></tr></thead><tbody></tbody></table></div><details><summary>Show local costs, handovers and scan accounting</summary><div class="details-body"><p>Forecast elapsed totals include support preparation where enabled; do not add support preparation again. The paid allowance is the charged forecast schedule, not measured runtime. Planner and local elapsed times below are separate.</p><div class="table-wrap"><table id="local-table"><thead><tr><th>Arm</th><th>Planner<br>elapsed (s)</th><th>Local decision /<br>validation (s)</th><th>Support prep<br>elapsed (s)</th><th>Autonomous<br>scans</th><th>Central / local<br>handovers</th></tr></thead><tbody></tbody></table></div><p>A scan costs 90 ms of radio interruption. Autonomous and background scans share a five-second per-client start-spacing budget. Every enabled local decision has a charged 10 ms delay; actual compute overruns are also charged. Recovery can change acquired scans and telemetry, so this is not an equal-evidence comparison.</p></div></details></section></div>
<section class="card section"><div class="sectionhead"><h2>Two evidence paths, two control authorities</h2><span class="pill">Architecture tested here</span></div><div class="archwrap"><svg class="arch" viewBox="0 0 1260 265" role="img" aria-label="Delivered server support and own-only local recovery are separate evidence and control paths"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#8db9ff"/></marker><marker id="mintarrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#7bdfbe"/></marker></defs>
<rect x="1" y="12" width="1258" height="108" rx="12" fill="#102e3a" stroke="#375b68"/><text x="18" y="34" class="title">SERVER PATH · DELIVERED PAST REPORTS ONLY</text>
<rect x="20" y="49" width="187" height="51" rx="8" fill="#203b4c"/><text x="33" y="70">Delivered report history</text><text x="33" y="87" class="caption">Own + neighboring AP evidence</text>
<path d="M207 75H235" stroke="#8db9ff" stroke-width="2" marker-end="url(#arrow)"/><rect x="237" y="49" width="194" height="51" rx="8" fill="#203b4c"/><text x="250" y="70">Forecast + spatial guard</text><text x="250" y="87" class="caption">RadioMap or frozen regression</text>
<path d="M431 75H459" stroke="#8db9ff" stroke-width="2" marker-end="url(#arrow)"/><rect x="461" y="49" width="207" height="51" rx="8" fill="#203b4c"/><text x="474" y="70">Planner → command creation</text><text x="474" y="87" class="caption">100 ms allowance · late output discarded</text>
<path d="M668 75H696" stroke="#8db9ff" stroke-width="2" marker-end="url(#arrow)"/><rect x="698" y="49" width="215" height="51" rx="8" fill="#203b4c"/><text x="711" y="70">Subsequent command packet</text><text x="711" y="87" class="caption">Must be delivered · expires after 1 s</text>
<path d="M913 75H941" stroke="#8db9ff" stroke-width="2" marker-end="url(#arrow)"/><rect x="943" y="49" width="295" height="51" rx="8" fill="#203b4c"/><text x="958" y="70">Execute central handover</text><text x="958" y="87" class="caption">Delivery must occur strictly after command creation</text>
<rect x="1" y="139" width="1258" height="106" rx="12" fill="#142f2b" stroke="#3c6156"/><text x="18" y="161" class="title">DEVICE PATH · OWN LOCAL OBSERVATIONS ONLY</text>
<rect x="20" y="176" width="266" height="50" rx="8" fill="#234239"/><text x="33" y="197">Own completed paid scans + valid pilots</text><text x="33" y="214" class="caption">Freshness / signal floor · no neighbor-only inputs</text>
<path d="M286 201H332" stroke="#7bdfbe" stroke-width="2" marker-end="url(#mintarrow)"/><rect x="334" y="176" width="278" height="50" rx="8" fill="#234239"/><text x="347" y="197">Preinstalled blackout watchdog</text><text x="347" y="214" class="caption">200 ms silence trigger · dwell / backoff / UNKNOWN</text>
<path d="M612 201H658" stroke="#7bdfbe" stroke-width="2" marker-end="url(#mintarrow)"/><rect x="660" y="176" width="285" height="50" rx="8" fill="#234239"/><text x="673" y="197">Validate → local handover</text><text x="673" y="214" class="caption">Own authority · decision delay / scan interruption charged</text>
<text x="973" y="196">Shared scan budget</text><text x="973" y="214" class="caption">90 ms interruption · ≥5 s start spacing</text>
<path d="M112 139V121" stroke="#8697af" stroke-width="1.5" stroke-dasharray="4 4"/><text x="170" y="134" class="caption">A local scan becomes server evidence only after its report is delivered.</text>
</svg></div><p class="architecture-note">The server guard uses delivered reports within 5 m and 6 s, and an −80 dBm heard floor. The local watchdog uses scans no older than 5 s and a −75 dBm target floor. Neither path has direct access to the simulator’s hidden RF state.</p><div class="legend"><span><b>RadioMap</b> map only</span><span><b>+ recovery</b> own-device watchdog</span><span><b>Regression</b> frozen forecast only</span><span><b>+ support</b> delivered-history guard</span><span><b>+ both</b> both additions enabled</span></div></section>
<section class="card section"><h2>What this evidence supports</h2><div class="qualifications"><p><b>A finite native simulation diagnostic</b>Four seeds in one warehouse family, not four independent physical warehouses. Three carriers: 2.4 / 5.2 / 6.5 GHz; 6.5 GHz extrapolates training. No physical-device calibration; interference disabled.</p><p><b>A control-stack result, not a JEPA win</b>48 fresh simulations; 2 additional disabled-parity runs. Frozen regression, zero new neural training fits. Support is an AP-neighbor heuristic, not a packet-risk model. The causal command kernel differs from the earlier native diagnostic.</p><p><b>No default or deployment promotion</b>The 4 s failures are preserved. SERVICE remains unopened; no physical-car actions. No JEPA-objective, SOTA, multi-site generalization or production-reliability claim follows from these four episodes.</p></div><details><summary>All episode outcomes and secondary metrics</summary><div class="details-body"><p>Every arm has the same four episode seeds for each handover condition. Rows below use the selected handover condition. End-pending commands are shown separately from observed misses.</p><div class="table-wrap"><table id="episodes-table"><thead><tr><th>Episode seed</th><th>Arm</th><th>Loss<br>%</th><th>Worst decile<br>%</th><th>Max silence<br>s</th><th>End pending<br>commands</th><th>Video on time<br>%</th><th>Bulk<br>Mbps</th><th>Safety-stop<br>%</th></tr></thead><tbody></tbody></table></div></div></details><details><summary>Source identity, verification and reproducibility</summary><div class="details-body"><p>This local page reads the sealed report after verifying all 543 study source files and 224 evidence files, plus the passing independent raw-reduction review. Renderer independently reconciles all arm means and paired harm counts from the 48 recorded summary rows; it does not execute the simulator, models or raw reductions.</p><p>Scientific manifest SHA-256: <code>f652af5c0c1671a978ad0b2ebefcf35d7e5a7474dc36b373d63b4f60aa41d28f</code><br>Independent review SHA-256: <code>5f79df8dd60f4e45eb3ac300813a00d2c5a8fd8f724c4d380ce16fa2b2461df3</code></p><p>Rebuild locally: <code>python artifacts/spatial_control_review/render_results.py --browser</code>. The optional browser check needs installed Playwright and Chromium; the renderer otherwise uses only Python standard library. Local artifact only; no remote Studio publication.</p><p><a href="../spatial_control/report.json">Sealed report</a> · <a href="result.json">Independent audit</a> · <a href="provenance.json">Renderer provenance</a> · <a href="render_results.py">Render script</a></p></div></details></section>
<footer>All displayed numbers are derived from the sealed spatial-control report. Percentages ×100; millisecond gaps ÷1000. Per-episode maxima are averaged only where explicitly labeled. <span id="identity"></span></footer>
</main><script id="study-data" type="application/json">__DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('study-data').textContent);
const ORDER=['radiomap','radiomap_recovery','regression','regression_support','regression_recovery','regression_both'];
const LABELS={radiomap:'RadioMap',radiomap_recovery:'RadioMap + recovery',regression:'Regression',regression_support:'Regression + support',regression_recovery:'Regression + recovery',regression_both:'Regression + both'};
const COLORS={radiomap:'#7bdfbe',radiomap_recovery:'#a0c9ba',regression:'#8db9ff',regression_support:'#c5a1ff',regression_recovery:'#ffb27f',regression_both:'#f0a0c6'};
const f=(x,n=3)=>x.toFixed(n), mean=(h,a)=>DATA.arm_means.find(x=>x.handover_seconds===h&&x.arm===a).mean;
const text=(id,t)=>document.getElementById(id).textContent=t;
const mfast=mean(.03,'radiomap'), bfast=mean(.03,'regression_both'), rfast=mean(.03,'regression');
const rslow=mean(4,'regression'), bslow=mean(4,'regression_both');
text('fast-headline',`${f(mfast.control_miss*100)}% RadioMap vs ${f(bfast.control_miss*100)}% regression + both`);
text('fast-detail',`Both additions lower regression loss from ${f(rfast.control_miss*100)}%, but still lose to RadioMap in 4/4 episodes.`);
text('slow-headline',`${f(rslow.control_miss*100)}% regression → ${f(bslow.control_miss*100)}% with both`);
text('slow-detail',`Recovery increases misses in all 4/4 episodes versus its own baseline; regression + both mean episode maximum silence is ${f(bslow.maximum_silence_ms/1000)} s.`);
text('identity','Report '+DATA.report_sha256.slice(0,16)+'… · 48 fresh simulations · 0 neural fits.');
function element(tag,content,cls){const e=document.createElement(tag);if(content!==undefined)e.textContent=content;if(cls)e.className=cls;return e;}
function cell(tr,value,cls){tr.appendChild(element('td',value,cls));}
function armcell(tr,arm){const td=element('td',undefined,'armname'),dot=element('span',undefined,'dot');dot.style.background=COLORS[arm];td.append(dot,document.createTextNode(LABELS[arm]));tr.appendChild(td);}
function table(id,rows,make){const body=document.querySelector('#'+id+' tbody');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');make(tr,row);body.appendChild(tr);}}
const NS='http://www.w3.org/2000/svg';
function sv(tag,attrs,content){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(content!==undefined)e.textContent=content;return e;}
function chart(id,h,key,factor,max,unit){const svg=document.getElementById(id);svg.replaceChildren();svg.dataset.handover=h;svg.dataset.metric=key;const x=137,width=197,top=26,step=36;
 for(let t=0;t<=4;t++){const pos=x+width*t/4;svg.append(sv('line',{x1:pos,x2:pos,y1:17,y2:239,class:t===0?'axis':'grid'}),sv('text',{x:pos,y:260,'text-anchor':'middle','data-axis-value':max*t/4},f(max*t/4,Number.isInteger(max*t/4)?0:1)+unit));}
 ORDER.forEach((arm,i)=>{const value=mean(h,arm)[key]*factor,y=top+i*step;svg.append(sv('text',{x:0,y:y+15,class:'label'},LABELS[arm]),sv('rect',{x,y,width:width*value/max,height:22,rx:4,fill:COLORS[arm],'data-arm':arm,'data-value':value}),sv('text',{x:x+width*value/max+7,y:y+15,class:'value'},f(value,3)));});}
function render(h){document.body.dataset.handover=h;for(const b of document.querySelectorAll('[data-handover]'))b.setAttribute('aria-pressed',String(Number(b.dataset.handover)===h));
 text('selected-note',h===.03?'30 ms: support and recovery repair regression failures; RadioMap remains better on mean command loss.':'4 s: recovery hurts all four paired episodes versus its baseline; support alone gives only a small descriptive improvement.');
 chart('loss-chart',h,'control_miss',100,2,'%');chart('tail-chart',h,'control_miss_worst_decile',100,8,'%');chart('gap-chart',h,'maximum_silence_ms',.001,6,' s');
 const means=ORDER.map(a=>({arm:a,mean:mean(h,a)}));
 table('costs-table',means,(tr,r)=>{armcell(tr,r.arm);for(const[k,n]of[['actual_forecast_compute_seconds',3],['paid_forecast_compute_seconds',3],['scans_started',2],['scan_interruption_seconds',3]])cell(tr,f(r.mean[k],n));});
 table('local-table',means,(tr,r)=>{const m=r.mean;armcell(tr,r.arm);cell(tr,f(m.central_planning_cpu_seconds));cell(tr,f(m.local_decision_cpu_seconds)+' / '+f(m.local_validation_cpu_seconds,6));cell(tr,f(m.support_preparation_cpu_seconds));cell(tr,f(m.autonomous_scans,2));cell(tr,f(m.central_handovers,2)+' / '+f(m.local_handovers,2));});
 table('harms-table',DATA.contrasts.filter(r=>r.handover_seconds===h),(tr,r)=>{cell(tr,LABELS[r.candidate]+' vs '+LABELS[r.baseline]);const d=r.mean_delta.control_miss*100;cell(tr,(d>0?'+':'')+f(d),d>0?'bad':d<0?'good':'neutral');cell(tr,r.episodes_with_more_misses+'/4',r.episodes_with_more_misses?'bad':'neutral');cell(tr,r.episodes_with_larger_max_gap+'/4',r.episodes_with_larger_max_gap?'bad':'neutral');});
 const rows=DATA.rows.filter(r=>r.handover_seconds===h).sort((a,b)=>a.seed-b.seed||ORDER.indexOf(a.arm)-ORDER.indexOf(b.arm));
 table('episodes-table',rows,(tr,r)=>{const m=r.metrics;cell(tr,r.seed);armcell(tr,r.arm);for(const[k,s,n]of[['control_miss',100,3],['control_miss_worst_decile',100,3],['maximum_silence_ms',.001,3],['end_pending_commands',1,0],['video_on_time',100,2],['bulk_mbps',1,2],['safety_stop_fraction',100,3]])cell(tr,f(m[k]*s,n));});}
for(const button of document.querySelectorAll('button[data-handover]'))button.addEventListener('click',()=>render(Number(button.dataset.handover)));
render(.03);
</script></body></html>
'''


def browser_check():
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    # Serve repository root so the local relative evidence links also resolve.
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    checks = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                                args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/artifacts/spatial_control_review/index.html", wait_until="load")
            for h in (.03, 4.):
                page.locator(f'button[data-handover="{h:g}"]').click()
                require(page.locator("#costs-table tbody tr").count() == 6, "Browser lost an arm")
                require(page.locator("#harms-table tbody tr").count() == 6, "Browser lost a contrast")
                require(page.locator("#episodes-table tbody tr").count() == 24, "Browser lost episodes")
                for chart_id, key, factor in (("loss-chart", "control_miss", 100),
                                               ("tail-chart", "control_miss_worst_decile", 100),
                                               ("gap-chart", "maximum_silence_ms", .001)):
                    bars = page.locator(f"#{chart_id} rect[data-arm]").evaluate_all(
                        "xs => xs.map(x=>({arm:x.dataset.arm,value:Number(x.dataset.value)}))")
                    ticks = page.locator(f"#{chart_id} text[data-axis-value]").evaluate_all(
                        "xs => xs.map(x=>({value:Number(x.dataset.axisValue),label:parseFloat(x.textContent)}))")
                    require(len(ticks) == 5 and all(t["value"] == t["label"] for t in ticks), "Chart axis label mismatch")
                    require(len(bars) == 6 and [b["arm"] for b in bars] == ARMS, "Browser chart membership mismatch")
                    for bar in bars:
                        source = next(r["mean"][key] * factor for r in PAYLOAD["arm_means"]
                                      if r["handover_seconds"] == h and r["arm"] == bar["arm"])
                        require(math.isclose(bar["value"], source, rel_tol=1e-12), "Browser chart value mismatch")
                checks.append({"handover_seconds": h, "six_arms": True, "six_contrasts": True,
                               "episode_rows": 24, "all_18_chart_values_match_report": True, "all_15_axis_labels_exact": True})
            require("4/4" in page.locator("#fast-detail").inner_text(), "Fast RadioMap harms hidden")
            require("4/4" in page.locator("#slow-detail").inner_text(), "Slow recovery harms hidden")
            require(page.locator('button[data-handover="4"]').get_attribute("aria-pressed") == "true", "Selector state mismatch")
            page.locator('button[data-handover="0.03"]').click()
            page.screenshot(path=str(HERE / "preview.png"), full_page=True)
            require(not errors, "Browser JavaScript errors: " + str(errors))
            overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
            require(not overflow, "Desktop document overflow")
            page.set_viewport_size({"width": 390, "height": 844})
            require(not page.evaluate("document.documentElement.scrollWidth > window.innerWidth"), "Mobile document overflow")
            browser.close()
            return {"passes": True, "engine": "installed Chromium / Playwright", "page_errors": errors,
                    "handover_checks": checks, "both_negative_findings_always_visible": True,
                    "desktop_viewport": [1440, 1080], "mobile_width": 390,
                    "no_document_overflow": True, "remote_network_used": False,
                    "screenshot_sha256": sha(HERE / "preview.png")}
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true", help="Validate both selections and save preview.png with installed Chromium")
    args = parser.parse_args()
    report, _protocol, verified = verify_inputs()
    global PAYLOAD
    PAYLOAD = {key: report[key] for key in ("arm_means", "contrasts", "rows")}
    PAYLOAD["report_sha256"] = REPORT_SHA
    encoded = json.dumps(PAYLOAD, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (HERE / "index.html").write_text(HTML.replace("__DATA__", encoded))
    browser = browser_check() if args.browser else {"performed": False, "reason": "Pass --browser for optional installed-browser checks"}
    # Re-read all sealed bytes after rendering / browser interactions.
    _, _, after = verify_inputs()
    require(verified == after, "Input inventory changed during rendering")
    provenance = {"schema": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "renderer_sha256": sha(Path(__file__).resolve()),
                  "scientific_manifest_sha256": MANIFEST_SHA, "independent_audit_sha256": AUDIT_SHA,
                  "sealed_report_sha256": REPORT_SHA, "inputs_verified_before_and_after": verified,
                  "summary_rows": 48, "arm_condition_means_reconciled": 12,
                  "paired_contrasts_reconciled": 12,
                  "original_scientific_files_unchanged": True, "scientific_calls": 0,
                  "raw_array_decodes": 0, "project_scientific_imports": 0,
                  "new_fits": 0, "new_simulations": 0, "remote_publication": False,
                  "timing_scope": "Elapsed wall time on CPU, not process CPU time; reported allowances are separate.",
                  "browser": browser, "html_sha256": sha(HERE / "index.html")}
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passes": True, "html": str(HERE / "index.html"), "source_files_verified": verified["source_sha256"]["files"],
                      "evidence_files_verified": verified["evidence_sha256"]["files"], "browser": browser.get("passes", False),
                      "scientific_calls": 0}))


if __name__ == "__main__":
    main()
