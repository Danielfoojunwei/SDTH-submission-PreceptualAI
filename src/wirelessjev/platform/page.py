"""The Field Platform page: one file, no internet needed."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Field Platform</title>
<style>
:root{--bg:#F3F5F2;--panel:#FFFFFF;--panel2:#F7F9F8;--ink:#12202B;--muted:#5A6A75;--line:#D8E0E4;--accent:#D9661F;--accent2:#FCE9DC;--good:#2E7D5B;--warn:#B7791F;--bad:#C0392B;--blue:#2D6CB5}
@media (prefers-color-scheme: dark){:root{--bg:#0E1820;--panel:#15232E;--panel2:#1A2B37;--ink:#E7EDF1;--muted:#9AAEBB;--line:#2A3B47;--accent:#F08A4B;--accent2:#3A2A1E;--good:#4CBF8C;--warn:#E3B04B;--bad:#F06C5E;--blue:#6FA8E8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;gap:16px;align-items:center;justify-content:space-between;padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0}
.sub{color:var(--muted);font-size:12.5px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;padding:16px 20px}
@media (max-width:1000px){.layout{grid-template-columns:1fr}}
nav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
nav button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--panel);color:var(--ink);cursor:pointer}
nav button.on{background:var(--ink);color:var(--panel);border-color:var(--ink)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px;min-width:0}
.card h2{font-size:14px;margin:0 0 8px 0}
.k{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:6px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:700;white-space:nowrap}
.done{background:var(--good);color:#fff}.todo{background:var(--panel2);color:var(--muted);border:1px solid var(--line)}.part{background:var(--warn);color:#13212C}
.kind{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 5px;white-space:nowrap}
.kind.car_change{color:var(--bad);border-color:var(--bad)}
button.act{font:inherit;font-size:13px;padding:5px 12px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;white-space:nowrap}
button.act:disabled{opacity:.45;cursor:not-allowed}
button.ghost{font:inherit;font-size:13px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.cmd{display:block;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:6px 8px;overflow-wrap:anywhere;margin-top:4px}
pre.out{background:#0E1820;color:#D7E1E7;border-radius:8px;padding:10px;max-height:380px;overflow:auto;font-size:12px;margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.ok{color:var(--good)}.bad{color:var(--bad)}.warnc{color:var(--warn)}
.next{border-left:4px solid var(--accent);background:var(--accent2)}
.marks{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.marks button{font:700 20px/1 inherit;min-width:56px;padding:12px;border-radius:8px;border:2px solid var(--accent);background:var(--panel);color:var(--ink);cursor:pointer}
iframe{width:100%;height:760px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
input,select,textarea{font:inherit;font-size:13px;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);width:100%}
textarea{min-height:64px;resize:vertical}
.grid2{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.msg{padding:8px 10px;border-radius:8px;margin-top:8px;font-size:13px;white-space:pre-wrap}
.msg.err{background:var(--bad);color:#fff}.msg.info{background:var(--panel2);border:1px solid var(--line)}
.hide{display:none}
img.cov{max-width:100%;border:1px solid var(--line);border-radius:8px}
#env td:nth-child(3){overflow-wrap:anywhere;width:38%}#env td:nth-child(4){width:42%}#env td:nth-child(2){width:16%}
#aps td:first-child{min-width:150px}
#toast{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);max-width:min(640px,calc(100vw - 32px));background:var(--bad);color:#fff;padding:10px 14px;border-radius:8px;font-size:13.5px;box-shadow:0 6px 24px rgba(0,0,0,.25);z-index:10;cursor:pointer}
#toast.hide{display:none}#aps td:nth-child(2){min-width:80px}
</style>
</head>
<body>
<header>
  <div><h1>WirelessJEV Field Platform</h1><div class="sub" id="sub">Real SmartCar testing · radio mapping · WirelessJEV in shadow</div></div>
  <div class="sub" id="envbadges"></div>
</header>
<div class="layout">
<main>
  <nav id="tabs">
    <button data-tab="plan" class="on">Plan</button><button data-tab="setup">Setup</button><button data-tab="live">Live lap</button>
    <button data-tab="analysis">Analysis</button><button data-tab="output">Command output</button>
  </nav>

  <section id="tab-plan">
    <div class="card"><h2>The day, step by step</h2>
      <div class="sub">Read-only steps run from here. Steps marked <span class="kind car_change">changes car</span> are never run from this page: copy the command into a terminal, where it asks you to retype the target before sending.</div>
      <table id="steps"></table>
    </div>
  </section>

  <section id="tab-setup" class="hide">
    <div class="card"><h2>This laptop</h2><table id="env"></table></div>
    <div class="card"><h2>Site sheet</h2>
      <div class="sub">One row per BSSID (each band of each AP). Positions in metres from one corner of the room. Saved to <span class="mono" id="logdir"></span> as site.json and waypoints.json.</div>
      <div class="grid2" style="margin-top:8px">
        <label>Site name<input id="site_id" value="site-1"></label>
        <label>Length x (m)<input id="dim_x" type="number" step="0.1"></label>
        <label>Width y (m)<input id="dim_y" type="number" step="0.1"></label>
        <label>Height z (m)<input id="dim_z" type="number" step="0.1" value="4"></label>
        <label>Noise floor (dBm)<input id="noise" type="number" value="-92"></label>
      </div>
      <h2 style="margin-top:12px">Access points</h2>
      <table id="aps"><tr><th>BSSID</th><th>SSID</th><th>Freq MHz</th><th>x</th><th>y</th><th>z</th><th>Tx dBm</th><th></th></tr></table>
      <button class="ghost" id="addap">+ AP row</button>
      <h2 style="margin-top:12px">Waypoints (taped floor markers)</h2>
      <table id="wps"><tr><th>Label</th><th>x</th><th>y</th><th></th></tr></table>
      <button class="ghost" id="addwp">+ waypoint</button>
      <div style="margin-top:12px"><button class="act" id="savesite">Save site sheet</button></div>
      <div id="sitemsg"></div>
    </div>
  </section>

  <section id="tab-live" class="hide">
    <div class="card" id="markcard"><h2>Waypoint marks</h2>
      <div class="sub" id="markhint">Marks appear here while a lap is recording. Press the letter as the car passes each taped marker.</div>
      <div class="marks" id="marks"></div><div id="markmsg"></div>
    </div>
    <iframe src="/live" title="Live lap"></iframe>
  </section>

  <section id="tab-analysis" class="hide">
    <div class="card"><h2>Analysis</h2>
      <div class="sub">Runs on this laptop from the recorded laps; nothing touches the car.</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px" id="anabtns"></div>
    </div>
    <div class="card"><h2>Coverage map</h2><div id="covwrap" class="sub">No coverage report yet.</div></div>
  </section>

  <section id="tab-output" class="hide">
    <div class="card"><h2 id="outtitle">Command output</h2>
      <div class="sub mono" id="outcmd"></div>
      <div style="margin:8px 0"><button class="ghost" id="stopbtn" style="display:none">Stop the running command</button></div>
      <pre class="out" id="out">No command has run yet.</pre>
    </div>
    <div class="card"><h2>Recent commands</h2><table id="jobs"></table></div>
  </section>
</main>

<aside>
  <div class="card next"><div class="k">Next step</div><div id="nexttext" style="margin-top:6px">…</div><div id="nextact" style="margin-top:8px"></div></div>
  <div class="card"><h2>Ask the assistant</h2>
    <textarea id="q" placeholder="What should I do next? Why did the format check fail?"></textarea>
    <div style="margin-top:8px;display:flex;gap:8px"><button class="act" id="ask">Ask</button><button class="ghost" id="asknext">What next?</button></div>
    <div id="answer"></div>
  </div>
  <div class="card"><div class="k">Format check (last run)</div><table id="check" style="margin-top:6px"></table></div>
</aside>
</div>
<div id="toast" class="hide" role="alert"></div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let S = null, watchJob = null, siteLoaded = false;
async function api(path, body){
  const r = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll("main > section").forEach(s => s.classList.toggle("hide", s.id !== "tab-" + b.dataset.tab));
});
function show(tab){ document.querySelector(`#tabs button[data-tab="${tab}"]`).click(); }
function pill(st){ return st.complete ? '<span class="pill done">done</span>' : st.done ? `<span class="pill part">${st.done}/${st.wanted}</span>` : `<span class="pill todo">0/${st.wanted}</span>`; }
function copy(text, el){ navigator.clipboard?.writeText(text).then(() => { if (el){ const t = el.textContent; el.textContent = "Copied"; setTimeout(() => el.textContent = t, 1200); } }); }

let stepsKey = "", nextKey = "";
const keys = {};
function changed(name, value){ const k = JSON.stringify(value); if (keys[name] === k) return false; keys[name] = k; return true; }
function renderSteps(){
  const busy = !!S.running;
  const key = JSON.stringify([busy, S.steps.map(s => [s.id, s.done, s.wanted, s.complete, s.command])]);
  if (key === stepsKey) return;                      // nothing changed: keep what the operator is typing
  stepsKey = key;
  const kept = {}; document.querySelectorAll("#steps input").forEach(i => kept[i.id] = i.value);
  const focus = document.activeElement && document.activeElement.id;
  let html = '<tr><th>Block</th><th>Step</th><th>Status</th><th></th></tr>';
  for (const s of S.steps){
    let act = "";
    if (s.kind === "read" || s.kind === "analysis"){
      const lap = s.lap_prefix ? `<div style="display:flex;gap:6px;margin-top:6px"><input id="sec-${s.id}" value="180" title="seconds" style="width:70px"><input id="note-${s.id}" placeholder="note (e.g. blocker at C)"></div>` : "";
      act = `<button class="act" data-run="${s.id}" ${busy ? "disabled" : ""}>Run</button>${lap}`;
    } else if (s.kind === "car_change"){
      act = `<button class="ghost" data-copy="${esc(s.command)}">Copy command</button><code class="cmd">${esc(s.command)}</code>`;
    } else { act = `<button class="ghost" data-tabgo="setup">Open Setup</button>`; }
    html += `<tr><td>${esc(s.block)}</td><td><b>${esc(s.title)}</b> <span class="kind ${s.kind}">${s.kind === "car_change" ? "changes car" : s.kind}</span><div class="sub">${esc(s.why)}</div></td><td>${pill(s)}</td><td>${act}</td></tr>`;
  }
  $("steps").innerHTML = html;
  Object.entries(kept).forEach(([id, v]) => { if ($(id)) $(id).value = v; });
  if (focus && $(focus)) $(focus).focus();
  document.querySelectorAll("[data-run]").forEach(b => b.onclick = () => run(b.dataset.run));
  document.querySelectorAll("[data-copy]").forEach(b => b.onclick = () => copy(b.dataset.copy, b));
  document.querySelectorAll("[data-tabgo]").forEach(b => b.onclick = () => show(b.dataset.tabgo));
}
async function run(id){
  const extra = {}; const sec = $("sec-" + id), note = $("note-" + id);
  if (sec) extra.seconds = sec.value; if (note && note.value) extra.note = note.value;
  try { const j = await api("/api/run", {step:id, extra}); watchJob = j.id; show(S.steps.find(s => s.id === id).lap_prefix ? "live" : "output"); tick(); }
  catch (e){ toast(e.message); }
}
let toastTimer = null;
function toast(msg){ const t = $("toast"); t.textContent = msg; t.classList.remove("hide"); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hide"), 9000); }
function renderEnv(){
  $("env").innerHTML = '<tr><th></th><th>Item</th><th>Status</th><th>How to fix</th></tr>' + S.environment.map(r =>
    `<tr><td class="${r.ok ? "ok" : "bad"}">${r.ok ? "✓" : "✕"}</td><td>${esc(r.item)}</td><td class="mono">${esc(r.seen)}</td><td>${r.ok ? "" : `<code class="cmd">${esc(r.fix)}</code>`}</td></tr>`).join("");
  $("envbadges").innerHTML = S.environment.map(r => `<span class="${r.ok ? "ok" : "bad"}">${r.ok ? "●" : "○"} ${esc(r.item)}</span>`).join(" &nbsp; ");
  $("logdir").textContent = S.log_dir;
  $("sub").textContent = S.demo ? "DEMO MODE: synthetic lap, car commands switched off" : "Real SmartCar testing · radio mapping · WirelessJEV in shadow";
}
function renderNext(){
  const n = S.next; const key = JSON.stringify([n, !!S.running, S.demo]);
  if (key === nextKey) return; nextKey = key;
  $("nexttext").textContent = n.text; let a = "";
  const step = S.steps.find(s => s.id === n.step);
  if (n.action === "run" && step && !S.running) a = `<button class="act" id="nextrun">Run: ${esc(step.title)}</button>`;
  if (n.command) a = `<button class="ghost" id="nextcopy">Copy command</button><code class="cmd">${esc(n.command)}</code>`;
  if (n.action === "setup") a = `<button class="ghost" id="nextsetup">Open Setup</button>`;
  if (S.demo && step && step.kind === "read") a = `<div class="sub">Demo mode: commands that talk to the car are switched off. Try the Analysis tab on the synthetic lap.</div>`;
  $("nextact").innerHTML = a;
  if ($("nextrun")) $("nextrun").onclick = () => run(n.step);
  if ($("nextcopy")) $("nextcopy").onclick = e => copy(n.command, e.target);
  if ($("nextsetup")) $("nextsetup").onclick = () => show("setup");
}
function renderCheck(){
  if (!changed("check", S.check)) return;
  $("check").innerHTML = S.check ? S.check.map(r => `<tr><td><span class="pill ${r.status === "PASS" ? "done" : r.status === "WARN" ? "part" : "todo"}" style="${r.status === "FAIL" ? "background:var(--bad);color:#fff;border:none" : ""}">${r.status}</span></td><td>${esc(r.item)}<div class="sub">${esc(r.seen)}</div></td></tr>`).join("") : '<tr><td class="sub">Not run yet.</td></tr>';
}
function renderMarks(){
  const lapRunning = S.running && S.steps.find(s => s.id === S.running.step)?.lap_prefix;
  const wps = S.site.waypoints ? Object.keys(S.site.waypoints) : [];
  markLabels = lapRunning ? wps : [];
  if (!changed("marks", [!!lapRunning, wps])) return;
  $("marks").innerHTML = lapRunning ? wps.map(w => `<button data-mark="${esc(w)}">${esc(w)}</button>`).join("") : "";
  $("markhint").textContent = lapRunning ? (wps.length ? "Recording. Press the letter as the car passes each marker." : "Recording, but no waypoints are saved: add them on the Setup tab.") : "Marks appear here while a lap is recording. Press the letter as the car passes each taped marker.";
  document.querySelectorAll("[data-mark]").forEach(b => b.onclick = () => mark(b.dataset.mark));
}
let markLabels = [];
async function mark(label){
  try { await api("/api/mark", {label}); $("markmsg").innerHTML = `<div class="msg info">Marked ${esc(label)} at ${new Date().toLocaleTimeString()}</div>`; }
  catch (e){ $("markmsg").innerHTML = `<div class="msg err">${esc(e.message)}</div>`; }
}
document.addEventListener("keydown", e => {               // while a lap records, the waypoint's letter on the keyboard marks it
  if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  const hit = markLabels.find(l => l.toLowerCase() === e.key.toLowerCase());
  if (hit){ e.preventDefault(); mark(hit); }
});
function renderAnalysis(){
  const ids = ["analyse", "twin", "coverage", "propose"]; const busy = !!S.running;
  const covJob = [...S.jobs].reverse().find(j => j.step === "coverage" && j.returncode === 0);
  if (!changed("analysis", [busy, covJob && covJob.id])) return;
  $("anabtns").innerHTML = ids.map(id => { const s = S.steps.find(x => x.id === id); return `<button class="act" data-arun="${id}" ${busy ? "disabled" : ""}>${esc(s.title)}</button>`; }).join("");
  document.querySelectorAll("[data-arun]").forEach(b => b.onclick = () => run(b.dataset.arun));
  if (covJob) $("covwrap").innerHTML = `<img class="cov" src="/files/coverage.png?job=${covJob.id}" alt="Coverage map from the recorded laps">`;
}
function renderJobs(){
  if (!changed("jobs", S.jobs)) return;
  $("jobs").innerHTML = '<tr><th>#</th><th>Step</th><th>Result</th><th>Seconds</th></tr>' + S.jobs.slice().reverse().map(j =>
    `<tr><td><a href="#" data-job="${j.id}">${j.id}</a></td><td>${esc(j.step)}</td><td class="${j.running ? "warnc" : j.returncode === 0 ? "ok" : "bad"}">${j.running ? "running" : j.returncode === 0 ? "ok" : "exit " + j.returncode}</td><td>${j.seconds}</td></tr>`).join("");
  document.querySelectorAll("[data-job]").forEach(a => a.onclick = e => { e.preventDefault(); watchJob = +a.dataset.job; show("output"); tick(); });
}
async function renderOutput(){
  const id = watchJob ?? (S.jobs.length ? S.jobs[S.jobs.length - 1].id : null);
  if (id == null) return;
  const j = await api("/api/job/" + id);
  $("outtitle").textContent = `Command output · ${j.step} · ${j.running ? "running" : (j.returncode === 0 ? "finished" : "exit " + j.returncode)}`;
  $("stopbtn").style.display = j.running ? "" : "none";
  $("outcmd").textContent = j.command;
  const out = $("out"), atEnd = out.scrollTop + out.clientHeight >= out.scrollHeight - 20;
  out.textContent = j.output.join("\n") || "(no output yet)"; if (atEnd) out.scrollTop = out.scrollHeight;
}
function loadSite(){
  if (siteLoaded || !S) return; siteLoaded = true;
  const s = S.site.site, w = S.site.waypoints;
  if (s){ $("site_id").value = s.site_id; [$("dim_x").value, $("dim_y").value, $("dim_z").value] = s.dimensions_m; $("noise").value = s.receiver.noise_floor_dbm; s.aps.forEach(a => addAp(a)); }
  else { addAp(); addAp(); }
  if (w) Object.entries(w).forEach(([k, v]) => addWp(k, v)); else { addWp("A", ["", ""]); addWp("B", ["", ""]); }
}
function addAp(a){
  const tr = document.createElement("tr"); a = a || {bssid:"", ssid:"", frequency_mhz:5180, xyz_m:["", "", 2.5], tx_power_dbm:20};
  tr.innerHTML = `<td><input class="b" value="${esc(a.bssid)}" placeholder="aa:bb:cc:dd:ee:ff"></td><td><input class="s" value="${esc(a.ssid)}"></td><td><input class="f" type="number" value="${a.frequency_mhz}"></td><td><input class="x" type="number" step="0.1" value="${a.xyz_m[0]}"></td><td><input class="y" type="number" step="0.1" value="${a.xyz_m[1]}"></td><td><input class="z" type="number" step="0.1" value="${a.xyz_m[2]}"></td><td><input class="t" type="number" value="${a.tx_power_dbm}"></td><td><button class="ghost">✕</button></td>`;
  tr.querySelector("button").onclick = () => tr.remove(); $("aps").appendChild(tr);
}
function addWp(k, v){
  const tr = document.createElement("tr"); v = v || ["", ""];
  tr.innerHTML = `<td><input class="l" value="${esc(k || "")}"></td><td><input class="x" type="number" step="0.1" value="${v[0]}"></td><td><input class="y" type="number" step="0.1" value="${v[1]}"></td><td><button class="ghost">✕</button></td>`;
  tr.querySelector("button").onclick = () => tr.remove(); $("wps").appendChild(tr);
}
$("addap").onclick = () => addAp(); $("addwp").onclick = () => addWp();
$("savesite").onclick = async () => {
  const num = v => v === "" ? NaN : Number(v);
  const aps = [...$("aps").querySelectorAll("tr")].slice(1).map(tr => ({bssid: tr.querySelector(".b").value.trim().toLowerCase(), ssid: tr.querySelector(".s").value,
    frequency_mhz: num(tr.querySelector(".f").value), xyz_m: [".x", ".y", ".z"].map(c => num(tr.querySelector(c).value)), tx_power_dbm: num(tr.querySelector(".t").value)}));
  const waypoints = {}; [...$("wps").querySelectorAll("tr")].slice(1).forEach(tr => { const l = tr.querySelector(".l").value.trim(); if (l) waypoints[l] = [num(tr.querySelector(".x").value), num(tr.querySelector(".y").value)]; });
  const site = {schema: 1, site_id: $("site_id").value.trim(), dimensions_m: [num($("dim_x").value), num($("dim_y").value), num($("dim_z").value)], aps, cad_boxes_m: [],
    receiver: {noise_floor_dbm: num($("noise").value), offset_db: 0.0, quantization_db: 1.0, reference_gauge_assumed: true}};
  const bad = aps.findIndex(a => !a.bssid || a.xyz_m.some(Number.isNaN));
  if (bad >= 0){ $("sitemsg").innerHTML = `<div class="msg err">Not saved: AP row ${bad + 1} needs a BSSID and x, y, z. Fill it in or remove the row (✕).</div>`; return; }
  try { const r = await api("/api/site", {site, waypoints}); $("sitemsg").innerHTML = `<div class="msg info">Saved ${r.aps} AP rows and ${Object.keys(waypoints).length} waypoints.</div>`; tick(); }
  catch (e){ $("sitemsg").innerHTML = `<div class="msg err">Not saved: ${esc(e.message)}</div>`; }
};
async function ask(q){
  $("answer").innerHTML = '<div class="msg info">Thinking…</div>';
  try { const r = await api("/api/ask", {question: q});
    const step = S.steps.find(s => s.id === r.suggested_step); let btn = "";
    if (step && (step.kind === "read" || step.kind === "analysis") && !S.running) btn = `<div style="margin-top:6px"><button class="act" id="ansrun">Run: ${esc(step.title)}</button></div>`;
    if (step && step.kind === "car_change") btn = `<code class="cmd">${esc(step.command)}</code>`;
    $("answer").innerHTML = `<div class="msg info">${esc(r.answer)}<div class="sub" style="margin-top:6px">${r.source === "claude" ? "Answered by Claude" : "Built-in guide"}${r.note ? " · " + esc(r.note) : ""}</div>${btn}</div>`;
    if ($("ansrun")) $("ansrun").onclick = () => run(step.id);
  } catch (e){ $("answer").innerHTML = `<div class="msg err">${esc(e.message)}</div>`; }
}
$("ask").onclick = () => ask($("q").value); $("asknext").onclick = () => ask("What should I do next?");
$("stopbtn").onclick = async () => { if (confirm("Stop the running command?")) { await api("/api/stop", {}); tick(); } };
let busy = false;
async function tick(){
  if (busy) return; busy = true;
  try { S = await api("/api/state"); renderSteps(); renderEnv(); renderNext(); renderCheck(); renderMarks(); renderAnalysis(); renderJobs(); loadSite();
    if (!$("tab-output").classList.contains("hide") || S.running) await renderOutput(); }
  catch (e){ $("nexttext").textContent = "The platform process is not answering. Is it still running in the terminal?"; }
  finally { busy = false; }
}
$("toast").onclick = () => $("toast").classList.add("hide");
tick(); setInterval(tick, 1500);
</script>
</body>
</html>
"""
