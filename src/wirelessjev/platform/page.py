"""The Field Platform page: one file, no internet needed."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Field Platform</title>
<style>
:root{--ap1:#2a78d6;--ap2:#6250d6;--ap3:#1baf7a;--ap4:#e87ba4;--ap5:#eb6834;--bg:#F3F5F2;--panel:#FFFFFF;--panel2:#F7F9F8;--ink:#12202B;--muted:#5A6A75;--line:#D8E0E4;--accent:#D9661F;--accent2:#FCE9DC;--good:#2E7D5B;--warn:#B7791F;--bad:#C0392B;--blue:#2D6CB5}
@media (prefers-color-scheme: dark){:root{--ap1:#3987e5;--ap2:#9085e9;--ap3:#199e70;--ap4:#d55181;--ap5:#d95926;--bg:#0E1820;--panel:#15232E;--panel2:#1A2B37;--ink:#E7EDF1;--muted:#9AAEBB;--line:#2A3B47;--accent:#F08A4B;--accent2:#3A2A1E;--good:#4CBF8C;--warn:#E3B04B;--bad:#F06C5E;--blue:#6FA8E8}}
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
.cmd{display:block;white-space:pre-wrap;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:6px 8px;overflow-wrap:anywhere;margin-top:4px}
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
#toast.hide{display:none}
#banner{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px 20px;background:var(--accent2);border-bottom:1px solid var(--line);font-size:13.5px}
#banner.hide{display:none}
.step{display:grid;grid-template-columns:28px 1fr;gap:8px;padding:8px 0;border-bottom:1px solid var(--line)}
.step .n{width:24px;height:24px;border-radius:50%;background:var(--ink);color:var(--panel);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700}
.step .ok{background:var(--good)}
#maptip{position:absolute;pointer-events:none;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px;line-height:1.4;box-shadow:0 4px 16px rgba(0,0,0,.18);white-space:nowrap;z-index:5}
#maptip.hide{display:none}
.scale{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.scale .bar{height:10px;flex:1;min-width:160px;border-radius:3px;border:1px solid var(--line)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:8px}
.tile{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 10px;min-width:0}
.tile .v{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}.tile .l{font-size:11.5px;color:var(--muted)}
svg.map{display:block;width:100%;height:auto;background:var(--panel2);border:1px solid var(--line);border-radius:8px;margin-top:8px}
svg.tl{display:block;width:100%;height:150px;background:var(--panel2);border:1px solid var(--line);border-radius:8px}
.layers{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;align-items:center;margin-top:8px}
.layers label{display:flex;gap:5px;align-items:center;cursor:pointer}.layers input{width:auto;margin:0}
.legend{display:flex;gap:6px 14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:6px;align-items:center}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
#modal{position:fixed;inset:0;background:rgba(8,16,22,.6);display:flex;align-items:center;justify-content:center;z-index:20;padding:16px}
#modal.hide{display:none}
#modal .box{background:var(--panel);color:var(--ink);border-radius:12px;max-width:600px;width:100%;padding:18px;border:2px solid var(--bad);box-shadow:0 10px 40px rgba(0,0,0,.35)}
button.change{font:inherit;font-size:13px;padding:5px 12px;border-radius:6px;border:1px solid var(--bad);background:var(--panel);color:var(--bad);cursor:pointer;white-space:nowrap;font-weight:600}
button.change:disabled{opacity:.45;cursor:not-allowed}
select{font:inherit;font-size:13px;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
.setgroup h3{font-size:13px;margin:14px 0 4px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.setgroup td:nth-child(2){width:210px}.setgroup tr.chg td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
.hold{border-left:4px solid var(--bad)}
details summary{cursor:pointer}
.chk{display:flex;gap:8px;align-items:flex-start;padding:6px 0;border-bottom:1px solid var(--line)}.chk input{width:auto;margin-top:3px}
.scroll{overflow-x:auto}#aps td:nth-child(2){min-width:80px}
</style>
</head>
<body>
<header>
  <div><h1>WirelessJEV Field Platform</h1><div class="sub" id="sub">Real SmartCar testing · radio mapping · WirelessJEV in shadow</div></div>
  <div class="sub" id="envbadges"></div>
</header>
<div id="banner" class="hide"><span id="bannertext"></span> <button class="ghost" id="bannergo">How to set it up</button> <button class="ghost" id="bannerx" aria-label="Hide this notice">Hide</button></div>
<div class="layout">
<main>
  <nav id="tabs">
    <button data-tab="plan" class="on">Plan</button><button data-tab="map">Map</button><button data-tab="setup">Setup</button>
    <button data-tab="live">Live lap</button><button data-tab="analysis">Analysis</button><button data-tab="settings">Settings</button>
    <button data-tab="activity">Activity</button><button data-tab="output">Command output</button><button data-tab="help">Help</button>
  </nav>

  <section id="tab-plan">
    <div class="card"><h2>The day, step by step</h2>
      <div class="sub">Read-only steps run from here. Steps marked <span class="kind car_change">changes car</span> start here too, but the car's state is read first and nothing is sent until you type the target in the confirm box. One command at a time.</div>
      <table id="steps"></table>
    </div>
  </section>

  <section id="tab-map" class="hide">
    <div class="card">
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:space-between">
        <h2 style="margin:0">Site radio map</h2>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <label style="display:flex;gap:6px;align-items:center;font-size:13px">Show <select id="mapview"></select></label>
          <label style="display:flex;gap:6px;align-items:center;font-size:13px">Lap <select id="maplap"></select></label></div>
      </div>
      <div class="layers" id="layers">
        <label><input type="checkbox" data-layer="fade" checked>Fade where not measured</label>
        <label><input type="checkbox" data-layer="contours" checked>Good / Fair lines</label>
        <label><input type="checkbox" data-layer="boxes" checked>Obstructions</label>
        <label><input type="checkbox" data-layer="dots">Car's readings</label>
        <label><input type="checkbox" data-layer="route" checked>Route</label>
        <label><input type="checkbox" data-layer="roams" checked>Roams</label>
        <label><input type="checkbox" data-layer="outages" checked>Outages</label>
        <label><input type="checkbox" data-layer="car" checked>Car</label>
      </div>
      <div style="position:relative"><svg id="map" class="map" viewBox="0 0 24 14" role="img" aria-label="Site radio map"></svg>
        <div id="maptip" class="hide"></div></div>
      <div id="mapscale" style="margin-top:8px"></div>
      <div class="legend" id="maplegend"></div>
      <div class="sub" id="mapnote" style="margin-top:6px"></div>
    </div>
    <div class="card"><h2>Link health</h2><div class="tiles" id="tiles"></div><div class="sub" id="healthdef" style="margin-top:8px"></div></div>
    <div class="card"><h2>Signal and gateway replies over the lap</h2>
      <svg id="tl" class="tl" viewBox="0 0 1000 150" preserveAspectRatio="none" role="img" aria-label="Signal over time"></svg>
      <div class="legend" id="tllegend"></div></div>
    <div class="card"><h2>Access points</h2><div class="scroll"><table id="aptable"></table></div></div>
    <div class="card"><h2>Events</h2><div class="scroll" style="max-height:320px;overflow-y:auto"><table id="events"></table></div></div>
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
    <div class="card"><h2>Router checklist (in each router's app)</h2>
      <div class="sub">Consumer routers are set up in their own app; the platform cannot change them. Tick each item once it is done on every router.</div>
      <div id="checklist" style="margin-top:6px"></div></div>
    <div class="card"><h2>Router check: what the car hears against the site sheet</h2>
      <div class="sub" id="rcsrc"></div><table id="rcrows" style="margin-top:6px"></table>
      <div style="margin-top:8px"><button class="ghost" id="rcrefresh">Check again</button></div></div>
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

  <section id="tab-settings" class="hide">
    <div class="card"><h2>Settings</h2>
      <div class="sub">Every value the platform's commands, map and assistant use. Changes are checked before saving, apply to the next command and are recorded in Activity. A marked row differs from the runbook's default.</div>
      <div id="setgroups"></div>
      <div style="display:flex;gap:8px;margin-top:12px;align-items:center;flex-wrap:wrap;position:sticky;bottom:0;background:var(--panel);padding:8px 0">
        <button class="act" id="setsave">Save changes</button><button class="ghost" id="setundo">Undo unsaved edits</button><span id="setmsg"></span></div>
    </div>
    <div class="card"><h2>Ghost traffic command (run in a second terminal during the load lap)</h2>
      <code class="cmd" id="ghostcmd"></code><div style="margin-top:6px"><button class="ghost" id="ghostcopy">Copy</button></div></div>
  </section>
  <section id="tab-activity" class="hide">
    <div class="card"><h2>Activity</h2>
      <div class="sub">Every action, newest first: commands and how they ended, confirmations, setting changes, site saves, checklist ticks, marks and assistant questions. Saved as platform_audit.jsonl next to the laps.</div>
      <div class="scroll"><table id="activity" style="margin-top:6px"></table></div></div>
  </section>

  <section id="tab-help" class="hide">
    <div class="card"><h2>Before the site day (needs internet)</h2><div id="before"></div></div>
    <div class="card"><h2>How to use the platform</h2>
      <input id="helpq" placeholder="Search: map, confirm, settings, router, UNKNOWN…" autocomplete="off">
      <div id="topics" style="margin-top:8px"></div></div>
    <div class="card"><h2>What's new</h2><div id="news"></div></div>
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
<div id="modal" class="hide"><div class="box" role="dialog" aria-modal="true" aria-labelledby="mtitle">
  <h2 id="mtitle" style="font-size:16px;margin:0 0 8px">Confirm a change to the car's radio</h2>
  <div class="sub">The car's current state has just been read. This is exactly what will be sent:</div>
  <div class="cmd mono" id="mprompt" style="margin-top:6px"></div>
  <label style="display:block;margin-top:12px;font-size:13px">Type the target yourself<input id="mtext" autocomplete="off" spellcheck="false" autocapitalize="off"></label>
  <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap"><button class="change" id="msend">Send to the car</button><button class="ghost" id="mcancel">Cancel (nothing is sent)</button></div>
  <div class="sub" style="margin-top:8px">A wrong or empty answer cancels with nothing sent. If the outcome comes back UNKNOWN, further changes are held until a Car status read.</div>
</div></div>
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
  const t = b.dataset.tab;
  if (t === "map") { loadLaps(); refreshMap(true); refreshSpectrum(true); }
  if (t === "setup") loadRouters();
  if (t === "settings") loadSettings();
  if (t === "activity") loadActivity();
  if (t === "help") loadHelp();
});
const visible = t => !$("tab-" + t).classList.contains("hide");
function show(tab){ document.querySelector(`#tabs button[data-tab="${tab}"]`).click(); }
function pill(st){ return st.complete ? '<span class="pill done">done</span>' : st.done ? `<span class="pill part">${st.done}/${st.wanted}</span>` : `<span class="pill todo">0/${st.wanted}</span>`; }
function copy(text, el){ navigator.clipboard?.writeText(text).then(() => { if (el){ const t = el.textContent; el.textContent = "Copied"; setTimeout(() => el.textContent = t, 1200); } }); }

let stepsKey = "", nextKey = "";
const keys = {};
function changed(name, value){ const k = JSON.stringify(value); if (keys[name] === k) return false; keys[name] = k; return true; }
function renderSteps(){
  const busy = !!S.running;
  const key = JSON.stringify([busy, S.hold, S.demo, S.targets, S.lap_seconds, S.steps.map(s => [s.id, s.done, s.wanted, s.complete, s.command])]);
  if (key === stepsKey) return;                      // nothing changed: keep what the operator is typing
  stepsKey = key;
  const kept = {}; document.querySelectorAll("#steps input").forEach(i => kept[i.id] = i.value);
  const focus = document.activeElement && document.activeElement.id;
  let html = '<tr><th>Block</th><th>Step</th><th>Status</th><th></th></tr>';
  for (const s of S.steps){
    let act = "";
    if (s.kind === "read" || s.kind === "analysis"){
      const lap = s.lap_prefix ? `<div style="display:flex;gap:6px;margin-top:6px"><input id="sec-${s.id}" placeholder="${S.lap_seconds} s" title="lap length in seconds (empty: the Settings value)" style="width:80px"><input id="note-${s.id}" placeholder="note (e.g. blocker at C)"></div>` : "";
      act = `<button class="act" data-run="${s.id}" ${busy ? "disabled" : ""}>Run</button>${lap}`;
    } else if (s.kind === "car_change"){
      const ssids = [...new Set(S.targets.map(t => t[0]))];
      let pick = "";
      if (s.id === "join") pick = `<select id="tgt-${s.id}" title="test network">${ssids.map(x => `<option>${esc(x)}</option>`).join("")}</select>`;
      if (s.id === "switch") pick = `<select id="tgt-${s.id}" title="target AP">${S.targets.map(t => `<option value="${esc(t[0])}|${esc(t[1])}">${esc(t[1])} (${esc(t[0])})</option>`).join("")}</select>`;
      if (s.id === "roam") pick = `<select id="tgt-${s.id}" title="target AP">${S.targets.map(t => `<option value="|${esc(t[1])}">${esc(t[1])}</option>`).join("")}</select>`;
      const none = !S.targets.length ? '<div class="sub">Save the site sheet first: targets come from it.</div>' : "";
      act = `<div style="display:flex;gap:6px;flex-wrap:wrap">${pick}<button class="change" data-change="${s.id}" ${busy || S.hold || S.demo || !S.targets.length ? "disabled" : ""}>Start (you confirm)</button></div>${none}
        <details style="margin-top:4px"><summary class="sub">Same command for a terminal</summary><code class="cmd">${esc(s.command)}</code><button class="ghost" data-copy="${esc(s.command)}">Copy</button></details>`;
    } else { act = `<button class="ghost" data-tabgo="setup">Open Setup</button>`; }
    html += `<tr><td>${esc(s.block)}</td><td><b>${esc(s.title)}</b> <span class="kind ${s.kind}">${s.kind === "car_change" ? "changes car" : s.kind}</span><div class="sub">${esc(s.why)}</div></td><td>${pill(s)}</td><td>${act}</td></tr>`;
  }
  $("steps").innerHTML = html;
  Object.entries(kept).forEach(([id, v]) => { if ($(id)) $(id).value = v; });
  if (focus && $(focus)) $(focus).focus();
  document.querySelectorAll("[data-run]").forEach(b => b.onclick = () => run(b.dataset.run));
  document.querySelectorAll("[data-change]").forEach(b => b.onclick = () => startChange(b.dataset.change));
  document.querySelectorAll("[data-copy]").forEach(b => b.onclick = () => copy(b.dataset.copy, b));
  document.querySelectorAll("[data-tabgo]").forEach(b => b.onclick = () => show(b.dataset.tabgo));
}
async function run(id){
  const extra = {}; const sec = $("sec-" + id), note = $("note-" + id);
  if (sec && sec.value) extra.seconds = sec.value; if (note && note.value) extra.note = note.value;
  try { const j = await api("/api/run", {step:id, extra}); watchJob = j.id; show(S.steps.find(s => s.id === id).lap_prefix ? "live" : "output"); tick(); }
  catch (e){ toast(e.message); }
}
async function startChange(id){
  const sel = $("tgt-" + id); const extra = {};
  if (sel){ if (id === "join") extra.ssid = sel.value; else { const [ssid, bssid] = sel.value.split("|"); if (ssid) extra.ssid = ssid; extra.bssid = bssid; } }
  try { const j = await api("/api/run", {step:id, extra}); watchJob = j.id; show("output"); tick(); }
  catch (e){ toast(e.message); }
}
// ---- confirm box: opens once per command that asks; the operator types the target
let modalJob = null;
function renderModal(){
  const r = S.running;
  if (r && r.confirm && modalJob !== r.id){
    modalJob = r.id; $("mprompt").textContent = r.confirm; $("mtext").value = ""; $("modal").classList.remove("hide"); $("mtext").focus();
  }
  if ((!r || !r.confirm) && !$("modal").classList.contains("hide")) $("modal").classList.add("hide");
}
async function sendConfirm(text){
  const id = modalJob; $("modal").classList.add("hide");
  try { const r = await api("/api/confirm", {job:id, text}); toast(r.cancelled ? "Cancelled: nothing was sent." : "Sent your confirmation; the car's command now checks it."); tick(); }
  catch (e){ toast(e.message); }
}
$("msend").onclick = () => sendConfirm($("mtext").value);
$("mcancel").onclick = () => sendConfirm("");
$("mtext").addEventListener("keydown", e => { if (e.key === "Enter") sendConfirm($("mtext").value); if (e.key === "Escape") sendConfirm(""); });
let toastTimer = null;
function toast(msg){ const t = $("toast"); t.style.background = /^(Sent|Saved|Cancelled)/.test(msg) ? "var(--ink)" : ""; t.textContent = msg; t.classList.remove("hide"); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hide"), 9000); }
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
  if (n.action === "car_change") a = `<button class="change" id="nextplan">Go to the step</button><details style="margin-top:6px"><summary class="sub">Terminal command</summary><code class="cmd">${esc(n.command)}</code></details>`;
  if (n.action === "confirm") a = `<button class="change" id="nextconfirm">Open the confirm box</button>`;
  if (n.action === "setup") a = `<button class="ghost" id="nextsetup">Open Setup</button>`;
  if (S.demo && step && step.kind === "read") a = `<div class="sub">Demo mode: commands that talk to the car are switched off. Try the Analysis tab on the synthetic lap.</div>`;
  $("nextact").innerHTML = a;
  if ($("nextrun")) $("nextrun").onclick = () => run(n.step);
  if ($("nextplan")) $("nextplan").onclick = () => show("plan");
  if ($("nextconfirm")) $("nextconfirm").onclick = () => { modalJob = null; renderModal(); };
  document.querySelector(".card.next").classList.toggle("hold", !!S.hold);
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
    if (step && step.kind === "car_change") btn = `<div style="margin-top:6px"><button class="change" id="ansplan">Go to: ${esc(step.title)}</button></div>`;
    const src = {local: "Answered by the local model on this laptop", claude: "Answered by Claude", guide: "Built-in guide"}[r.source] || r.source;
    $("answer").innerHTML = `<div class="msg info">${esc(r.answer)}<div class="sub" style="margin-top:6px">${src}${r.note ? " · " + esc(r.note) : ""}</div>${btn}</div>`;
    if ($("ansplan")) $("ansplan").onclick = () => show("plan");
    if ($("ansrun")) $("ansrun").onclick = () => run(step.id);
  } catch (e){ $("answer").innerHTML = `<div class="msg err">${esc(e.message)}</div>`; }
}
$("ask").onclick = () => ask($("q").value); $("asknext").onclick = () => ask("What should I do next?");
let stopArmed = null;
$("stopbtn").onclick = async () => {
  const b = $("stopbtn");
  if (!stopArmed){ b.textContent = "Click again to stop"; stopArmed = setTimeout(() => { stopArmed = null; b.textContent = "Stop the running command"; }, 4000); return; }
  clearTimeout(stopArmed); stopArmed = null; b.textContent = "Stop the running command";
  await api("/api/stop", {}); tick();
};

// ================= map =================
const LAYERS = {}; document.querySelectorAll("[data-layer]").forEach(i => { LAYERS[i.dataset.layer] = i.checked; i.onchange = () => { LAYERS[i.dataset.layer] = i.checked; drawMap(); }; });
let M = null, mapBusy = false, mapAt = 0, SP = null, spAt = 0, spBusy = false;
const VIRIDIS = ["#440154","#482878","#3e4989","#31688e","#26828e","#1f9e89","#35b779","#6ece58","#b5de2b","#fde725"];
const GREYS = ["#f2f4f6","#d5dbe1","#b3bdc7","#8d9aa7","#687786","#475566"];
const VIEWS = {
  best: {label: "Signal strength (best AP)", unit: "dBm", lo: -90, hi: -35, ramp: VIRIDIS, key: "best",
         help: "The strongest AP's signal at each spot: what the car's radio would hear there."},
  bestap: {label: "Best AP (who serves where)", kind: "ap", help: "Which AP is strongest at each spot."},
  backup: {label: "Backup signal (second AP location)", unit: "dBm", lo: -90, hi: -35, ramp: VIRIDIS, key: "backup",
           help: "The strongest AP at a different place from the best one: the second path WirelessJEV relies on."},
  snr: {label: "SNR", unit: "dB", lo: 0, hi: 50, ramp: VIRIDIS, key: "snr", help: "Best signal minus the site sheet's noise floor."},
  sinr: {label: "SINR (with co-channel APs)", unit: "dB", lo: -5, hi: 45, ramp: VIRIDIS, key: "sinr",
         help: "Best signal against noise plus every other AP on the same channel."},
  cochan: {label: "Co-channel APs heard", kind: "count", key: "cochan", help: "Other APs on the serving AP's channel heard at or above −82 dBm: they share airtime."},
  uncertainty: {label: "Uncertainty (RadioMap)", unit: "± dB", lo: 0, hi: 8, ramp: GREYS, key: "uncertainty",
                help: "RadioMap's own estimate of its error. It can be optimistic; use it with the fade."},
  measured: {label: "Measured readings only", kind: "measured", help: "Only what the car measured on the chosen lap, averaged per cell. No model."},
};
function viewOptions(){
  const sel = $("mapview"), cur = sel.value || "best";
  let opts = Object.entries(VIEWS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join("");
  if (SP) opts += SP.aps.map((a, i) => `<option value="ap${i}">Signal: AP${i + 1} ${esc(a.bssid)} (ch ${a.channel})</option>`).join("");
  if (sel.dataset.n !== String(SP ? SP.aps.length : 0)){ sel.innerHTML = opts; sel.dataset.n = String(SP ? SP.aps.length : 0); sel.value = cur; if (!sel.value) sel.value = "best"; }
}
$("mapview") && ($("mapview").onchange = () => drawMap());
async function refreshSpectrum(force){
  if (spBusy || (!force && Date.now() - spAt < 10000)) return;
  spBusy = true; spAt = Date.now();
  try { SP = await api("/api/spectrum"); viewOptions(); drawMap(); }
  catch (e){ SP = null; viewOptions(); $("mapnote").textContent = e.message; }
  finally { spBusy = false; }
}
function hexrgb(h){ h = h.trim(); if (h.startsWith("#")) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  const m = h.match(/\d+/g); return m ? m.slice(0, 3).map(Number) : [128, 128, 128]; }
function rampAt(ramp, f){ f = Math.max(0, Math.min(1, f)); const x = f * (ramp.length - 1), i = Math.min(ramp.length - 2, Math.floor(x)), t = x - i;
  const a = hexrgb(ramp[i]), b = hexrgb(ramp[i + 1]); return a.map((v, k) => Math.round(v + t * (b[k] - v))); }
function raster(view){
  const g = SP.grid, L = SP.layers, n = g.nx * g.ny, cv = document.createElement("canvas"); cv.width = g.nx; cv.height = g.ny;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(g.nx, g.ny), V = VIEWS[view];
  const apcols = SP.aps.map((a, i) => hexrgb(css("--ap" + (i % 5 + 1))));
  const vals = view.startsWith("ap") ? L.per_ap[+view.slice(2)] : V && V.key ? L[V.key] : null;
  const lo = view.startsWith("ap") ? -90 : V.lo, hi = view.startsWith("ap") ? -35 : V.hi, ramp = view.startsWith("ap") ? VIRIDIS : V.ramp;
  for (let i = 0; i < n; i++){
    let rgb = null, a = 215;
    if (view === "bestap") { rgb = apcols[L.best_ap[i]]; a = 170; }
    else if (view === "cochan") { const c = L.cochan[i]; if (c > 0) rgb = hexrgb(c === 1 ? css("--warn") : css("--bad")); a = c > 0 ? 190 : 0; }
    else { const v = vals[i]; if (v != null && !(view !== "uncertainty" && v < lo - 8)) rgb = rampAt(ramp, (v - lo) / (hi - lo)); }
    if (!rgb) { img.data[i * 4 + 3] = 0; continue; }
    if (LAYERS.fade) { const d = L.near_m[i]; a = Math.round(a * (d <= 3 ? 1 : Math.max(0.28, 1 - (d - 3) / 7))); }
    img.data.set([rgb[0], rgb[1], rgb[2], a], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  return cv.toDataURL("image/png");
}
function drawScale(view){
  const V = VIEWS[view] || {label: "Signal", unit: "dBm", lo: -90, hi: -35, ramp: VIRIDIS};
  let h = "";
  if (view === "bestap" && SP) h = `<div class="legend">${SP.aps.map((a, i) => `<span><span class="sw" style="background:var(--ap${i % 5 + 1})"></span>AP${i + 1} ${esc(a.bssid)} · ch ${a.channel}</span>`).join("")}</div>`;
  else if (view === "cochan") h = `<div class="legend"><span><span class="sw" style="background:var(--warn)"></span>1 other AP on the channel</span><span><span class="sw" style="background:var(--bad)"></span>2 or more</span><span>clear: none</span></div>`;
  else if (view !== "measured") {
    const grad = `linear-gradient(90deg, ${V.ramp.join(",")})`;
    h = `<div class="scale"><span>${V.lo} ${V.unit}</span><div class="bar" style="background:${grad}"></div><span>${V.hi} ${V.unit}</span></div>`;
  }
  const help = (VIEWS[view] || {}).help || "This AP's signal at each spot.";
  $("mapscale").innerHTML = h + `<div class="sub" style="margin-top:4px">${esc(help)}</div>`;
}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function apColor(b, list){ const i = list.indexOf((b || "").toLowerCase()); return i < 0 ? css("--muted") : css("--ap" + (i % 5 + 1)); }
function sigColor(v, th){ return v >= th.good_dbm ? css("--good") : v >= th.fair_dbm ? css("--warn") : css("--bad"); }
async function loadLaps(){
  try { const r = await fetch("/laps").then(r => r.json()); const laps = r.laps || r;
    const sel = $("maplap"), cur = sel.value || "newest";
    sel.innerHTML = '<option value="newest">Newest lap (live)</option><option value="all">All laps: the radio map so far</option>' +
      laps.map(l => `<option value="${esc(l.file)}">${esc(l.name)} · ${new Date(l.modified * 1000).toLocaleTimeString()}</option>`).join("");
    sel.value = [...sel.options].some(o => o.value === cur) ? cur : "newest";
  } catch (e){}
}
$("maplap").onchange = () => refreshMap(true);
async function refreshMap(force){
  if (mapBusy) return; const sel = $("maplap").value || "newest";
  const live = sel === "newest";
  if (!force && Date.now() - mapAt < (live ? 1800 : 10000)) return;
  mapBusy = true; mapAt = Date.now();
  try { M = await api("/api/map?lap=" + encodeURIComponent(sel)); drawMap(); drawHealth(); drawTimeline(); drawAps(); drawEvents(); }
  catch (e){ $("mapnote").textContent = e.message; }
  finally { mapBusy = false; }
}
function drawMap(){
  if (!M) return;
  const dims = (M.site && M.site.dimensions_m) || null, svg = $("map");
  if (!dims){ svg.innerHTML = ""; $("mapnote").textContent = "Save the site sheet (Setup tab) to draw the map: its size, APs and waypoints come from it."; return; }
  const [W, H] = dims, pad = 1.2; svg.setAttribute("viewBox", `${-pad} ${-pad} ${W + 2 * pad} ${H + 2 * pad}`);
  const apList = (M.site.aps || []).map(a => a.bssid.toLowerCase());
  const th = M.thresholds || {good_dbm: -67, fair_dbm: -75}; let g = "";
  g += `<rect x="0" y="0" width="${W}" height="${H}" fill="none" stroke="${css("--line")}" stroke-width="0.08"/>`;
  for (let x = 0; x <= W; x += 2) g += `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="${css("--line")}" stroke-width="0.02"/>`;
  for (let y = 0; y <= H; y += 2) g += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="${css("--line")}" stroke-width="0.02"/>`;
  const view = $("mapview").value || "best";
  if (SP && view !== "measured" && SP.grid.width === W) {
    g += `<image href="${raster(view)}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="none" style="image-rendering:auto"/>`;
    if (LAYERS.contours && (view === "best" || view === "bestap")) {
      for (const [name, col] of [["good", css("--ink")], ["fair", css("--muted")]])
        for (const [x1, y1, x2, y2] of SP.contours[name]) g += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="0.07" ${name === "fair" ? 'stroke-dasharray="0.25 0.18"' : ""}/>`;
    }
  }
  if (LAYERS.boxes && SP) for (const [lo, hi] of SP.boxes)
    g += `<rect x="${Math.min(lo[0], hi[0])}" y="${Math.min(lo[1], hi[1])}" width="${Math.abs(hi[0] - lo[0])}" height="${Math.abs(hi[1] - lo[1])}" fill="${css("--ink")}" fill-opacity="0.55"><title>Obstruction from the site sheet</title></rect>`;
  drawScale(view);
  if (view === "measured") for (const [x, y, v, n] of (M.cells || [])) {
    const c = M.cell_m; g += `<rect x="${x - c / 2}" y="${y - c / 2}" width="${c}" height="${c}" fill="${sigColor(v, th)}" fill-opacity="0.45"><title>${v} dBm mean of ${n} readings</title></rect>`; }
  if (LAYERS.dots) for (const [x, y, v, b, t] of (M.points || [])) {
    const col = sigColor(v, th);
    g += `<circle cx="${x}" cy="${y}" r="0.13" fill="${col}"><title>${v} dBm via ${esc(b)} at ${t} s</title></circle>`; }
  const wps = M.waypoints || {}, names = Object.keys(wps);
  if (LAYERS.route && names.length > 1) g += `<polyline points="${names.concat(names[0]).map(k => wps[k].join(",")).join(" ")}" fill="none" stroke="${css("--muted")}" stroke-width="0.07" stroke-dasharray="0.4 0.3"/>`;
  for (const k of names) { const [x, y] = wps[k];
    g += `<rect x="${x - 0.28}" y="${y - 0.28}" width="0.56" height="0.56" fill="${css("--panel")}" stroke="${css("--ink")}" stroke-width="0.06"/><text x="${x}" y="${y + 0.2}" font-size="0.55" text-anchor="middle" fill="${css("--ink")}" font-weight="700">${esc(k)}</text>`; }
  (M.site.aps || []).forEach((a, i) => { const [x, y] = a.xyz_m, col = css("--ap" + (i % 5 + 1));
    g += `<circle cx="${x}" cy="${y}" r="0.5" fill="${col}" stroke="${css("--panel")}" stroke-width="0.12"><title>${esc(a.bssid)} · ${a.frequency_mhz} MHz</title></circle>
      <text x="${x}" y="${y - 0.75}" font-size="0.6" text-anchor="middle" fill="${css("--ink")}">AP${i + 1} · ch ${chan(a.frequency_mhz)}</text>`; });
  if (LAYERS.roams) for (const r of (M.roams || [])) if (r.x != null) {
    const s = 0.38, col = r.bad ? css("--bad") : css("--ink");
    g += `<path d="M${r.x} ${r.y - s} L${r.x + s} ${r.y} L${r.x} ${r.y + s} L${r.x - s} ${r.y} Z" fill="${css("--panel")}" stroke="${col}" stroke-width="0.09"><title>${r.bad ? "Bad roam" : "Roam"}: ${esc(r.from)} → ${esc(r.to)}, reply gap ${r.reply_gap_s} s${r.lap ? " (" + esc(r.lap) + ")" : ""}</title></path>`; }
  if (LAYERS.outages) for (const o of (M.outages || [])) if (o.x != null)
    g += `<circle cx="${o.x}" cy="${o.y}" r="0.55" fill="none" stroke="${css("--bad")}" stroke-width="0.12"><title>No gateway replies for ${o.seconds} s${o.lap ? " (" + esc(o.lap) + ")" : ""}</title></circle>`;
  if (LAYERS.car && M.car) { const c = M.car, col = css("--accent");
    g += `<circle cx="${c.x}" cy="${c.y}" r="0.95" fill="${col}" fill-opacity="0.15"/><circle cx="${c.x}" cy="${c.y}" r="0.42" fill="${c.estimated ? css("--panel") : col}" stroke="${col}" stroke-width="0.14" ${c.estimated ? 'stroke-dasharray="0.25 0.15"' : ""}><title>SmartCar${c.estimated ? " (estimated after mark " + esc(c.last_mark) + ")" : ""}: ${c.signal_dbm} dBm via ${esc(c.bssid)}</title></circle>`; }
  svg.innerHTML = g;
  const apLeg = (M.site.aps || []).map((a, i) => `<span><span class="sw" style="background:var(--ap${i % 5 + 1})"></span>AP${i + 1} ${esc(a.bssid)}</span>`).join("");
  $("maplegend").innerHTML = `<span>── Good ${th.good_dbm} dBm line</span><span>- - Fair ${th.fair_dbm} dBm line</span><span><span class="sw" style="background:var(--ink);opacity:.55"></span>obstruction</span>
    <span>◇ roam (red: bad)</span><span style="color:var(--bad)">○</span><span>outage</span><span style="color:var(--accent)">●</span><span>car (dashed: estimated)</span>${apLeg}`;
  let note = "";
  if (SP) { const f = SP.fit; note += f.readings ? `Radio map: RadioMap fitted to ${f.readings.toLocaleString()} positioned readings from ${f.laps_used} lap(s) across ${f.aps_measured} AP bands, over the whole floor. Faded: more than 3 m from any reading, so mostly the model's guess. ` : "Radio map: no positioned readings yet, so this is the site sheet's physics only (distance, power, obstructions). "; }
  if (M.mode === "all") note += `${M.laps} laps merged: ${M.readings} positioned readings${M.readings > (M.points || []).length ? ` (${(M.points || []).length} drawn as dots; the heatmap uses all)` : ""}.`;
  else if (M.empty) note += "No lap recorded yet.";
  else if (M.lap) note += `Lap ${esc(M.lap.name)}${M.lap.finished ? "" : " (recording)"} · ${M.lap.marks} marks · ${M.lap.samples} samples. Positions are interpolated between waypoint marks; readings before the first or after the last mark have none.`;
  $("mapnote").innerHTML = note;
}
$("map").addEventListener("mousemove", e => {
  const tip = $("maptip"); if (!SP || !M) { tip.classList.add("hide"); return; }
  const svg = $("map"), pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse()), g = SP.grid;
  const i = Math.floor(p.x / g.cell), j = Math.floor(p.y / g.cell);
  if (i < 0 || j < 0 || i >= g.nx || j >= g.ny) { tip.classList.add("hide"); return; }
  const k = j * g.nx + i, L = SP.layers, ap = L.best_ap[k];
  tip.innerHTML = `<b>${p.x.toFixed(1)}, ${p.y.toFixed(1)} m</b><br>Best AP${ap + 1}: ${L.best[k]} dBm · backup ${L.backup[k] ?? "–"} dBm<br>SNR ${L.snr[k]} dB · SINR ${L.sinr[k]} dB · co-channel ${L.cochan[k]}<br>± ${L.uncertainty[k]} dB · nearest reading ${L.near_m[k] >= 99 ? "none" : L.near_m[k] + " m"}`;
  const box = svg.getBoundingClientRect(); tip.style.left = Math.min(box.width - 230, e.clientX - box.left + 14) + "px"; tip.style.top = (e.clientY - box.top + 14) + "px"; tip.classList.remove("hide");
});
$("map").addEventListener("mouseleave", () => $("maptip").classList.add("hide"));
function chan(f){ f = +f; if (!f) return "?"; if (f < 3000) return f === 2484 ? 14 : Math.round((f - 2407) / 5); if (f < 5925) return Math.round((f - 5000) / 5); return Math.round((f - 5950) / 5); }
function pct(v){ return v == null ? "–" : Math.round(v * 100) + "%"; }
function drawHealth(){
  const h = M && M.health; if (!h){ $("tiles").innerHTML = '<div class="sub">Health is shown for one lap; pick a lap above.</div>'; $("healthdef").textContent = ""; return; }
  const t = (l, v, sub) => `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="l">${sub || ""}</div></div>`;
  $("tiles").innerHTML = t("Score", h.score ?? "–", "out of 100") + t("Good signal", pct(h.signal_good_share), `median ${h.signal_median_dbm ?? "–"} dBm`) +
    t("Gateway replies", h.reply_share == null ? "–" : (h.reply_share * 100).toFixed(1) + "%", `${h.outages} outages · longest ${h.longest_outage_s} s`) +
    t("Latency", h.latency_median_ms == null ? "–" : h.latency_median_ms + " ms", `95th percentile ${h.latency_p95_ms ?? "–"} ms`) +
    t("Roams", h.roams, `${h.bad_roams} bad · mean gap ${h.mean_roam_gap_s ?? "–"} s`) + t("Positioned", pct(h.positioned_share), "readings with a map position");
  $("healthdef").textContent = h.definition;
}
function drawTimeline(){
  const svg = $("tl"); if (!M || !M.timeline){ svg.innerHTML = ""; $("tllegend").textContent = "The timeline is shown for one lap; pick a lap above."; return; }
  const tl = M.timeline, T = Math.max(1, ...tl.map(r => r[0] || 0), ...(M.rtt || []).map(r => r[0] || 0));
  const X = t => (t / T) * 1000, Y = v => 150 - ((Math.max(-95, Math.min(-30, v)) + 95) / 65) * 150;
  const apList = (M.site.aps || []).map(a => a.bssid.toLowerCase()); const th = M.thresholds; let g = "";
  for (const v of [th.good_dbm, th.fair_dbm]) g += `<line x1="0" x2="1000" y1="${Y(v)}" y2="${Y(v)}" stroke="${css("--muted")}" stroke-dasharray="6 5" vector-effect="non-scaling-stroke" stroke-width="1"/>`;
  for (const o of (M.outages || [])) g += `<rect x="${X(o.t_s)}" y="0" width="${Math.max(2, X(o.seconds))}" height="150" fill="${css("--bad")}" fill-opacity="0.25"><title>Outage ${o.seconds} s at ${o.t_s} s</title></rect>`;
  let seg = [], last = null; const flush = () => { if (seg.length > 1) g += `<polyline points="${seg.join(" ")}" fill="none" stroke="${apColor(last, apList)}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`; };
  for (const [t, v, b] of tl) { if (v == null) continue; if (b !== last){ flush(); seg = seg.length ? [seg[seg.length - 1]] : []; last = b; } seg.push(`${X(t).toFixed(1)},${Y(v).toFixed(1)}`); }
  flush();
  for (const r of (M.roams || [])) g += `<line x1="${X(r.t_s)}" x2="${X(r.t_s)}" y1="0" y2="150" stroke="${r.bad ? css("--bad") : css("--ink")}" stroke-width="1" vector-effect="non-scaling-stroke"><title>Roam at ${r.t_s} s</title></line>`;
  for (const m of (M.marks || [])) g += `<line x1="${X(m.t_s)}" x2="${X(m.t_s)}" y1="0" y2="10" stroke="${css("--accent")}" stroke-width="2" vector-effect="non-scaling-stroke"><title>Mark ${esc(m.label)} at ${m.t_s} s</title></line>`;
  svg.innerHTML = g;
  $("tllegend").innerHTML = `<span>0 s</span><span>→ ${Math.round(T)} s</span><span>Line: signal (−95 to −30 dBm), coloured by serving AP</span><span><span class="sw" style="background:var(--bad);opacity:.4"></span>outage</span><span>| roam</span><span style="color:var(--accent)">ˈ</span><span>waypoint mark</span><span>dashed: Good and Fair</span>`;
}
function drawAps(){
  const aps = (M && M.aps) || []; const apList = ((M && M.site && M.site.aps) || []).map(a => a.bssid.toLowerCase());
  if (!aps.length){ $("aptable").innerHTML = '<tr><td class="sub">Per-AP figures appear for one lap.</td></tr>'; return; }
  $("aptable").innerHTML = '<tr><th>AP</th><th>On sheet</th><th>Channel</th><th>Serving</th><th>Signal while serving</th><th>Roams in / out</th><th>Last scan</th></tr>' +
    aps.map(a => `<tr><td><span class="sw" style="background:${apColor(a.bssid, apList)}"></span><span class="mono">${esc(a.bssid)}</span></td><td>${a.in_sheet ? "yes" : '<span class="warnc">no</span>'}</td><td>${a.channel ?? "?"} · ${esc(a.band || "")}</td>
      <td>${pct(a.serving_share)} · ${a.serving_s} s</td><td>${a.mean_dbm ?? "–"} dBm (min ${a.min_dbm ?? "–"})</td><td>${a.roams_in} / ${a.roams_out}</td><td>${a.seen_in_scan ? a.scan_dbm + " dBm" : '<span class="sub">not heard</span>'}</td></tr>`).join("");
}
function drawEvents(){
  const ev = (M && M.events) || [];
  $("events").innerHTML = ev.length ? '<tr><th>Time</th><th>Event</th></tr>' + ev.slice().reverse().map(e => `<tr><td class="mono">${e.t_s} s</td><td><span class="kind ${e.kind === "outage" || e.kind === "bad_roam" ? "car_change" : ""}">${esc(e.kind.replace("_", " "))}</span> ${esc(e.text)}</td></tr>`).join("")
    : '<tr><td class="sub">Events appear for one lap: marks, roams and outages.</td></tr>';
}
// ================= routers =================
async function loadRouters(){
  try { const r = await api("/api/routers");
    $("checklist").innerHTML = r.checklist.map(c => `<label class="chk"><input type="checkbox" data-chk="${c.id}" ${c.done ? "checked" : ""}><span>${esc(c.text)}</span></label>`).join("");
    document.querySelectorAll("[data-chk]").forEach(i => i.onchange = async () => { try { await api("/api/checklist", {item: i.dataset.chk, done: i.checked}); tick(); } catch (e){ toast(e.message); } });
    $("rcsrc").textContent = r.scan ? `Newest scan: ${r.scan.source}, ${r.scan.aps} BSSIDs heard.` : "No scan yet.";
    $("rcrows").innerHTML = r.rows.map(x => `<tr><td><span class="pill ${x.status === "PASS" ? "done" : x.status === "WARN" ? "part" : "todo"}" style="${x.status === "FAIL" ? "background:var(--bad);color:#fff;border:none" : ""}">${x.status}</span></td><td>${esc(x.item)}<div class="sub">${esc(x.seen)}</div></td><td class="sub">${esc(x.fix)}</td></tr>`).join("");
  } catch (e){ toast(e.message); }
}
$("rcrefresh").onclick = loadRouters;
// ================= settings =================
let SET = [];
async function loadSettings(){ try { SET = (await api("/api/config")).settings; drawSettings(); } catch (e){ toast(e.message); } }
function input(p){
  const v = p.value, id = `set-${p.id}`;
  if (p.kind === "bool") return `<select id="${id}" data-set="${p.id}"><option value="true" ${v ? "selected" : ""}>yes</option><option value="false" ${!v ? "selected" : ""}>no</option></select>`;
  if (p.kind === "choice") return `<select id="${id}" data-set="${p.id}">${p.choices.map(c => `<option ${String(c) === String(v) ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
  if (p.kind === "int" || p.kind === "float") return `<input id="${id}" data-set="${p.id}" type="number" value="${v}" min="${p.lo ?? ""}" max="${p.hi ?? ""}" step="${p.kind === "int" ? 1 : "any"}">`;
  return `<input id="${id}" data-set="${p.id}" value="${esc(v)}" autocomplete="off" spellcheck="false">`;
}
function drawSettings(){
  const groups = [...new Set(SET.map(p => p.group))];
  $("setgroups").innerHTML = groups.map(g => `<div class="setgroup"><h3>${esc(g)}</h3><div class="scroll"><table>
    <tr><th>Setting</th><th>Value</th><th>Default</th><th>Used by</th><th></th></tr>
    ${SET.filter(p => p.group === g).map(p => `<tr class="${p.changed ? "chg" : ""}"><td><b>${esc(p.label)}</b><div class="sub">${esc(p.why)}</div></td><td>${input(p)}${p.unit ? `<div class="sub">${esc(p.unit)}${p.lo != null ? ` · ${p.lo}–${p.hi}` : ""}</div>` : ""}</td>
      <td class="mono">${esc(p.default === "" ? "(empty)" : p.default)}</td><td class="sub">${esc(p.used_by)}</td><td>${p.changed ? `<button class="ghost" data-reset="${p.id}">Reset</button>` : ""}</td></tr>`).join("")}</table></div></div>`).join("");
  document.querySelectorAll("[data-reset]").forEach(b => b.onclick = async () => { try { const r = await api("/api/config/reset", {ids: [b.dataset.reset]}); SET = r.settings; drawSettings(); $("setmsg").textContent = "Reset."; tick(); } catch (e){ toast(e.message); } });
}
$("setsave").onclick = async () => {
  const changes = {};
  document.querySelectorAll("[data-set]").forEach(i => { const p = SET.find(x => x.id === i.dataset.set); if (String(i.value) !== String(p.value)) changes[p.id] = p.kind === "bool" ? i.value === "true" : i.value; });
  if (!Object.keys(changes).length){ $("setmsg").textContent = "Nothing changed."; return; }
  try { const r = await api("/api/config", {changes}); SET = r.settings; drawSettings(); $("setmsg").innerHTML = `<span class="ok">Saved ${r.changed.length} change(s); they apply to the next command.</span>`; mapAt = 0; spAt = 0; tick(); }
  catch (e){ $("setmsg").innerHTML = `<span class="bad">Not saved: ${esc(e.message)}</span>`; }
};
$("setundo").onclick = () => { drawSettings(); $("setmsg").textContent = "Unsaved edits undone."; };
$("ghostcopy").onclick = e => copy($("ghostcmd").textContent, e.target);
// ================= activity =================
async function loadActivity(){
  try { const a = (await api("/api/activity")).activity;
    $("activity").innerHTML = '<tr><th>Time</th><th>Action</th><th>Detail</th></tr>' + a.map(r => { const {t, action, ...d} = r;
      return `<tr><td class="mono">${new Date(t * 1000).toLocaleTimeString()}</td><td><b>${esc(action.replace(/_/g, " "))}</b></td><td class="mono" style="overflow-wrap:anywhere">${esc(Object.entries(d).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  "))}</td></tr>`; }).join("");
  } catch (e){}
}
// ================= help and the setup banner =================
let HELP = null;
async function loadHelp(){
  try { HELP = HELP || await api("/api/help");
    const asst = S && S.environment.find(r => r.item === "Assistant");
    $("before").innerHTML = HELP.before.map((b, i) => { const done = (i === 0) || (i === 1 && S && S.environment[0].ok && S.environment[1].ok) || (i === 2 && asst && asst.ok && !/guide/.test(asst.seen.split(";")[0])) || (i === 3 && S && (S.environment.find(r => r.item.startsWith("WirelessJEV")) || {}).ok);
      return `<div class="step"><div class="n ${done ? "ok" : ""}">${done ? "✓" : i + 1}</div><div><b>${esc(b.title)}</b><div class="sub">${esc(b.text)}</div>${b.commands ? `<code class="cmd">${esc(b.commands)}</code>` : ""}${i === 2 && asst ? `<div class="sub" style="margin-top:4px">Now: ${esc(asst.seen)}</div>` : ""}</div></div>`; }).join("");
    drawTopics();
    $("news").innerHTML = HELP.whats_new.map(n => `<div><b>${esc(n.date)}</b><ul style="margin:6px 0 0 18px;padding:0">${n.items.map(x => `<li style="margin:3px 0">${esc(x)}</li>`).join("")}</ul></div>`).join("");
  } catch (e){ toast(e.message); }
}
function drawTopics(){
  if (!HELP) return; const q = $("helpq").value.toLowerCase().trim();
  const hits = HELP.topics.filter(t => !q || (t.title + " " + t.text + " " + t.keys).toLowerCase().includes(q));
  $("topics").innerHTML = hits.length ? hits.map(t => `<details ${q ? "open" : ""} style="padding:6px 0;border-bottom:1px solid var(--line)"><summary><b>${esc(t.title)}</b></summary><div class="sub" style="margin-top:4px">${esc(t.text)}</div></details>`).join("") : '<div class="sub">No topic matches; ask the assistant instead.</div>';
}
$("helpq").oninput = drawTopics;
let bannerHidden = false; try { bannerHidden = sessionStorage.getItem("wj-banner") === "hidden"; } catch (e) {}
function renderBanner(){
  const asst = S.environment.find(r => r.item === "Assistant");
  const needs = asst && (!asst.ok || /built-in guide/.test(asst.seen.split(";")[0]));
  const missingFiles = !S.environment[0].ok || !S.environment[1].ok;
  const msg = missingFiles ? "The car's files are not on this laptop yet." : needs ? "The assistant has no local model, so on site (no internet) it can only use the built-in guide. Set one up before the site day; it needs internet." : "";
  $("bannertext").textContent = msg;
  $("banner").classList.toggle("hide", !msg || bannerHidden);
}
$("bannergo").onclick = () => show("help");
$("bannerx").onclick = () => { bannerHidden = true; try { sessionStorage.setItem("wj-banner", "hidden"); } catch (e) {} $("banner").classList.add("hide"); };
let slowAt = 0;
let busy = false;
async function tick(){
  if (busy) return; busy = true;
  try { S = await api("/api/state"); renderSteps(); renderEnv(); renderNext(); renderCheck(); renderMarks(); renderAnalysis(); renderJobs(); loadSite(); renderModal(); renderBanner();
    $("ghostcmd").textContent = S.ghost_command;
    if (visible("map")) { refreshMap(false); refreshSpectrum(false); }
    if (Date.now() - slowAt > 8000){ slowAt = Date.now(); if (visible("activity")) loadActivity(); if (visible("setup")) loadRouters(); if (visible("map")) loadLaps(); }
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
