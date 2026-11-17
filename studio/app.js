// WirelessJEV Studio: simulation evidence only.
// Every number on screen comes from data/evidence.json, written by studio/build_bundle.py
// from the committed result files in artifacts/native/ and the campaign code. Nothing is typed in here.

const view = document.getElementById("view");
let B = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (x, d) => {
  if (typeof x !== "number" || !isFinite(x)) return "—";
  if (d === undefined) d = x === 0 ? 0 : Math.abs(x) >= 100 ? 0 : 1;
  return x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
};
const pct = (x, d = 1) => (typeof x === "number" ? (100 * x).toFixed(d) + "%" : "—");
const slug = (id) => "c-" + id.replace(/_/g, "-");
const FAM = { "WirelessJEV": "--ours", "5G model": "--nr", "Wi-Fi 8 model": "--w8", "Wi-Fi 7 baseline": "--base", "Wi-Fi baseline": "--base" };
const famColor = (f) => `var(${FAM[f] || "--base"})`;
const src = (p) => (p ? `<span class="src">${esc(p)}</span>` : "");
const statusChip = (s) => (s === "development" ? `<span class="chip dev" title="Development scenes: used to build and tune, not a pre-registered fresh test">development scenes</span>` : `<span class="chip fresh" title="Pre-registered test on scenes never used in development">fresh scenes</span>`);

// ---------------------------------------------------------------- log-axis charts
function logScale(maxV) {
  const top = Math.max(maxV, 8) * 1.35;
  const x = (v) => (v <= 1 ? 0 : (100 * Math.log10(v)) / Math.log10(top));
  const ticks = [];
  for (let t = 1; t <= top; t *= 10) ticks.push(t);
  return { x, ticks, top };
}
function axis(sc, caption) {
  const lab = (t) => (t >= 1000 ? (t >= 1e6 ? t / 1e6 + "M" : t / 1000 + "k") : String(t));
  return `<div class="axis"><span></span><div class="ticks">${sc.ticks.map((t) => `<span style="left:${sc.x(t)}%">${lab(t)}</span>`).join("")}</div><span></span>
    <div class="cap">${esc(caption)}</div></div>`;
}
const gridLines = (sc) => sc.ticks.map((t) => `<span class="grid" style="left:${sc.x(t)}%"></span>`).join("");

function barChart(rows, caption = "missed control packets per million · log scale · whiskers: 95% interval over scenes") {
  const maxV = Math.max(...rows.map((r) => Math.max(r.mpm ?? 0, r.ci ? r.ci[1] : 0)));
  const sc = logScale(maxV);
  const body = rows.map((r) => {
    const w = sc.x(r.mpm);
    const ci = r.ci ? `<span class="ci" style="left:${sc.x(r.ci[0])}%;width:${Math.max(sc.x(r.ci[1]) - sc.x(r.ci[0]), 0.4)}%"></span>` : "";
    const bar = r.mpm > 0 ? `<span class="bar" style="width:${Math.max(w, 0.6)}%;background:${famColor(r.family)}"></span>` : `<span class="zero">0</span>`;
    return `<div class="row"><div class="lab">${esc(r.label)}<small>${esc(r.id)}</small></div>
      <div class="track" role="img" aria-label="${esc(r.label)}: ${num(r.mpm)} per million">${gridLines(sc)}${bar}${ci}</div>
      <div class="val">${num(r.mpm)}</div></div>`;
  }).join("");
  return `<div class="chart">${body}${axis(sc, caption)}</div>`;
}
function legend(fams) {
  return `<div class="legend">${fams.map((f) => `<span><i style="background:${famColor(f)}"></i>${esc(f)}</span>`).join("")}</div>`;
}

// ---------------------------------------------------------------- overview
function overview() {
  const C = B.campaigns;
  const hyps = C.flatMap((c) => c.hypotheses);
  const pass = hyps.filter((h) => h.pass).length;
  const fresh = C.filter((c) => c.status === "fresh");
  const scenes = C.reduce((a, c) => a + (c.setup.n_scenes || 0), 0);
  const L = B.learned;
  let s = `<div class="eyebrow">Proof of record</div><h1>Every canonical simulation, with its setup and its verdicts</h1>
    <p class="lede">${C.length} campaigns on the 100-robot warehouse simulator, each with pre-registered hypotheses scored on scenes it was not tuned on, plus ${L.ours.length + L.codex.length} learned-model studies. Losses are kept next to wins.</p>
    <div class="scope"><b>Scope</b><span>${esc(B.scope)}</span></div>
    <div class="tiles">
      <div class="card tile"><div class="k">Campaigns</div><div class="v">${C.length}</div><div class="n">${fresh.length} on fresh scenes, ${C.length - fresh.length} on development scenes</div></div>
      <div class="card tile"><div class="k">Hypotheses scored</div><div class="v">${pass}<span style="color:var(--faint)">/${hyps.length}</span></div><div class="n">passed; ${hyps.length - pass} failed and kept</div></div>
      <div class="card tile"><div class="k">Scenes simulated</div><div class="v">${scenes}</div><div class="n">100 robots each, 120–300 s</div></div>
      <div class="card tile"><div class="k">Learned-model studies</div><div class="v">${L.ours.length + L.codex.length}</div><div class="n">${L.ours.length} ours · ${L.codex.length} Codex · none promoted</div></div>
    </div>`;

  // headline
  const H = B.headline;
  const vals = H.flatMap((h) => [h.ours, h.nr, h.wifi8].filter(Boolean).flatMap((v) => [v.mpm, v.ci ? v.ci[1] : 0]));
  const sc = logScale(Math.max(...vals));
  const dot = (v, fam, name) => {
    if (!v) return "";
    const ci = v.ci ? `<span class="ci" style="left:${sc.x(v.ci[0])}%;width:${Math.max(sc.x(v.ci[1]) - sc.x(v.ci[0]), 0.4)}%;background:${famColor(fam)};opacity:.6"></span>` : "";
    return `${ci}<span class="dot" title="${esc(name)}: ${num(v.mpm)}" style="left:${sc.x(v.mpm)}%;background:${famColor(fam)}"></span>`;
  };
  s += `<h2>As the setup gets harsher: our stack against the 5G and Wi-Fi 8 models</h2>
    <p class="lede">Same scenes for every arm in a row. Each row names the campaign and its scenes. Ours is the WirelessJEV stack with the second path named in the row; the 5G and Wi-Fi 8 arms are generous models of each.</p>
    <div class="card hlchart">${legend(["WirelessJEV", "5G model", "Wi-Fi 8 model"])}<div class="chart">`;
  for (const h of H) {
    s += `<div class="stage"><div class="row"><div class="lab"><b>${esc(h.stage)}</b><small>${esc(h.ours_label)} · <a href="#${slug(h.campaign)}">${esc(h.campaign)}</a> · ${esc(h.condition)} · scenes ${esc(h.scenes)}</small></div>
      <div class="track">${gridLines(sc)}${dot(h.wifi8, "Wi-Fi 8 model", "Wi-Fi 8")}${dot(h.nr, "5G model", "5G")}${dot(h.ours, "WirelessJEV", "Ours")}</div>
      <div class="val" style="color:var(--ours)">${num(h.ours?.mpm)}</div></div>
      <div class="row" style="padding:0"><span></span><div class="src" style="font-size:.72rem">5G ${num(h.nr?.mpm)}${h.wifi8 ? ` · Wi-Fi 8 ${num(h.wifi8.mpm)}` : " · Wi-Fi 8 not in this test"} · ${esc(h.source)}</div><span></span></div></div>`;
  }
  s += `</div>${axis(sc, "missed control packets per million · log scale · lines: 95% interval over scenes")}</div>
    <p class="foot">Where ours loses: two AP paths alone lose the hardest condition (row 4) and need robot relay; a single radio loses to 5G under independent blockage (see <a href="#c-wifi-vs-5g">Wi-Fi vs private 5G</a>, V4).</p>`;

  // scoreboard
  s += `<h2>Verdict scoreboard</h2><p class="lede">Each pre-registered hypothesis as written before the run, scored by the campaign's own evaluation. Green passed, red outline failed.</p>
    <div class="wrap board"><table><tr><th>Campaign</th><th>Scenes</th><th>Hypotheses</th><th class="num">Passed</th></tr>`;
  for (const c of C) {
    const p = c.hypotheses.filter((h) => h.pass).length;
    s += `<tr><td><a href="#${slug(c.id)}">${esc(c.title)}</a> ${c.status === "development" ? `<span class="chip dev">dev</span>` : ""}<div class="src">${esc(c.family)}</div></td>
      <td class="mono">${esc(c.setup.scenes)}</td>
      <td><div class="pips">${c.hypotheses.map((h) => `<span class="pip ${h.pass ? "pass" : "fail"}" title="${esc(h.id)}: ${esc(h.text)}">${esc(h.id)}</span>`).join("")}</div></td>
      <td class="num">${p}/${c.hypotheses.length}</td></tr>`;
  }
  s += `</table></div>`;
  return s;
}

// ---------------------------------------------------------------- campaigns
const FAMILY_ORDER = ["Native stack", "Against 5G and Wi-Fi 8", "Fleet capabilities", "Learned models"];
function sideNav(cur) {
  const C = B.campaigns;
  let nav = "", opts = "";
  for (const f of FAMILY_ORDER) {
    const cs = C.filter((c) => c.family === f);
    if (!cs.length) continue;
    nav += `<h6>${esc(f)}</h6>` + cs.map((c) => `<a href="#${slug(c.id)}" class="${c.id === cur ? "on" : ""}">${esc(c.title)}<small>${esc(c.setup.scenes)}${c.status === "development" ? " · dev" : ""}</small></a>`).join("");
    opts += `<optgroup label="${esc(f)}">${cs.map((c) => `<option value="${slug(c.id)}" ${c.id === cur ? "selected" : ""}>${esc(c.title)}</option>`).join("")}</optgroup>`;
  }
  return `<aside class="side"><nav>${nav}</nav><select id="pick" aria-label="Choose a campaign">${opts}</select></aside>`;
}
function fmtParam(k, v) {
  if (v === true) return `<span class="p">${esc(k)}</span>`;
  if (v === false) return `<span class="p off">${esc(k)}</span>`;
  if (Array.isArray(v)) v = v.join("–");
  return `<span class="p">${esc(k)}=${esc(v)}</span>`;
}
function condParams(o) {
  return Object.entries(o || {}).map(([k, v]) => fmtParam(k, v)).join(" ");
}
function blockageText(b) {
  if (!b) return "—";
  if (b.model === "independent") return `Independent per robot: ${b.rate_per_robot_min}/robot-min, ${b.depth_db} dB, mean ${b.mean_s} s`;
  return `Moving forklifts: ${b.blockers} blockers, radius ${b.radius_m} m, ${b.speed_mps.join("–")} m/s, ${b.depth_db} dB`;
}
function campaign(id) {
  const c = B.campaigns.find((x) => x.id === id) || B.campaigns[0];
  const S = c.setup;
  let d = `<div class="meta"><span class="eyebrow">${esc(c.family)}</span>${statusChip(c.status)}</div>
    <h1>${esc(c.title)}</h1><p class="q">${esc(c.question)}</p>
    <div class="src">results ${esc(c.sources.results)} · protocol ${esc(c.sources.protocol)}${c.sources.results_doc && c.sources.results_doc !== c.sources.protocol ? ` · write-up ${esc(c.sources.results_doc)}` : ""}</div>
    <h2>Setup</h2><div class="setup">
      <div class="card"><div class="k">Scenes</div><div class="v mono">${esc(S.scenes)} <span style="color:var(--faint)">(${S.n_scenes})</span></div></div>
      <div class="card"><div class="k">Robots per scene</div><div class="v mono">${esc(S.robots)}</div></div>
      <div class="card"><div class="k">Simulated time per scene</div><div class="v mono">${S.duration_s ? num(S.duration_s, 0) + " s" : S.survey_s ? "survey " + num(S.survey_s, 0) + " s" : "—"}</div></div>
      <div class="card"><div class="k">Radio map</div><div class="v">${esc(S.map)}</div></div>
      <div class="card"><div class="k">Blockage</div><div class="v">${esc(blockageText(S.blockage))}</div><div class="src">${esc(S.blockage?.source)}</div></div>
      ${S.sensing ? `<div class="card"><div class="k">Obstacle sensing</div><div class="v">${esc(S.sensing)}</div></div>` : ""}
    </div>
    <h3>Conditions</h3><div class="wrap"><table><tr><th>Condition</th><th>Parameters</th></tr>
      ${Object.entries(S.conditions || {}).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td><div class="params">${condParams(v)}</div></td></tr>`).join("")}</table></div>`;

  if (c.arms.length) {
    const ours = c.arms.filter((a) => a.family === "WirelessJEV");
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const common = ours.length > 1 ? Object.entries(ours[0].params).filter(([k, v]) => ours.every((a) => k in a.params && same(a.params[k], v))) : [];
    const ck = new Set(common.map(([k]) => k));
    d += `<h2>Arms and their parameters</h2><p class="lede">Each arm is a policy. Parameters shown are the ones that differ from the policy defaults in the simulator, read from the campaign code. Struck through means switched off.</p>
      ${common.length ? `<div class="card" style="margin-bottom:10px"><div class="k" style="font-size:.78rem;color:var(--muted);margin-bottom:6px">Shared by every WirelessJEV arm in this campaign</div><div class="params" style="display:flex;flex-wrap:wrap;gap:4px">${common.map(([k, v]) => fmtParam(k, v)).join("")}</div></div>` : ""}
      <div class="wrap"><table><tr><th>Arm</th><th>Family</th><th>${common.length ? "Parameters (WirelessJEV arms: beyond the shared set)" : "Parameters"}</th></tr>
      ${c.arms.map((a) => {
        const ps = Object.entries(a.params).filter(([k]) => !(a.family === "WirelessJEV" && ck.has(k)));
        return `<tr><td style="min-width:200px">${esc(a.label)}<div class="src">${esc(a.id)}</div></td><td style="white-space:nowrap"><span class="legend" style="margin:0"><span><i style="background:${famColor(a.family)}"></i>${esc(a.family)}</span></span></td>
        <td><div class="params">${ps.length ? ps.map(([k, v]) => fmtParam(k, v)).join("") : "<span class='src'>shared set only</span>"}</div></td></tr>`;
      }).join("")}</table></div>`;
  }

  if (Object.keys(c.results).length) {
    const conds = Object.keys(c.results);
    d += `<h2>Results</h2><div class="seg" role="tablist">${conds.map((k, i) => `<button type="button" data-cond="${esc(k)}" class="${i ? "" : "on"}">${esc(k)}</button>`).join("")}</div>
      <div id="res"></div>`;
  }
  if (c.table) d += `<h2>Results</h2>${tableResults(c)}`;

  if (c.hypotheses.length) {
    d += `<h2>Pre-registered hypotheses</h2><ul class="hyps">${c.hypotheses.map((h) => `<li><div><span class="id">${esc(h.id)}</span><br><span class="chip ${h.pass ? "pass" : "fail"}">${h.pass ? "passed" : "failed"}</span></div>
      <div>${esc(h.text) || "<i>text not found in the protocol</i>"}${Object.keys(h.detail || {}).length ? `<details><summary>numbers from the evaluation</summary><div>${Object.entries(h.detail).map(([k, v]) => `${esc(k)}: ${esc(typeof v === "number" ? +v.toFixed(4) : v)}`).join(" · ")}</div></details>` : ""}</div></li>`).join("")}</ul>
      <div class="src" style="margin-top:6px">text: ${esc(c.sources.protocol)} · verdicts: ${esc(c.sources.results)}</div>`;
  }
  if (c.says.length) d += `<h2>What it says</h2><ul class="says">${c.says.map((x) => `<li>${esc(x)}</li>`).join("")}</ul><div class="src" style="margin-top:6px">${esc(c.sources.results_doc || c.sources.protocol)}</div>`;
  return `<div class="split">${sideNav(c.id)}<div class="detail">${d}</div></div>`;
}
function resultsFor(c, cond) {
  const R = c.results[cond];
  const armById = Object.fromEntries(c.arms.map((a) => [a.id, a]));
  const rows = Object.entries(R).map(([id, st]) => ({ id, label: armById[id]?.label || id, family: armById[id]?.family || "WirelessJEV", ...st }))
    .sort((a, b) => a.mpm - b.mpm);
  const fams = [...new Set(rows.map((r) => r.family))];
  const hasCounts = rows.some((r) => r.misses !== undefined);
  const extra = [["availability_1s", "1 s uptime", (v) => pct(v, 3)], ["safety_stop_fraction", "stopped", (v) => pct(v, 3)], ["control_gap_p99_ms", "p99 gap ms", (v) => num(v, 0)], ["airtime_domain_sum", "airtime", (v) => num(v, 2)]]
    .filter(([k]) => rows.some((r) => r[k] !== undefined));
  let s = `<div class="card">${legend(fams)}${barChart(rows)}</div>
    <div class="wrap" style="margin-top:10px"><table><tr><th>Arm</th><th class="num">per million</th><th class="num">95% interval</th>${hasCounts ? `<th class="num">missed / sent</th>` : ""}${extra.map((e) => `<th class="num">${e[1]}</th>`).join("")}<th>per scene</th></tr>
    ${rows.map((r) => `<tr><td style="min-width:200px">${esc(r.label)}</td><td class="num">${num(r.mpm)}</td><td class="num">${r.ci ? `${num(r.ci[0])}–${num(r.ci[1])}` : "—"}</td>
      ${hasCounts ? `<td class="num">${r.misses !== undefined ? `${r.misses.toLocaleString()} / ${r.sent.toLocaleString()}` : "—"}</td>` : ""}
      ${extra.map(([k, , f]) => `<td class="num">${r[k] !== undefined ? f(r[k]) : "—"}</td>`).join("")}
      <td class="mono" style="font-size:.72rem;white-space:nowrap">${(r.per_scene || []).map((v) => num(v)).join(" · ")}</td></tr>`).join("")}</table></div>
    <div class="src" style="margin-top:6px">${esc(rows[0]?.pooling === "mean of scene rates" ? "Older result file without counts: per-million figure is the mean of per-scene rates." : "Pooled: total missed ÷ total sent across scenes.")} Source ${esc(c.sources.results)}, condition ${esc(cond)}.</div>`;
  return s;
}
function tableResults(c) {
  const t = c.table;
  const keys = Object.keys(t[0]);
  const lab = { scene: "scene", level_rmse_db: "level RMSE dB", level_bias_db: "bias dB", no_backup_precision: "no-backup right", no_backup_recall: "no-backup found", weak_precision: "weak right", weak_recall: "weak found",
    clearly_weak_found: "clearly weak found", clearly_fine_flagged: "clearly fine flagged", worst_ap_rank: "worst-AP rank", covered_control: "control covered", with_backup_control: "control with backup",
    survey_s: "survey s", prior: "prior", rmse_db: "map RMSE dB", coverage_90: "90% coverage", sites: "test sites" };
  return `<div class="wrap"><table><tr>${keys.map((k) => `<th class="${k === "prior" ? "" : "num"}">${esc(lab[k] || k)}</th>`).join("")}</tr>
    ${t.map((r) => `<tr>${keys.map((k) => `<td class="${k === "prior" ? "" : "num"}">${typeof r[k] === "number" ? (r[k] <= 1 && !/rmse|bias|scene|rank|survey|sites/.test(k) ? pct(r[k], 0) : num(r[k], Number.isInteger(r[k]) ? 0 : 2)) : esc(r[k])}</td>`).join("")}</tr>`).join("")}</table></div>
    <div class="src" style="margin-top:6px">${esc(c.sources.results)}</div>`;
}

// ---------------------------------------------------------------- setups
function setups() {
  const C = B.campaigns;
  let s = `<div class="eyebrow">How the setups differ</div><h1>Setup matrix</h1>
    <p class="lede">Every campaign runs the same simulator core (10 ms slots, 100 robots, a 50 ms control deadline). What changes between them is listed here, read from the result files and the campaign code.</p>
    <div class="wrap"><table><tr><th>Campaign</th><th>Status</th><th>Scenes</th><th class="num">Time</th><th>Conditions</th><th>Blockage</th><th>Radio map</th><th class="num">Arms</th><th>Compared against</th></tr>`;
  for (const c of C) {
    const fams = [...new Set(c.arms.map((a) => a.family))].filter((f) => f !== "WirelessJEV");
    s += `<tr><td><a href="#${slug(c.id)}">${esc(c.title)}</a></td><td>${statusChip(c.status)}</td><td class="mono">${esc(c.setup.scenes)}</td>
      <td class="num">${c.setup.duration_s ? num(c.setup.duration_s, 0) + " s" : "—"}</td>
      <td>${Object.keys(c.setup.conditions || {}).map((k) => `<span class="p">${esc(k)}</span>`).join(" ")}</td>
      <td>${c.setup.blockage?.model === "independent" ? "independent" : "moving forklifts"}</td><td style="min-width:200px">${esc(c.setup.map)}</td>
      <td class="num">${c.arms.length || "—"}</td><td>${fams.length ? legend(fams) : "<span class='src'>within our stack</span>"}</td></tr>`;
  }
  s += `</table></div>`;

  // conditions glossary
  const conds = {};
  for (const c of C) for (const [k, v] of Object.entries(c.setup.conditions || {})) {
    const key = k + JSON.stringify(v);
    (conds[key] ||= { name: k, params: v, used: [] }).used.push(c);
  }
  s += `<h2>Every named condition</h2><p class="lede">The same name can mean different things in different campaigns; each distinct definition is listed once.</p>
    <div class="wrap"><table><tr><th>Name</th><th>Parameters</th><th>Used in</th></tr>
    ${Object.values(conds).map((x) => `<tr><td class="mono">${esc(x.name)}</td><td><div class="params">${condParams(x.params)}</div></td><td>${x.used.map((c) => `<a href="#${slug(c.id)}">${esc(c.title)}</a>`).join(", ")}</td></tr>`).join("")}</table></div>`;

  // parameter usage
  const P = {};
  for (const c of C) for (const a of c.arms) for (const [k, v] of Object.entries(a.params)) {
    if (k === "kind") continue;
    const e = (P[k] ||= { arms: 0, camps: new Set(), vals: new Set() });
    e.arms++; e.camps.add(c.title); e.vals.add(Array.isArray(v) ? v.join("–") : String(v));
  }
  const rows = Object.entries(P).sort((a, b) => b[1].arms - a[1].arms);
  s += `<h2>Policy parameters in use</h2><p class="lede">Each parameter that any arm changes from the simulator default, how many arms change it, and the values used.</p>
    <div class="wrap"><table><tr><th>Parameter</th><th class="num">Arms</th><th class="num">Campaigns</th><th>Values used</th></tr>
    ${rows.map(([k, e]) => `<tr><td class="mono">${esc(k)}</td><td class="num">${e.arms}</td><td class="num">${e.camps.size}</td><td class="mono" style="font-size:.76rem">${[...e.vals].slice(0, 12).map(esc).join(" · ")}</td></tr>`).join("")}</table></div>
    <div class="src" style="margin-top:6px">Defaults: src/wirelessjev/native/sim.py and src/wirelessjev/native_ext/sim_fleet.py (Policy).</div>`;
  return s;
}

// ---------------------------------------------------------------- learned models
function learned() {
  const L = B.learned;
  const oc = (o) => ({ "rf-win": "better signal prediction", win: "win", neutral: "no clear effect", loss: "loss", finding: "headroom finding" }[o] || o);
  const counts = (list) => Object.entries(list.reduce((a, x) => ((a[x.outcome] = (a[x.outcome] || 0) + 1), a), {}))
    .map(([k, n]) => `<span class="chip ${k}">${n} ${oc(k)}</span>`).join("");
  const mini = (vals, unit) => {
    const e = Object.entries(vals || {}).filter(([, v]) => typeof v === "number");
    if (!e.length) return "";
    const mx = Math.max(...e.map(([, v]) => v)) || 1;
    return e.map(([k, v]) => `<div class="mini"><span>${esc(k)}</span><span class="t"><i style="width:${(100 * v) / mx}%"></i></span><span class="n">${num(v, v >= 100 ? 0 : v >= 10 ? 1 : 2)}</span></div>`).join("") + `<div class="src">${esc(unit)}</div>`;
  };
  const card = (x) => `<div class="card study"><div class="meta"><span class="chip ${x.outcome}">${esc(oc(x.outcome))}</span>${x.owner === "ours" ? statusChip(x.status) : `<span class="chip">Codex</span>`}</div>
    <h3>${esc(x.title)}</h3><div style="font-size:.86rem">${esc(x.verdict)}</div>
    <dl><dt>Model</dt><dd>${esc(x.model)}</dd><dt>Trained on</dt><dd>${esc(x.trained_on)}</dd><dt>Baseline</dt><dd>${esc(x.baseline)}</dd><dt>Metric</dt><dd>${esc(x.metric)}</dd></dl>
    ${mini(x.values, x.unit)}${x.quote ? `<div class="quote">“${esc(x.quote)}”</div>` : ""}<div class="src" style="margin-top:6px">${esc(x.source)}</div></div>`;
  return `<div class="eyebrow">Learned models and RadioJEPA</div><h1>What learned models have and have not shown</h1>
    <div class="default"><b>${esc(L.default)}</b> <span class="src">(${esc(L.default_source)})</span></div>
    <p class="lede">Learned radio models have repeatedly predicted signal better than RadioMap without controlling better, because a perfect static map is worth only about 9% of misses (<a href="#c-headroom-ladder">headroom ladder</a>). The gain sits in knowing what is blocked now and next, which is where the one learned control win (foresight) came from.</p>
    <h2>Ours</h2><p class="lede">Numbers computed from the result files.</p><div class="counts">${counts(L.ours)}</div><div class="cards">${L.ours.map(card).join("")}</div>
    <h2>Codex's RadioJEPA and world-model line</h2><p class="lede">These results exist only in Codex's documents; each figure is quoted with its document path and a short quote.</p><div class="counts">${counts(L.codex)}</div>
    <div class="cards">${L.codex.map(card).join("")}</div>
    <p class="foot">Not simulation, so not shown here: ${L.not_simulation.map((x) => `${esc(x.title)} (<span class="src">${esc(x.doc)}</span>)`).join("; ")}.</p>`;
}

// ---------------------------------------------------------------- method
function method() {
  return `<div class="method"><div class="eyebrow">Method</div><h1>How the numbers are made</h1>
    <h2>The metric</h2><p>${esc(B.metric)} <span class="src">${esc(B.metric_source)}</span></p>
    <h2>Intervals</h2><p>${esc(B.intervals)}</p>
    <h2>Fresh and development scenes</h2><p>Each campaign's hypotheses were written down before it ran and are scored on scenes that were not used to build or tune anything ("fresh"). The headroom ladder runs on development scenes 400–403 on purpose, because it measures a ceiling rather than testing a claim.</p>
    <h2>Arms and parameters</h2><p>Result files record an arm's name. Its parameters are rebuilt by importing the same arm definitions the campaign used and listing the fields that differ from the policy defaults. The ladder, foresight and shared-sensing arms are built inline in their runners; they are rebuilt here from the same constants.</p>
    <h2>Codex's studies</h2><p>Codex's learned-model studies are recorded only in documents. They are listed with quoted figures and the document path, never recomputed.</p>
    <h2>Rebuild</h2><p><code>.venv/bin/python studio/build_bundle.py</code> then <code>.venv/bin/python studio/build_artifact.py</code>.</p>
    <p class="src">Bundle generated ${esc(B.generated)}${B.commit ? ` at commit ${esc(B.commit)}` : ""}. Sources: ${[...new Set(B.campaigns.flatMap((c) => [c.sources.results, c.sources.protocol]))].map(esc).join(" · ")}</p>
    <p class="foot">Field testing (the real SmartCar, radio mapping on site, running WirelessJEV live) lives in the field platform, not in this Studio.</p></div>`;
}

// ---------------------------------------------------------------- routing
function route() {
  const h = (location.hash || "#overview").slice(1);
  let tab = h, html;
  if (h.startsWith("c-")) { tab = "campaigns"; html = campaign(h.slice(2).replace(/-/g, "_")); }
  else if (h === "campaigns") html = campaign(B.campaigns[0].id);
  else if (h === "setups") html = setups();
  else if (h === "learned") html = learned();
  else if (h === "method") html = method();
  else { tab = "overview"; html = overview(); }
  view.innerHTML = html;
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("on", a.dataset.tab === tab));
  const pick = document.getElementById("pick");
  if (pick) pick.onchange = () => (location.hash = pick.value);
  const res = document.getElementById("res");
  if (res) {
    const id = h.startsWith("c-") ? h.slice(2).replace(/-/g, "_") : B.campaigns[0].id;
    const c = B.campaigns.find((x) => x.id === id) || B.campaigns[0];
    const show = (cond) => {
      res.innerHTML = resultsFor(c, cond);
      document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.cond === cond));
    };
    document.querySelectorAll(".seg button").forEach((b) => (b.onclick = () => show(b.dataset.cond)));
    show(Object.keys(c.results)[0]);
  }
  window.scrollTo(0, 0);
}

// theme: system by default; the button pins the opposite of what is showing
const themeBtn = document.getElementById("theme");
try { const t = localStorage.getItem("studio-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
themeBtn.onclick = () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("studio-theme", next); } catch (e) {}
};

const embedded = document.getElementById("bundle");
(embedded ? Promise.resolve(JSON.parse(embedded.textContent)) : fetch("data/evidence.json", { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }))
  .then((b) => { B = b; route(); window.addEventListener("hashchange", route); })
  .catch((e) => { view.innerHTML = `<div class="card"><b>No data bundle.</b> Run <code>.venv/bin/python studio/build_bundle.py</code> and serve <code>studio/</code> over HTTP. (${esc(e.message)})</div>`; });
