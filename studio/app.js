// Preceptual WirelessJEV Studio. Every number on screen comes from
// data/bundle.json, built by `python -m wirelessjev.studio_data` from real run
// artifacts. Missing artifacts are shown as missing, never invented.

const view = document.getElementById("view");
const crumb = document.getElementById("crumb");
let B = null;
let player = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "—");
const pct = (x, d = 0) => (typeof x === "number" ? (100 * x).toFixed(d) + "%" : "—");
const md = (s) => esc(s).replace(/\*([^*]+)\*/g, "<em>$1</em>").replace(/`([^`]+)`/g, "<code>$1</code>");
const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
const missing = (what, cmd) => `<div class="callout"><b>${esc(what)} not available yet.</b> Generate it with <code>${esc(cmd)}</code>, then rebuild the bundle with <code>uv run python -m wirelessjev.studio_data</code>.</div>`;

function table(head, rows) {
  return `<div class="wrap"><table><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr>${rows
    .map((r) => `<tr${r.hl ? ' class="hl"' : ""}>${(r.cells || r).map((c) => `<td>${c}</td>`).join("")}</tr>`)
    .join("")}</table></div>`;
}
function kpi(label, value, note = "", cls = "") {
  return `<div class="card kpi"><div class="label">${esc(label)}</div><div class="value ${cls}">${value}</div><div class="note">${note}</div></div>`;
}
function spark(series, colors, h = 120) {
  const all = series.flatMap((s) => s.values);
  if (!all.length) return "";
  const lo = Math.min(...all), hi = Math.max(...all), n = Math.max(...series.map((s) => s.values.length));
  const X = (i) => (i / Math.max(n - 1, 1)) * 100, Y = (v) => h - 8 - ((v - lo) / Math.max(hi - lo, 1e-9)) * (h - 16);
  const paths = series.map((s, k) => `<path stroke="${colors[k]}" d="${s.values.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(2)},${Y(v).toFixed(2)}`).join("")}"/>`).join("");
  const legend = series.map((s, k) => `<span><i style="background:${colors[k]}"></i>${esc(s.name)}</span>`).join("");
  return `<svg class="spark" viewBox="0 0 100 ${h}" preserveAspectRatio="none">${paths}</svg><div class="legend">${legend}<span>range ${fmt(lo, 2)} – ${fmt(hi, 2)}</span></div>`;
}
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ------------------------------------------------------------------ pages
const STEPS = [
  ["map", "Twin & site", "100 AMRs, racked hall, 32 customer BSSIDs"],
  ["decisions", "Candidate plans", "Exact oracles build complete fleet options"],
  ["decisions", "WirelessJEV decides", "Frozen d3-lite picks a plan from the mission text"],
  ["swarm", "Validate & share", "Swarm merge must beat the live release"],
  ["execution", "Supervised execution", "SmartCarCmd transactions, never resent"],
];

function overview() {
  const t = B.bench_10car, w = B.bench_warehouse, cl = B.closedloop_10car;
  const k = [];
  if (t) {
    const d = t.fleet_distilled.metrics, z = t.fleet_zero_shot_single_choice;
    k.push(kpi("10-car regret · distilled WirelessJEV", fmt(d.jev_regret), `MILP ${fmt(z.baseline_milp_regret)} · legacy ${fmt(z.baseline_legacy_regret)} · ${d.n_test_cases} test decisions`, "good"));
    k.push(kpi("Intent-flip (distilled)", pct(t.fleet_distilled.intent_flip.directionally_correct_flips), "choice changes the right way when only the mission text changes"));
  }
  if (B.bench_intent) {
    const it = B.bench_intent;
    k.push(kpi("100-robot plan gap · intent + fast planner", fmt(it.plan_regret_vs_best_known.fast_intent_distilled), `previous ${fmt(it.plan_regret_vs_best_known.old_optionwise_choice)} · MILP ${fmt(it.plan_regret_vs_best_known.milp)} · ${it.latency_ms.planner_search_mean.toFixed(0)} ms/decision`, "good"));
  } else if (w) {
    const cands = [["warehouse", w.distilled_warehouse], ["combined", w.combined_on_warehouse]].filter((x) => x[1]);
    const [name, c] = cands.reduce((a, b) => (b[1].jev_regret < a[1].jev_regret ? b : a));
    k.push(kpi(`100-robot regret · ${name} adapter`, fmt(c.jev_regret), `MILP ${fmt(w.zero_shot_single_choice.baseline_milp_regret)} · legacy ${fmt(w.zero_shot_single_choice.baseline_legacy_regret)}`, "good"));
  } else k.push(kpi("100-robot benchmark", "pending", "run bench_warehouse"));
  if (cl) k.push(kpi("10-car closed loop utility", fmt(cl.summary.wirelessjev.mean_utility, 2), `MILP ${fmt(cl.summary.milp.mean_utility, 2)} · legacy ${fmt(cl.summary.legacy.mean_utility, 2)}`));
  return `<h1>WirelessJEV workspace</h1>
  <p class="lede">Mission-conditioned Wi-Fi coordination for a Raspbot/AMR fleet. The numeric layer builds complete association plans; a frozen open decision model (<code>${esc(B.model.repo)}@${esc(B.model.revision.slice(0, 7))}</code>) chooses one from the mission text; the swarm layer only releases updates that beat what is live; execution uses the existing SmartCarCmd protocol.</p>
  <div class="steps">${STEPS.map((s, i) => `<div class="step${i === 0 ? " active" : ""}" data-go="${s[0]}"><div class="t"><span class="n">${i + 1}</span>${esc(s[1])}</div><div class="d">${esc(s[2])}</div></div>`).join("")}</div>
  <div class="grid k4">${k.join("")}</div>
  <div class="callout"><b>Claim discipline.</b> ${esc(B.evidence_class)}. Regret is measured against the exact oracle on held-out twin scenes. Inference latency, RF handover time and application interruption are separate measurements; none of these results is a field acceptance test. The one-car real test is <b>${esc(B.one_car_test.status)}</b>.</div>
  <div class="grid k2">
    <div class="card"><h3>What runs where</h3>${table(["Layer", "Implementation", "Learns?"], [
      ["Twin", "120 × 80 m racked hall, 16 dual-band AP sites, route-planned AMRs", "no"],
      ["Oracles", "Exact evaluator; Shen/Li/Lam reward families; MILP, Hungarian, colouring", "no"],
      ["Decision", "d3-lite option-wise yes/no scoring over complete plans", "readout only (1024 params)"],
      ["Swarm", "Leader election, weighted merge, validate-vs-live, HMAC hash-chained log", "no"],
      ["Execution", "SmartCarCmd status/connect; APPLIED / NOT_APPLIED / UNKNOWN", "no"],
    ])}</div>
    <div class="card"><h3>Not in this build</h3><ul class="note">
      <li>Motion control: SmartCar.zip ships no motor/camera interface.</li>
      <li>OCUDU / 5G NR / custom PHY execution (needs Linux + SDR).</li>
      <li>Sub-10 ms control: d3-lite decides in seconds on the M5; the fast loop stays conventional.</li>
      <li>Channel planning on customer APs (modelled, disabled).</li></ul></div>
  </div>`;
}

function mapPage() {
  const cl = B.closedloop_warehouse;
  const runs = (cl?.runs || []).filter((r) => r.frames);
  if (!runs.length) return `<h1>Fleet map</h1>` + missing("Warehouse replay", "uv run python -m wirelessjev.closedloop --warehouse --total 600 --out artifacts/closedloop_warehouse.json");
  const pref = runs.find((r) => r.policy === "wirelessjev_fast") || runs.find((r) => r.policy === "wirelessjev") || runs[0];
  setTimeout(() => mountMap(runs, pref.policy), 0);
  return `<h1>Fleet map</h1>
  <p class="lede">Replay of a 10-minute warehouse shift in the twin: 100 robots on planned aisle routes, mission changes, a robot exceeding its power cap, and a 30 s coordinator outage. Click a robot to see its link.</p>
  ${cl.provisional ? `<div class="callout">${esc(cl.provisional)}.</div>` : ""}
  <div class="grid map"><div class="card"><div id="mission" class="mission"></div><div class="mapbox" id="mapbox"></div>
    <div class="controls"><button id="play" class="primary">▶ Play</button><input id="scrub" type="range" min="0" value="0"><span id="clock" class="mono"></span>
      <select id="policy">${runs.map((r) => `<option value="${r.policy}"${r.policy === pref.policy ? " selected" : ""}>${esc(r.policy)}</option>`).join("")}</select>
      <select id="colour"><option value="sat">colour: service</option><option value="rssi">colour: signal</option><option value="band">colour: band</option></select></div>
    <div class="legend" id="legend"></div></div>
    <div><div class="card" id="frame-stats"></div><div class="card" id="robot" style="margin-top:12px"><h3>Robot</h3><p class="note">Select a robot on the map.</p></div></div></div>
  <div class="card" style="margin-top:16px"><h3>Mission utility over the shift</h3>${spark(runs.map((r) => ({ name: r.policy, values: r.timeline.map((s) => s.utility) })), [css("--accent"), css("--warn"), css("--muted")])}</div>`;
}

function floorSVG(aps) {
  const L = B.layout, W = L.width, H = L.height;
  let s = `<svg class="floor" viewBox="-2 -2 ${W + 4} ${H + 4}" role="img" aria-label="Warehouse floor plan">`;
  for (let x = 0; x <= W; x += 10) s += `<line class="grid-l" x1="${x}" y1="0" x2="${x}" y2="${H}"/><text class="lbl" x="${x}" y="${H + 1.8}">${x}</text>`;
  for (let y = 0; y <= H; y += 10) s += `<line class="grid-l" x1="0" y1="${y}" x2="${W}" y2="${y}"/>`;
  for (const x of L.cross_aisles) s += `<rect class="xaisle" x="${x - 2}" y="0" width="4" height="${H}"/>`;
  L.rack_rows.forEach((y, i) => { for (const [a, b] of L.rack_segments) s += `<rect class="rack" x="${a}" y="${y - 0.6}" width="${b - a}" height="1.2"/>`; s += `<text class="lbl" x="${L.rack_segments[0][0] - 4}" y="${y + 0.8}">R${String(i + 1).padStart(2, "0")}</text>`; });
  s += `<text class="lbl" x="12" y="2.6">SHIPPING · DROP POINTS</text><text class="lbl" x="12" y="${H - 0.8}">RECEIVING</text><text class="lbl" x="0.3" y="12">CHARGE</text>`;
  for (const [x, y] of L.drop_points) s += `<rect x="${x - 1.5}" y="${y - 1}" width="3" height="2" fill="none" stroke="${css("--muted")}" stroke-width=".3"/>`;
  for (const [x, y] of L.receive_points) s += `<rect x="${x - 1.5}" y="${y - 1}" width="3" height="2" fill="none" stroke="${css("--muted")}" stroke-width=".3"/>`;
  for (const [x, y] of L.chargers) s += `<rect x="${x - 1}" y="${y - 1}" width="2" height="2" fill="none" stroke="${css("--good")}" stroke-width=".3"/>`;
  const sites = {};
  aps.forEach((a) => (sites[a.site] = a));
  for (const a of Object.values(sites)) s += `<rect class="ap" x="${a.x - 1.2}" y="${a.y - 1.2}" width="2.4" height="2.4" rx=".4"/><text class="lbl" x="${a.x + 1.6}" y="${a.y - 1.2}">${esc(a.site.toUpperCase())}</text>`;
  return s + `<g id="links"></g><g id="robots"></g></svg>`;
}

function mountMap(runs, policy) {
  const byPolicy = Object.fromEntries(runs.map((r) => [r.policy, r]));
  let run = byPolicy[policy], k = 0, sel = null;
  const box = document.getElementById("mapbox");
  const scrub = document.getElementById("scrub");
  box.innerHTML = floorSVG(run.aps);
  scrub.max = run.frames.length - 1;
  const colourSel = document.getElementById("colour");
  const colour = (f, i) => {
    const mode = colourSel.value;
    if (mode === "band") return run.aps[f.ap[i]].band === "5g" ? css("--accent") : css("--warn");
    const v = mode === "sat" ? f.sat[i] : f.rssi[i];
    const [g, w] = mode === "sat" ? [0.98, 0.8] : [-65, -75];
    return v >= g ? css("--good") : v >= w ? css("--warn") : css("--bad");
  };
  const legends = { sat: ["service ≥ 98%", "80–98%", "< 80%"], rssi: ["signal ≥ −65 dBm", "−65 to −75", "< −75 dBm"], band: ["5 GHz", "2.4 GHz"] };
  function draw() {
    const f = run.frames[k], tl = run.timeline[k];
    const g = document.getElementById("robots"), links = document.getElementById("links");
    const q = new Set(f.quarantined);
    let out = "";
    f.xy.forEach(([x, y], i) => {
      const id = run.robots[i];
      out += `<circle class="robot${sel === i ? " sel" : ""}" data-i="${i}" cx="${x}" cy="${y}" r="0.9" fill="${colour(f, i)}"><title>${id}</title></circle>`;
      if (q.has(id)) out += `<circle class="quar" cx="${x}" cy="${y}" r="1.8"/>`;
    });
    g.innerHTML = out;
    links.innerHTML = sel == null ? "" : (() => { const a = run.aps[f.ap[sel]], [x, y] = f.xy[sel]; return `<line class="link" x1="${x}" y1="${y}" x2="${a.x}" y2="${a.y}"/>`; })();
    const mission = B.missions?.[tl.mission];
    document.getElementById("mission").innerHTML = `<b>${esc(tl.mission)}</b>${mission ? " — " + esc(mission.intent) : ""}${tl.coordinator_up ? "" : ' <span class="badge bad">COORDINATOR OFFLINE · local fallback</span>'}`;
    document.getElementById("clock").textContent = `t = ${tl.t.toFixed(0)} s`;
    const dec = run.decisions.filter((d) => d.t <= tl.t).slice(-1)[0];
    document.getElementById("frame-stats").innerHTML = `<h3>This step · ${esc(run.policy)}</h3><dl class="dl">
      <dt>Mission utility</dt><dd>${fmt(tl.utility, 3)}</dd><dt>Handovers this step</dt><dd>${tl.handovers}</dd>
      <dt>Critical flows below floor</dt><dd>${tl.critical_violations}</dd><dt>Fleet delivered</dt><dd>${fmt(tl.total_mbps, 0)} Mbps</dd>
      <dt>Quarantined</dt><dd>${f.quarantined.length ? esc(f.quarantined.join(", ")) : "none"}</dd>
      <dt>Last fleet decision</dt><dd>${dec ? `t=${dec.t}s${dec.choice ? " · " + esc(dec.choice) : ""}${dec.latency_ms ? " · " + dec.latency_ms.toFixed(0) + " ms" : ""}` : "—"}</dd></dl>
      <p class="note">Shift mean utility ${fmt(run.mean_utility, 3)} · ${run.total_handovers} handovers · ${fmt(run.mean_critical_violations, 2)} critical violations/step.</p>`;
    if (sel != null) {
      const a = run.aps[f.ap[sel]];
      document.getElementById("robot").innerHTML = `<h3>${esc(run.robots[sel])}</h3><dl class="dl">
        <dt>BSSID</dt><dd>${esc(a.id)}</dd><dt>Band · channel</dt><dd>${a.band === "5g" ? "5 GHz" : "2.4 GHz"} · ${a.channel}</dd>
        <dt>Signal</dt><dd>${fmt(f.rssi[sel], 1)} dBm</dd><dt>Service</dt><dd>${pct(f.sat[sel])} of demand</dd>
        <dt>Position</dt><dd>${f.xy[sel][0]}, ${f.xy[sel][1]} m</dd><dt>Status</dt><dd>${q.has(run.robots[sel]) ? '<span class="badge bad">QUARANTINED</span>' : '<span class="badge ok">TRUSTED</span>'}</dd></dl>`;
    }
    const swatch = colourSel.value === "band" ? [css("--accent"), css("--warn")] : [css("--good"), css("--warn"), css("--bad")];
    document.getElementById("legend").innerHTML = legends[colourSel.value].map((t, i) => `<span><i style="background:${swatch[i]}"></i>${t}</span>`).join("") + `<span><i style="background:none;border:1.5px solid ${css("--bad")}"></i>quarantined</span><span>■ AP site (5 + 2.4 GHz)</span>`;
    scrub.value = k;
  }
  box.addEventListener("click", (e) => { const i = e.target.dataset?.i; if (i != null) { sel = +i; draw(); } });
  scrub.oninput = () => { k = +scrub.value; draw(); };
  colourSel.onchange = draw;
  document.getElementById("policy").onchange = (e) => { run = byPolicy[e.target.value]; k = Math.min(k, run.frames.length - 1); scrub.max = run.frames.length - 1; draw(); };
  const btn = document.getElementById("play");
  btn.onclick = () => {
    if (player) { clearInterval(player); player = null; btn.textContent = "▶ Play"; return; }
    btn.textContent = "❚❚ Pause";
    player = setInterval(() => { k = (k + 1) % run.frames.length; draw(); }, matchMedia("(prefers-reduced-motion: reduce)").matches ? 900 : 350);
  };
  draw();
}

function decisionsPage() {
  const cl = B.closedloop_warehouse;
  const run = (cl?.runs || []).find((r) => r.policy === "wirelessjev" && r.decisions?.some((d) => d.options));
  if (!run) return `<h1>Decisions</h1>` + missing("WirelessJEV warehouse decisions", "uv run python -m wirelessjev.closedloop --warehouse --adapter artifacts/warehouse/adapters/combined --total 600 --out artifacts/closedloop_warehouse.json");
  const decs = run.decisions.filter((d) => d.options);
  setTimeout(() => {
    const show = (i) => {
      const d = decs[i];
      document.querySelectorAll(".list-btn").forEach((b, j) => b.classList.toggle("active", j === i));
      const opts = [...d.options].sort((a, b) => b.p_true - a.p_true);
      const maxP = Math.max(...opts.map((o) => o.p_true), 1e-9);
      document.getElementById("dec-detail").innerHTML = `<div class="mission"><b>${esc(d.mission)}</b> — ${esc(d.intent)}</div>
        <p class="note">t = ${d.t} s · ${d.options.length} options · decision latency ${d.latency_ms.toFixed(0)} ms · regret vs oracle-best <b class="${d.regret <= 1e-6 ? "good" : d.regret < 0.1 ? "warn" : "bad"}">${fmt(d.regret, 3)}</b>${d.pinned.length ? ` · pinned (quarantined): ${esc(d.pinned.join(", "))}` : ""}</p>
        ${opts.map((o) => `<div class="opt${o.id === d.choice ? " chosen" : ""}"><div class="head"><b>${esc(o.id)}</b>
          ${o.id === d.choice ? '<span class="badge info">WIRELESSJEV CHOICE</span>' : ""}${o.id === d.oracle_best ? '<span class="badge ok">ORACLE BEST</span>' : ""}
          <span class="spacer"></span><span class="note">P(true) ${fmt(o.p_true, 3)} · oracle ${fmt(o.oracle, 3)} · moves ${o.moves}</span></div>
          <div class="bar" style="margin-top:6px"><span style="width:${(100 * o.p_true / maxP).toFixed(1)}%"></span></div>
          <div class="text">${esc(o.text)}</div></div>`).join("")}`;
    };
    document.querySelectorAll(".list-btn").forEach((b, i) => (b.onclick = () => show(i)));
    show(0);
  }, 0);
  const regrets = decs.map((d) => d.regret);
  return `<h1>Decisions</h1>
  <p class="lede">Every fleet decision in the recorded warehouse shift: the mission text the model saw, each complete plan with its digested consequences, the model's option-wise P(true), and the exact oracle's score. The model never sees which teacher produced an option.</p>
  <div class="grid k4">${kpi("Decisions", decs.length)}${kpi("Mean regret", fmt(regrets.reduce((a, b) => a + b, 0) / regrets.length))}${kpi("Matched oracle-best", pct(decs.filter((d) => d.regret <= 1e-6).length / decs.length))}${kpi("Mean latency", (decs.reduce((a, d) => a + d.latency_ms, 0) / decs.length / 1000).toFixed(1) + " s", B.bench_warehouse?.latency_ms?.device || "")}</div>
  <div class="grid map" style="margin-top:16px"><div class="card" id="dec-detail"></div>
  <div class="card"><h3>Shift decisions</h3>${decs.map((d) => `<button class="list-btn">t=${d.t}s · ${esc(d.mission)} · ${esc(d.choice)} <span class="${d.regret <= 1e-6 ? "good" : "warn"}">(${fmt(d.regret, 2)})</span></button>`).join("")}</div></div>`;
}

function regretRows(m, label, hl) {
  return { hl, cells: [esc(label), `<b>${fmt(m.jev_regret)}</b>`, pct(m.jev_top1), pct(m.defer_rate), m.n_test_cases] };
}
function researchResults() {
  let html = "";
  const run = B.training_run;
  if (run) {
    html += `<div class="card" style="margin-top:16px"><h3>Full RadioJEV rerun · ${esc(run.status)}</h3>
      <p class="sub">Seed ${esc(run.seed)} · snapshot at bundle export, not a live monitor. Historical results below remain separate from this rerun.</p>
      ${table(["Stage", "Status", "Epochs recorded", "Best validation regret", "Completed held-out result"],
        Object.entries(run.stages).map(([name, stage]) => {
          const history = stage.history || [], result = run.results[name];
          const best = history.length ? Math.min(...history.map((entry) => entry.val_regret)) : undefined;
          return [esc(name), esc(stage.status === "pending" ? "queued" : stage.status),
            history.length ? history[history.length - 1].epoch : "—", fmt(best),
            result?.test?.n ? `${fmt(result.test.regret)} (n=${result.test.n})` : result ? "available" : "pending"];
        }))}
      <p class="note">Validation progress is not a held-out test result. Only completed stages contribute new test metrics; an interrupted process is marked interrupted when exported on the training machine.</p></div>`;
  }
  const radio = B.radiojev || {};
  const arms = [["e1_radio_lora", "RadioJEV LoRA"], ["e1_radio_frozen", "RadioJEV frozen"],
    ["e1_intent", "Intent baseline"], ["e1_no_llm", "Numeric encoder (no LLM)"]];
  if (arms.some(([key]) => radio[key])) {
    html += `<div class="card" style="margin-top:16px"><h3>Published RadioJEV E1 · native radio inputs</h3>${table(
      ["Arm", "Held-out regret ± 95% CI", "n", "New-mission regret ± 95% CI", "n", "Held-out latency"],
      arms.map(([key, label]) => {
        const r = radio[key];
        if (!r) return [esc(label), "pending", "—", "pending", "—", "—"];
        const interval = (m) => m?.n ? `${fmt(m.regret)} ± ${fmt(m.regret_ci95)}` : "not evaluated";
        return [esc(label), interval(r.test), r.test?.n ?? "—", interval(r.novel), r.novel?.n ?? "—",
          r.test?.n ? `${fmt(r.test.latency_ms, 0)} ms` : "—"];
      }))}<p class="note">E1 LoRA is a completed exploratory result, not evidence that native radio inputs beat intent + planning. The separate intent benchmark reports shortlist regret 0.027 at about 134 ms per re-plan; it is not a paired E1 comparison. The numeric-only arm has no new-mission evaluation. Latencies are from the original experiment machine.</p>
      ${table(["Remaining experiment", "Status"], [["e2_radio_q", "E2 radio Q"], ["e2_e1_zero", "E2 E1 zero-shot"],
        ["e2_intent", "E2 intent"], ["e3", "E3 multitask"], ["e45", "E4/E5 swarm + unlearning"]]
        .map(([key, label]) => [label, radio[key] ? "result available" : "pending"]))}</div>`;
  }
  const env = B.envlearn;
  if (env) {
    const s = env.summary;
    const corrected = env.method === "measured-top8-censored-gaussian-v2";
    html += `<div class="card" style="margin-top:16px"><h3>E7 · learning the radio environment</h3>
      <p class="sub">${s.n_envs} synthetic environments · ${corrected ? "corrected scan censoring and Gaussian likelihood" : "legacy pipeline; censoring defect known"}. RMSE is evaluated on audible paths (truth RSSI ≥ −90 dBm); robust RMSE includes the pessimistic planning margin.</p>
      ${table(["Planner belief", "Audible RMSE (dB)", "Unheard RMSE (dB)", "Plan gap to truth search"],
        Object.keys(s.map_rmse_db).map((name) => [esc(name), fmt(s.map_rmse_db[name], 2),
          fmt(s.map_rmse_unheard_db?.[name], 2), fmt(s.regret_vs_truth_oracle[name])]))}
      <p class="note">The truth planner uses local search, not a proven global optimum. Lower plan gap does not establish lower map error. Unheard paths remain weakly constrained; these simulation results do not calibrate a real site.</p>`;
    if (corrected && B.envlearn_legacy) {
      const old = B.envlearn_legacy.summary.map_rmse_db;
      html += `<p class="note">Historical audible RMSE with the defective pipeline: ${fmt(old.learned_1min, 2)} dB at 1 minute → ${fmt(old.learned_15min, 2)} dB at 15 minutes. The original report is retained separately.</p>`;
    }
    html += "</div>";
  }
  const calibration = B.envlearn_calibration;
  if (calibration) {
    html += `<div class="card" style="margin-top:16px"><h3>E7 calibration study · new environments</h3>
      <p class="sub">${calibration.seeds.length} new environment seeds · matched surveys · map error only. Global censored calibration with nominal rack geometry is compared with E7 v2; it does not replace the default planner.</p>
      ${table(["Survey budget", "V2 audible RMSE", "Candidate audible RMSE", "V2 inaudible RMSE", "Candidate inaudible RMSE"],
        Object.entries(calibration.summary).map(([budget, results]) => [
          `${esc(budget)} min`, fmt(results.v2.audible, 2), fmt(results.geometry_censored.audible, 2),
          fmt(results.v2.inaudible, 2), fmt(results.geometry_censored.inaudible, 2)]))}
      <p class="note">Known AP positions/powers/height and nominal floor layout are inputs; hidden walls, materials and truth coefficients are not fitted from the evaluator. Improved map RMSE does not establish a better plan, a real-site calibration, or monotonic improvement with more data.</p></div>`;
  }
  return html;
}
function evidencePage() {
  const t = B.bench_10car, w = B.bench_warehouse;
  let s = `<h1>Evidence</h1><p class="lede">Held-out benchmark results. Regret = gap to the best option under the exact oracle (lower is better). Every WirelessJEV number comes from executing the pinned model; parity between the captured features and the model's own probabilities is checked on every case.</p>`;
  if (t) {
    const z = t.fleet_zero_shot_single_choice;
    s += `<div class="card"><h3>10-car fleet · ${t.n_cases} decisions</h3><div class="sub">parity error ≤ ${t.parity_err_max.toExponential(1)} · ${esc(t.model.device)}</div>${table(["Policy", "Mean regret", "Top-1", "DEFER", "n"], [
      regretRows(t.fleet_distilled.metrics, "WirelessJEV distilled (option-wise)", true),
      regretRows(z, "WirelessJEV zero-shot single choice"), regretRows(t.fleet_zero_shot_optionwise, "WirelessJEV zero-shot option-wise"),
      ...["milp", "legacy", "local_greedy", "coloring", "hungarian", "random", "strongest_rssi"].map((b) => [esc("Baseline: " + b), fmt(z[`baseline_${b}_regret`]), "—", "—", z.n_test_cases]),
    ])}</div>`;
    const b = t.bias_audit, d = t.fleet_distilled;
    s += `<div class="grid k3" style="margin-top:16px">
      <div class="card"><h3>Intent-flip</h3>${table(["Model", "Changes", "Right way"], [["zero-shot", pct(t.intent_flip_zero_shot_single_choice.model_flips_when_oracle_differs), pct(t.intent_flip_zero_shot_single_choice.directionally_correct_flips)], ["distilled", pct(d.intent_flip.model_flips_when_oracle_differs), `<b>${pct(d.intent_flip.directionally_correct_flips)}</b>`]])}<p class="note">${d.intent_flip.oracle_differs} pairs where the oracle's choice changes.</p></div>
      <div class="card"><h3>Bias audit</h3>${table(["Probe", "Rate"], [["same choice, order reversed", pct(b.order_reversal_consistency)], ["padding captures choice", pct(b.length_padding_capture)], ["“senior engineer” captures", pct(b.authority_wording_capture)], ["chance capture", pct(b.chance_capture)]])}</div>
      <div class="card"><h3>Privacy (MIA AUC)</h3>${table(["Readout", "AUC"], [["released", fmt(d.mia_auc_base_readout)], ["distilled", fmt(d.mia_auc_distilled)], ["region R0 before unlearning", fmt(d.mia_auc_region_before_unlearning)], ["region R0 after unlearning", fmt(d.mia_auc_region_after_unlearning)]])}<p class="note">0.5 = no membership leakage.</p></div></div>`;
    const tr = t.traffic, la = tr.label_accuracy;
    s += `<div class="grid k2" style="margin-top:16px"><div class="card"><h3>Traffic labels (Chen, Lam &amp; Li framework)</h3>${table(["Label", "Jev raw", "Jev digest", "Jev distilled", "Rules"], Object.keys(la.rule).map((k) => [k, pct(la.jev_raw[k]), pct(la.jev_digest[k]), `<b>${pct(tr.label_distillation_test_accuracy[k].distilled)}</b>`, pct(la.rule[k])]))}<p class="note">Chronos-Bolt MAE ${fmt(tr.forecast_mae.chronos_bolt_small, 2)} vs naive ${fmt(tr.forecast_mae.naive_last_value, 2)} Mbps.</p></div>
      <div class="card"><h3>Admission level (price-step)</h3>${table(["Policy", "Best one-step move"], [["WirelessJEV", pct(t.admission.jev_local_direction_accuracy)], ["demand rule", pct(t.admission.demand_rule_accuracy)], ["chance", pct(t.admission.chance)]])}<p class="note">An honest weak spot: numeric comparison even with stated consequences.</p></div></div>`;
  } else s += missing("10-car benchmark", "uv run python -m wirelessjev.bench --scenes 80");
  if (w) {
    const z = w.zero_shot_single_choice;
    const rows = [regretRows(z, "zero-shot single choice"), regretRows(w.zero_shot_optionwise, "zero-shot option-wise")];
    if (w.transfer_10car_adapter) rows.push(regretRows(w.transfer_10car_adapter, "10-car adapter, unchanged (transfer)"));
    const best = Math.min(w.distilled_warehouse.jev_regret, w.combined_on_warehouse?.jev_regret ?? Infinity);
    rows.push(regretRows(w.distilled_warehouse, "warehouse-distilled adapter", w.distilled_warehouse.jev_regret === best));
    if (w.combined_on_warehouse) rows.push(regretRows(w.combined_on_warehouse, "combined adapter (both sites)", w.combined_on_warehouse.jev_regret === best));
    ["milp", "legacy", "local_greedy", "coloring"].forEach((b) => rows.push([esc("Baseline: " + b), fmt(z[`baseline_${b}_regret`]), "—", "—", z.n_test_cases]));
    s += `<div class="card" style="margin-top:16px"><h3>100-robot warehouse · ${w.n_cases} decisions · route-aware oracle</h3><div class="sub">decision latency ${(w.latency_ms.optionwise_mean / 1000).toFixed(1)} s mean on ${esc(w.latency_ms.device)}</div>${table(["Policy", "Mean regret", "Top-1", "DEFER", "n"], rows)}${w.combined_on_10car ? `<p class="note">The combined adapter on the 10-car test split: regret ${fmt(w.combined_on_10car.jev_regret)}.</p>` : ""}</div>`;
  } else s += missing("100-robot warehouse benchmark", "uv run python -m wirelessjev.bench_warehouse --scenes 50");
  const it = B.bench_intent;
  if (it) {
    const sr = it.shortlist_regret, pr = it.plan_regret_vs_best_known, nv = it.novel_mission_plan_regret, la = it.intent_level_accuracy, lt = it.latency_ms;
    s += `<div class="card" style="margin-top:16px"><h3>Intent-to-objective WirelessJEV · ${it.n_decisions} warehouse decisions</h3><div class="sub">d3-lite reads the mission into objective weights (once per mission); an exact batched planner optimises them every step.</div>
      <div class="grid k3"><div>${table(["Shortlist regret", ""], [["option-wise distilled (previous)", fmt(sr.old_optionwise_distilled)], ["intent zero-shot", fmt(sr.intent_zero_shot)], ["intent distilled, held-out paraphrase", `<b class="good">${fmt(sr.intent_distilled_heldout_paraphrase)}</b>`], ["intent distilled, seen wording", fmt(sr.intent_distilled)]])}</div>
      <div>${table(["Gap to best known plan", ""], [["fast planner, intent distilled", `<b class="good">${fmt(pr.fast_intent_distilled)}</b>`], ["fast planner, intent zero-shot", fmt(pr.fast_intent_zero_shot)], ["best shortlist option", fmt(pr.best_shortlist)], ["previous option-wise choice", fmt(pr.old_optionwise_choice)], ["MILP", fmt(pr.milp)], ["legacy", fmt(pr.legacy)]])}</div>
      <div>${table(["Never-seen missions", ""], [["fast planner, intent distilled", `<b class="good">${fmt(nv.fast_intent_distilled)}</b>`], ["fast planner, intent zero-shot", fmt(nv.fast_intent_zero_shot)], ["MILP", fmt(nv.milp)], ["legacy", fmt(nv.legacy)]])}
      <p class="note">Intent level accuracy: held-out ${pct(la.test.zero_shot)} → ${pct(la.test.distilled)}; novel ${pct(la.novel.zero_shot)} → ${pct(la.novel.distilled)}.</p></div></div>
      <p class="note">Latency: planner ${lt.planner_setup_mean.toFixed(0)} ms setup + ${lt.planner_search_mean.toFixed(0)} ms search (p90 ${lt.planner_search_p90.toFixed(0)} ms) per decision; intent read ${(lt.intent_read_once_per_mission / 1000).toFixed(1)} s once per mission change; previously ~15 s per decision.</p></div>`;
  }
  s += researchResults();
  const cls = [["10-car (300 s)", B.closedloop_10car], ["warehouse shift (600 s)", B.closedloop_warehouse]].filter((x) => x[1]);
  s += `<div class="card" style="margin-top:16px"><h3>Closed loop</h3>${table(["Scenario", "Policy", "Mean utility", "Handovers", "Critical violations / step", "Decision latency", "Connect resends"], cls.flatMap(([name, c]) => Object.entries(c.summary).sort((a, b) => b[1].mean_utility - a[1].mean_utility).map(([p, v]) => [esc(name), esc(p), fmt(v.mean_utility, 3), v.total_handovers, fmt(v.mean_critical_violations, 2), v.decision_latency_ms_mean ? (v.decision_latency_ms_mean >= 1000 ? (v.decision_latency_ms_mean / 1000).toFixed(1) + " s" : v.decision_latency_ms_mean.toFixed(0) + " ms") : "—", v.connects_beyond_one_per_txn ?? "—"])))}</div>`;
  return s;
}

function swarmPage() {
  const sw = B.bench_10car?.swarm;
  let s = `<h1>Swarm &amp; trust</h1><p class="lede">Serverless merge following Li/Wang/Shen/Lam's swarm-learning DSA: admission check, leader by performance score, weighted merge, and validation against the <b>currently live</b> release before anything is released. Cars violating their contract (MSN 2024 reputation rule) are excluded from merges and pinned to their BSSID.</p>`;
  if (!sw) return s + missing("Swarm rounds", "uv run python -m wirelessjev.bench");
  const r1 = sw.round1, r2 = sw.round2_poisoned_updates;
  s += `<div class="grid k2"><div class="card"><h3>Round 1 · trusted cars</h3><p>${r1.released ? '<span class="badge ok">RELEASED</span>' : '<span class="badge bad">REJECTED</span>'} leader <b>${esc(r1.leader)}</b></p><dl class="dl"><dt>Participants</dt><dd>${r1.participants.length}</dd><dt>Excluded</dt><dd>${esc(r1.excluded_low_reputation.join(", ") || "none")}</dd><dt>Validation regret</dt><dd>${fmt(r1.validation.val_regret)}</dd><dt>Live regret</dt><dd>${fmt(r1.validation.live_val_regret ?? r1.validation.base_val_regret)}</dd></dl></div>
  <div class="card"><h3>Round 2 · poisoned updates</h3><p>${r2.released ? '<span class="badge bad">RELEASED</span>' : '<span class="badge ok">REJECTED</span>'} ${esc(r2.reason)}</p><dl class="dl"><dt>Validation regret</dt><dd>${fmt(r2.validation.val_regret)}</dd><dt>Live regret</dt><dd>${fmt(r2.validation.live_val_regret ?? r2.validation.base_val_regret)}</dd></dl><p class="note">A merge must beat what is live, not just the released base model.</p></div></div>
  <div class="grid k2" style="margin-top:16px"><div class="card"><h3>Audit log</h3><p>${sw.log_entries} entries · hash chain and HMAC ${sw.log_verifies ? '<span class="badge ok">VERIFIED</span>' : '<span class="badge bad">BROKEN</span>'}</p><p class="note">Append-only, hash-chained, HMAC-signed. The papers use a permissioned blockchain; one site does not need adversarial consensus.</p></div>
  <div class="card"><h3>Reputation</h3>${table(["Car", "Score", "Trusted"], Object.entries(sw.reputation).map(([k, v]) => [esc(k), fmt(v, 2), v >= 0.5 ? "yes" : '<span class="bad">no</span>']))}</div></div>`;
  const run = (B.closedloop_warehouse?.runs || []).find((r) => r.frames);
  if (run) {
    const first = run.frames.find((f) => f.quarantined.length);
    s += `<div class="callout">Warehouse shift: ${first ? `<b>${esc(first.quarantined.join(", "))}</b> quarantined from t = ${first.t} s (power above its contract cap)` : "no robot quarantined"}.</div>`;
  }
  return s;
}

function executionPage() {
  const c = B.smartcarcmd_contract;
  const runs = [...(B.closedloop_10car?.runs || []).map((r) => ["10-car", r]), ...(B.closedloop_warehouse?.runs || []).map((r) => ["warehouse", r])];
  return `<h1>Execution</h1><p class="lede">WirelessJEV only proposes; execution is a separate, supervised step through the existing on-car <code>SmartCarCmd.py</code>. The management hotspot radio (wlan0, 192.168.1.11) is never touched; only the BE200 client radio (wlan1) changes BSSID.</p>
  <div class="grid k2"><div class="card"><h3>Wire contract</h3><div class="sub">invocation over SSH (host key pinned)</div><pre>${esc(c.invocation)}</pre><pre>${esc(JSON.stringify(c.status))}</pre><pre>${esc(JSON.stringify(c.connect, null, 1))}</pre></div>
  <div class="card"><h3>Outcome semantics</h3>${table(["Outcome", "Meaning"], Object.entries(c.outcomes).map(([k, v]) => [`<b>${k}</b>`, esc(v)]))}<p class="note">A timeout after sending is never treated as “nothing happened” and a connect is never resent.</p></div></div>
  <div class="card" style="margin-top:16px"><h3>Transactions in the closed-loop runs</h3>${table(["Scenario", "Policy", "Outcomes", "Connect resends"], runs.map(([n, r]) => [n, esc(r.policy), esc(Object.entries(r.transactions).map(([k, v]) => `${k} ${v}`).join(" · ")), r.connects_beyond_one_per_txn]))}</div>
  <div class="card" style="margin-top:16px"><h3>One-car real test · <span class="badge">${esc(B.one_car_test.status.toUpperCase())}</span></h3>${table(["Step", "Command", "Changes the car?"], [
    ["0 Preflight", "<code>onecar preflight</code>", "no (offline)"], ["1 Status", "<code>onecar status --repeat 10</code>", "no"],
    ["2 Scan", "<code>onecar scan --seconds 20</code>", "no"], ["3 Propose", "<code>onecar propose</code>", "no (local)"],
    ["4 Switch", "<code>onecar switch</code> + typed BSSID", "<b>yes — one connect</b>"], ["5 Return", "<code>onecar switch</code> to original", "yes — separate, confirmed"]])}<p class="note">Plan: <code>${esc(B.one_car_test.plan)}</code>. ${B.one_car_test.logs.length ? "Logs: " + esc(B.one_car_test.logs.join(", ")) : "No real-car logs yet."}</p></div>`;
}

function researchPage() {
  return `<h1>Research map</h1><p class="lede">How each Bowen Shen / Feng Li / Kwok-Yan Lam paper is used. FT = full text read; Abs = abstract only. Learned components in the papers (DQN, PPO, GNN, LSTM, LLaMA-LoRA) are replaced by their formulation as an exact oracle, a frozen open model, or counting statistics.</p>
  <div class="card">${table(["Paper", "Read", "Mechanism used", "Where in the code"], B.papers.map((p) => [md(p.paper), esc(p.status), md(p.mechanism), md(p.code)]))}</div>`;
}

function dataPage() {
  const h = B.hardware;
  return `<h1>Hardware &amp; data</h1><p class="lede">Every robot is SmartCar hardware. This bundle was generated ${esc(B.generated_at)} from local artifacts.</p>
  <div class="grid k2"><div class="card"><h3>Robot hardware (SmartCar.zip)</h3><dl class="dl">${Object.entries(h).map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(Array.isArray(v) ? v.join(", ") : v)}</dd>`).join("")}</dl></div>
  <div class="card"><h3>Model</h3><dl class="dl"><dt>Decision model</dt><dd><code>${esc(B.model.repo)}</code></dd><dt>Pinned revision</dt><dd class="mono">${esc(B.model.revision)}</dd><dt>Backbone</dt><dd>frozen</dd><dt>Adapters</dt><dd>1024-parameter readout direction, distilled from exact oracles</dd></dl>
  <h3 style="margin-top:14px">Regenerate</h3><pre>uv run python -m wirelessjev.bench --scenes 80
uv run python -m wirelessjev.bench_warehouse --scenes 50
uv run python -m wirelessjev.closedloop --warehouse --total 600 \\
  --adapter artifacts/warehouse/adapters/combined \\
  --out artifacts/closedloop_warehouse.json
uv run python -m wirelessjev.studio_data</pre></div></div>
  <div class="card" style="margin-top:16px"><h3>Warehouse layout</h3><dl class="dl"><dt>Hall</dt><dd>${B.layout.width} × ${B.layout.height} m</dd><dt>Rack rows</dt><dd>${B.layout.rack_rows.length}, cross-aisles at x = ${B.layout.cross_aisles.join(", ")} m</dd><dt>Rack loss</dt><dd>${B.layout.rack_db["5g"]} dB (5 GHz) / ${B.layout.rack_db["2g"]} dB (2.4 GHz) per row crossed</dd><dt>AP sites</dt><dd>${B.layout.ap_sites.length} ceiling sites, dual-band</dd></dl></div>`;
}

// ------------------------------------------------------------------ router
const PAGES = { overview: ["Overview", overview], map: ["Fleet map", mapPage], decisions: ["Decisions", decisionsPage], evidence: ["Evidence", evidencePage], swarm: ["Swarm & trust", swarmPage], execution: ["Execution", executionPage], research: ["Research map", researchPage], data: ["Hardware & data", dataPage] };

function route() {
  if (player) { clearInterval(player); player = null; }
  const key = (location.hash || "#overview").slice(1);
  const [title, fn] = PAGES[key] || PAGES.overview;
  crumb.textContent = title;
  document.querySelectorAll(".rail a.nav").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + key));
  try { view.innerHTML = fn(); } catch (e) { view.innerHTML = `<div class="callout"><b>Could not render this view.</b> ${esc(e.message)}</div>`; console.error(e); }
  view.querySelectorAll("[data-go]").forEach((el) => (el.onclick = () => (location.hash = el.dataset.go)));
  window.scrollTo(0, 0);
}

document.getElementById("theme").onclick = () => {
  const r = document.documentElement, cur = r.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  r.dataset.theme = cur === "dark" ? "light" : "dark";
  route();
};

const embedded = document.getElementById("bundle");
const loadBundle = embedded
  ? Promise.resolve().then(() => JSON.parse(embedded.textContent))
  : fetch("data/bundle.json", { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`bundle.json: HTTP ${r.status}`); return r.json(); });
loadBundle.then((b) => {
  B = b;
  window.addEventListener("hashchange", route);
  route();
}).catch((e) => { view.innerHTML = `<div class="callout"><b>No data bundle.</b> Run <code>uv run python -m wirelessjev.studio_data</code> and serve <code>studio/</code> over HTTP. (${esc(e.message)})</div>`; });
