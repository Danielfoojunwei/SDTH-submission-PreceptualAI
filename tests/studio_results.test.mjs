import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../studio/app.js', import.meta.url), 'utf8');
const context = vm.createContext({document: {getElementById: () => ({})}});
vm.runInContext(source.slice(0, source.indexOf('const embedded =')) +
  '\nglobalThis.renderResearch = (bundle) => { B = bundle; return researchResults(); };', context);
const render = context.renderResearch;

test('missing results stay missing', () => {
  assert.equal(render({}), '');
});

test('E1 displays real confidence intervals and pending stages', () => {
  const lora = JSON.parse(readFileSync(new URL('../artifacts/radiojev/e1_radio_lora.json', import.meta.url)));
  const noLlm = JSON.parse(readFileSync(new URL('../artifacts/radiojev/e1_no_llm.json', import.meta.url)));
  const html = render({radiojev: {e1_radio_lora: lora, e1_no_llm: noLlm}});
  assert.match(html, /0\.093 ± 0\.022/);
  assert.match(html, /0\.074 ± 0\.020/);
  assert.match(html, /not evaluated/);
  assert.match(html, /pending/);
  assert.match(html, /not a paired E1 comparison/);
});

test('E7 distinguishes corrected and historical evidence', () => {
  const legacy = JSON.parse(readFileSync(new URL('../artifacts/envlearn/report.json', import.meta.url)));
  const oldHtml = render({envlearn: legacy});
  assert.match(oldHtml, /legacy pipeline; censoring defect known/);
  const corrected = {...legacy, method: 'measured-top8-censored-gaussian-v2'};
  const html = render({envlearn: corrected, envlearn_legacy: legacy});
  assert.match(html, /corrected scan censoring/);
  assert.match(html, /original report is retained/);
  assert.match(html, /not a proven global optimum/);
});

test('training snapshot distinguishes validation progress from completed test results', () => {
  const html = render({training_run: {status: 'running', seed: 0, stages: {
    e1_no_llm: {status: 'completed'}, e1_radio_lora: {status: 'running', history: [
      {epoch: 1, val_regret: 0.23}, {epoch: 2, val_regret: 0.1347}]},
    e2_radio_q: {status: 'completed'}, e3: {status: 'pending'}},
    results: {e1_no_llm: {test: {n: 200, regret: 0.001}},
      e2_radio_q: {test: {n: 120, norm_regret: 0.117}}}}});
  assert.match(html, /Full RadioJEV rerun/);
  assert.match(html, /0\.135/);
  assert.match(html, /0\.001 \(n=200\)/);
  assert.match(html, /0\.117 normalized regret \(n=120\)/);
  assert.match(html, /queued/);
  assert.match(html, /not a live monitor/);
  assert.match(html, /Validation progress is not a held-out test result/);
});

test('held-out calibration results retain their map-only scope', () => {
  const report = JSON.parse(readFileSync(new URL('../artifacts/envlearn_calibration/report.json', import.meta.url)));
  const html = render({envlearn_calibration: report});
  assert.equal(report.seeds.length, 15);
  assert.equal(report.map_only, true);
  assert.match(html, /15\.71/);
  assert.match(html, /14\.23/);
  assert.match(html, /map error only/);
  assert.match(html, /does not replace the default planner/);
});

test('completed E3 displays mixed results and the admission label audit', () => {
  const e3 = JSON.parse(readFileSync(new URL('../artifacts/radiojev_cloud/e3.json', import.meta.url)));
  const audit = JSON.parse(readFileSync(new URL('../artifacts/radiojev_cloud/e3_label_audit.json', import.meta.url)));
  const html = render({radiojev: {e3}, radiojev_audits: {e3: audit}});
  assert.match(html, /RadioJEV E3 · multitask result/);
  assert.match(html, /0\.105/);
  assert.match(html, /traffic accuracy remains below the rules/);
  assert.match(html, /one 500-step evaluation episode/);
  assert.match(html, /54 test cases/);
  assert.match(html, /does not establish admission-policy generalisation/);
});

test('completed E4/E5 shows measured results without certifying deletion', () => {
  const arm = {mean: 0.2, worst: 0.3, log_verifies: true};
  const e45 = {local: arm, fedavg_all: arm, swarm_pers: arm,
    e5: {forgotten_site: 'S2_heavy_scanners', mia_auc_before: 0.6,
      mia_auc_exact_retrain: 0.5, mia_auc_gated_distill: 0.52, mia_auc_never_trained: 0.5,
      retained_regret_before: 0.2, retained_regret_exact: 0.25, retained_regret_distill: 0.23}};
  const html = render({radiojev: {e45}});
  assert.match(html, /RadioJEV E4\/E5/);
  assert.match(html, /0\.520/);
  assert.match(html, /does not certify deletion/);
  assert.match(html, /no separate final release gate/);
});
