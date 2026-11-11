#!/usr/bin/env python3
"""Read-only browser checks of the generated presentation, never research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def local_server():
    """Managed Chromium blocks file URLs; serve unchanged repo bytes locally."""
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            if self.path == '/favicon.ico':
                self.send_response(204);self.end_headers()
            else:
                super().do_GET()
    server = ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown();server.server_close();thread.join()


def main():
    data = json.loads((HERE/'data.json').read_bytes())
    provenance = json.loads((HERE/'provenance.json').read_bytes())
    frozen = {name: sha(ROOT/name) for name in provenance['inputs_sha256']}
    if frozen != provenance['inputs_sha256']:
        raise RuntimeError('Presentation input identity changed before browser check')
    errors = []; console_errors = []; network = []; assertions = []
    with local_server() as origin, sync_playwright() as p:
        options = {'headless': True, 'args': ['--no-sandbox', '--disable-dev-shm-usage']}
        executable = shutil.which('chromium') or shutil.which('chromium-browser')
        if executable:
            options['executable_path'] = executable
        browser = p.chromium.launch(**options)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: console_errors.append(message.text) if message.type=='error' else None)
        page.on('request', lambda request: network.append(request.url) if request.url.startswith(('http:', 'https:')) and not request.url.startswith(origin+'/') else None)
        page.goto(origin+'/artifacts/hierarchical_radio_review/viewer/index.html', wait_until='load')
        if page.evaluate('window.__viewerChecks()')['cases'] != 72:
            raise AssertionError('All72 actual case views required')
        for element, key in [('hier-rmse', 'selected_hierarchy'), ('global-rmse', 'global_reg'), ('map-rmse', 'radiomap_operator')]:
            expected = f"{data['field']['summary']['sparse'][key]['rmse_db']:.3f}"
            if not page.locator('#'+element).inner_text().startswith(expected):
                raise AssertionError('Wrong primary metric: '+element)
            assertions.append(element)
        contrast_rows = page.locator('#contrasts tr').all_inner_texts()
        if len(contrast_rows) != 4:
            raise AssertionError('All four predeclared contrasts required')
        for expected, actual in zip(data['field']['primary_contrasts'], contrast_rows):
            values = [expected['mean_delta'], *expected['central_98_75_interval']]
            if any(f'{value:+.3f}' not in actual for value in values):
                raise AssertionError('Displayed contrast differs from the completed report')
        assertions.append('all-four-primary-contrasts')
        page.locator('details summary').first.click()
        stress = [data['field']['summary']['full'], *[data['field']['secondary_descriptive'][key] for key in
            ('test-sparse-held6GHz', 'test-sparse-stale', 'ap_transfer-sparse', 'ap_transfer-full')]]
        for source, actual in zip(stress, page.locator('#secondary tr').all_inner_texts()):
            if any(f"{source[key]['rmse_db']:.3f} dB" not in actual for key in ('selected_hierarchy','global_reg','radiomap_operator')):
                raise AssertionError('Descriptive stress value differs from report')
        assertions.append('all-five-descriptive-stress-assays')
        page.locator('details summary').first.click()
        coverage = page.evaluate('''() => {
          const failures=[];let n=0;
          const select=(id,value,emit=true)=>{document.getElementById(id).value=String(value);if(emit)document.getElementById(id).dispatchEvent(new Event('change'));};
          for(let layout=0;layout<24;layout++)for(let state=0;state<3;state++)for(let ap=0;ap<4;ap++)for(let band=0;band<2;band++){
            select('layout',layout,false);select('state',state,false);select('ap',ap,false);select('band',band);
            const c=DATA.cases[layout*3+state], check=window.__viewerChecks();
            if(check.selectedCase!==c.id||check.selectedAP!==ap||check.selectedBand!==band)failures.push(c.id+' selection');
            const heard=c.ap.map((_,i)=>i).filter(i=>c.ap[i]===ap&&c.band[i]===band&&c.heard[i]);
            for(const method of ['hier','map']){
              const text=document.getElementById(method+'-scene-error').textContent;
              const err=heard.map(i=>c[method+'_mean'][i]-c.reported_center[i]);
              if(err.length){const rmse=Math.sqrt(err.reduce((a,b)=>a+b*b,0)/err.length).toFixed(2);if(!text.startsWith(rmse+' dB'))failures.push(c.id+' '+method+' RMSE');}
              else if(!text.startsWith('No heard'))failures.push(c.id+' missing heard');
            }
            const counts=document.getElementById('event-count').textContent;
            const cens=c.ap.map((_,i)=>i).filter(i=>c.ap[i]===ap&&c.band[i]===band&&c.censored[i]).length;
            if(!counts.includes(heard.length+' heard · '+cens+' upper-censored'))failures.push(c.id+' events');
            n++;
          }
          return {combinations:n,failures};
        }''')
        if coverage != {'combinations': 576, 'failures': []}:
            raise AssertionError(str(coverage))
        page.locator('#scene-hier').focus()
        before = page.evaluate('angle')
        page.keyboard.press('ArrowRight')
        if abs(page.evaluate('angle') - before - .1) > 1e-10:
            raise AssertionError('Keyboard rotation did not update the shared view')
        page.locator('#reset').click()
        page.locator('#scene-hier').scroll_into_view_if_needed()
        box = page.locator('#scene-hier').bounding_box()
        x, y = box['x']+box['width']/2, box['y']+box['height']/2
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+30,y+10);page.mouse.up()
        if abs(page.evaluate('angle') + .72) < .01:
            raise AssertionError('Pointer rotation did not update the shared view')
        page.locator('#reset').click()
        assertions.append('keyboard-and-pointer-rotation')
        for ho in (.03, 4.):
            page.locator(f'[data-ho="{ho:g}"]').click()
            labels = page.locator('#native-bars .bar-label').all_inner_texts()
            expected = [f"{row['mean']['control_miss']*100:.4f}%" for row in data['native']['method_means'] if row['handover_seconds'] == ho]
            if len(labels) != 4 or any(value not in actual for value, actual in zip(expected, labels)):
                raise AssertionError('Native handover values differ from report')
            costs = page.locator('#native-costs tr').all_inner_texts()
            methods = [row for row in data['native']['method_means'] if row['handover_seconds'] == ho]
            for row, actual in zip(methods, costs):
                expected = [f"{row['mean']['maximum_silence_ms']/1000:.3f} s",
                    f"{row['mean']['paid_forecast_compute_seconds']:.3f} s"]
                if row['method'] != 'full_radiomap':
                    episodes = [r for r in data['native']['rows'] if r['handover_seconds'] == ho and r['method'] == row['method']]
                    harms = sum(r['metrics']['control_misses'] > next(s['metrics']['control_misses'] for s in data['native']['rows']
                        if s['seed'] == r['seed'] and s['handover_seconds'] == ho and s['method'] == 'full_radiomap') for r in episodes)
                    expected.append(f'{harms} / 4')
                if any(value not in actual for value in expected):
                    raise AssertionError('Displayed native cost/tail/harm differs from report')
            page.locator('details summary').last.click()
            episodes = [row for row in data['native']['rows'] if row['handover_seconds'] == ho]
            actual_rows = page.locator('#native-episodes tr').all_inner_texts()
            if len(episodes) != 16 or len(actual_rows) != 16:
                raise AssertionError('Every paired episode outcome must remain visible')
            for row, actual in zip(episodes, actual_rows):
                m = row['metrics']
                expected = [str(row['seed']), f"{m['control_miss']*100:.4f}%",
                    f"{m['maximum_silence_ms']/1000:.3f} s", f"{m['observed_plus_pending_miss_fraction']*100:.4f}%",
                    f"{m['scan_interruption_seconds']:.3f} s", f"{m['paid_forecast_compute_seconds']:.3f} s"]
                if any(value not in actual for value in expected):
                    raise AssertionError('Episode tail/cost/pending value differs from the recorded row')
            page.locator('details summary').last.click()
            assertions.append('native-'+str(ho))
        page.locator('#layout').select_option('0');page.locator('#state').select_option('0');page.locator('#ap').select_option('0');page.locator('#band').select_option('0')
        page.locator('[data-ho="0.03"]').click()
        page.screenshot(path=str(HERE/'desktop.png'), full_page=True)
        page.locator('#scene-hier').scroll_into_view_if_needed()
        page.screenshot(path=str(HERE/'geometry.png'))
        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#colour').select_option('mean')
        if page.evaluate('document.documentElement.scrollWidth > innerWidth + 1'):
            raise AssertionError('Mobile document overflows horizontally')
        page.screenshot(path=str(HERE/'mobile.png'), full_page=True)
        browser.close()
    if errors or console_errors or network:
        raise AssertionError({'page_errors': errors, 'console_errors': console_errors, 'external_requests': network})
    after = {name: sha(ROOT/name) for name in frozen}
    if frozen != after:
        raise RuntimeError('Browser presentation changed a scientific input')
    receipt = {'schema': 1, 'passes': True, 'browser': 'Playwright Chromium',
        'geometry_AP_band_combinations': coverage['combinations'], 'assertions': assertions,
        'primary_contrasts': 4, 'native_handover_values_checked': 8,
        'native_tail_cost_harm_rows_checked': 8,
        'native_individual_episode_rows_checked': 32,
        'desktop_viewport': [1440,1000], 'mobile_viewport': [390,844],
        'page_errors': errors, 'console_errors': console_errors, 'external_requests': network,
        'served_only_over_ephemeral_loopback': True,
        'scientific_calls': 0, 'scientific_input_hashes_unchanged': frozen == after,
        'browser_verifier_sha256': sha(__file__),
        'generator_sha256': provenance['generator_sha256'], 'index_sha256': sha(HERE/'index.html'),
        'screenshots_sha256': {name: sha(HERE/name) for name in ('desktop.png','geometry.png','mobile.png')}}
    with (HERE/'browser_review.json').open('x') as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'passes': True, 'combinations': 576, 'page_errors': 0, 'external_requests': 0}))


if __name__ == '__main__':
    main()
