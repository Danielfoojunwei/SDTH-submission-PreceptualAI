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


def displayed_signed(value):
    """Match the UI's explicit sign convention, including IEEE negative zero."""
    return ('+' if value >= 0 else '-') + f'{abs(value):.3f}'


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


def open_details(page, table_id):
    section=page.locator('details').filter(has=page.locator('#'+table_id))
    section.locator('summary').first.click()
    return section


def main():
    data=json.loads((HERE/'data.json').read_bytes())
    provenance=json.loads((HERE/'provenance.json').read_bytes())
    if not (ROOT/'artifacts/nested_radio/verification/manifest.json').is_file():
        raise RuntimeError('A completed scientific seal must precede browser work')
    frozen={name:sha(ROOT/name) for name in provenance['inputs_sha256']}
    if frozen!=provenance['inputs_sha256']:
        raise RuntimeError('Scientific/presentation input identity changed')
    errors=[];console=[];network=[];assertions=[]
    with local_server() as origin,sync_playwright() as p:
        options={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
        executable=shutil.which('chromium') or shutil.which('chromium-browser')
        if executable:options['executable_path']=executable
        browser=p.chromium.launch(**options)
        page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('console',lambda msg:console.append(msg.text) if msg.type=='error' else None)
        page.on('request',lambda request:network.append(request.url) if request.url.startswith(('http:','https:')) and not request.url.startswith(origin+'/') else None)
        page.goto(origin+'/artifacts/nested_radio_review/viewer/index.html',wait_until='load')
        checked=page.evaluate('window.__viewerChecks()')
        assert checked['cases']==72 and checked['layouts']==24 and checked['controls']==9 and checked['fits']==15 and checked['nativeSimulations']==0
        for element,key in [('hier-rmse','selected'),('global-rmse','global8_reg'),('map-rmse','best_numerical')]:
            expected=f"{data['field']['summary']['sparse'][key]['rmse_db']:.3f}"
            assert page.locator('#'+element).inner_text().startswith(expected),(element,expected)
            assertions.append(element)
        rows=page.locator('#contrasts tr').all_inner_texts();assert len(rows)==4
        for expected,actual in zip(data['field']['primary_contrasts'],rows):
            assert all(f'{v:+.3f}' in actual for v in (expected['mean_delta'],*expected['central_98_75_interval']))
        agreement=next(row for row in data['field']['primary_contrasts']
            if row['candidate']=='nested_joint' and row['reference']=='nested_reg')
        assert f"{agreement['mean_delta']:.6f} dB" in page.locator('#agreement-reading').inner_text()
        if data['field']['selected_cell']=='global8_reg':
            assert 'Exact alias' in rows[1] and data['field']['primary_contrasts'][1]['central_98_75_interval']==[0,0]
        assertions.append('all-four-primary-contrasts-and-exact-alias-semantics')
        details=open_details(page,'all-controls')
        rows=page.locator('#all-controls tr').all_inner_texts();assert len(rows)==9
        for key,actual in zip(data['control_order'],rows):
            m=data['field']['summary']['sparse'][key]
            values=[f"{m['rmse_db']:.3f} dB",f"{data['field']['summary']['full'][key]['rmse_db']:.3f} dB",
                f"{data['field']['localized_delta_descriptive'][key]['mean_delta_RMSE_db']:.3f} dB",
                f"{m['p95_abs_error_db']:.3f} dB",f"{m['heard_interval_95_coverage']*100:.2f}%"]
            assert all(v in actual for v in values),(key,actual)
        details.locator('summary').first.click();assertions.append('all-nine-control-means-tails-and-coverage')
        details=open_details(page,'secondary')
        stress=[data['field']['summary']['full'],*[data['field']['secondary_descriptive'][key] for key in
            ('test-sparse-held6GHz','test-sparse-stale','ap_transfer-sparse','ap_transfer-full')]]
        rows=page.locator('#secondary tr').all_inner_texts();assert len(rows)==45
        for i,metrics in enumerate(stress):
            for j,key in enumerate(data['control_order']):
                m=metrics[key];actual=rows[i*9+j]
                assert all(v in actual for v in (f"{m['rmse_db']:.3f} dB",f"{m['p95_abs_error_db']:.3f} dB",f"{m['heard_interval_95_coverage']*100:.2f}%"))
        details.locator('summary').first.click();assertions.append('all-five-stress-assays-times-nine-controls')
        coverage=page.evaluate('''() => {
            const failures=[];let n=0;const select=(id,value,emit=true)=>{document.getElementById(id).value=String(value);if(emit)document.getElementById(id).dispatchEvent(new Event('change'));};
            for(let layout=0;layout<24;layout++)for(let state=0;state<3;state++)for(let ap=0;ap<4;ap++)for(let band=0;band<2;band++){
              select('layout',layout,false);select('state',state,false);select('ap',ap,false);select('band',band);
              const c=DATA.cases[layout*3+state],check=window.__viewerChecks();
              if(check.selectedCase!==c.id||check.selectedAP!==ap||check.selectedBand!==band)failures.push(c.id+' selection');
              const heard=c.ap.map((_,i)=>i).filter(i=>c.ap[i]===ap&&c.band[i]===band&&c.heard[i]);
              for(const method of ['hier','map']){const text=document.getElementById(method+'-scene-error').textContent,err=heard.map(i=>c[method+'_mean'][i]-c.reported_center[i]);
                if(err.length){const rmse=Math.sqrt(err.reduce((a,b)=>a+b*b,0)/err.length).toFixed(2);if(!text.startsWith(rmse+' dB'))failures.push(c.id+' '+method+' RMSE');}
                else if(!text.startsWith('No heard'))failures.push(c.id+' no heard');}
              const cens=c.ap.map((_,i)=>i).filter(i=>c.ap[i]===ap&&c.band[i]===band&&c.censored[i]).length;
              if(!document.getElementById('event-count').textContent.includes(heard.length+' heard · '+cens+' upper-censored'))failures.push(c.id+' report events');n++;}
            return {combinations:n,failures};}''')
        assert coverage=={'combinations':576,'failures':[]},coverage
        page.locator('#scene-hier').focus();before=page.evaluate('angle');page.keyboard.press('ArrowRight')
        assert abs(page.evaluate('angle')-before-.1)<1e-10
        page.locator('#reset').click();page.locator('#scene-hier').scroll_into_view_if_needed()
        box=page.locator('#scene-hier').bounding_box();x,y=box['x']+box['width']/2,box['y']+box['height']/2
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+30,y+10);page.mouse.up()
        assert abs(page.evaluate('angle')+.72)>.01
        page.locator('#reset').click();assertions.append('pointer-and-keyboard-shared-rotation')
        titles=page.locator('#budget-chart circle title').all_text_contents();assert len(titles)==20
        for i,cell in enumerate(('global8_reg','global8_joint','hier_reg','nested_reg','nested_joint')):
            for j,epoch in enumerate((40,80,120,160)):
                mean=sum(next(v['rmse'] for v in row['checks'] if v['epoch']==epoch) for row in data['fits'] if row['cell']==cell)/3
                assert f'{mean:.6f} dB' in titles[i*4+j]
        rows=page.locator('#cell-selection tr').all_inner_texts();assert len(rows)==4
        for cell,actual in zip(('global8_reg','global8_joint','nested_reg','nested_joint'),rows):
            assert f"{data['selection']['mean_VAL_RMSE'][cell]:.6f} dB" in actual
            assert ('Selected' in actual)==(cell==data['field']['selected_cell'])
        rows=page.locator('#numeric-selection tr').all_inner_texts();assert len(rows)==4
        for key,actual in zip(('radiomap-5m','radiomap-10m','radiomap-20m','hierarchy-fixed'),rows):
            assert f"{data['numerical']['scores'][key]:.6f} dB" in actual
            assert ('Selected' in actual)==(key==data['field']['best_numerical'])
        details=open_details(page,'checkpoints');rows=page.locator('#checkpoints tr').all_inner_texts();assert len(rows)==15
        for row,actual in zip(data['fits'],rows):
            assert all(v in actual for v in (row['member'],str(row['selected_epoch']),f"{row['validation_rmse']:.6f} dB",f"{row['validation_nll']:.6f}",*[f"{v['rmse']:.6f} dB" for v in row['checks']]))
        details.locator('summary').first.click();assertions.append('all-sixty-budget-checkpoints-and-VAL-only-selection')
        diagnostic_rows=0;product_rows=0
        for row in data['representation']['rows']:
            page.locator('#diagnostic-member').select_option(row['member'])
            d=row['gates_and_effective_priors'];active=row['learned_local_features_used_in_this_cell']
            status=page.locator('#gate-status').inner_text()
            assert ('inactive by design' in status)==(not active)
            gate_text=page.locator('#gate-bars .mini-row b').all_inner_texts();rms_text=page.locator('#rms-bars .mini-row b').all_inner_texts()
            assert gate_text==[f'{v:.3f}' for v in d['gates']]
            assert rms_text==[f'{v:.3f} dB' for v in d['effective_gated_prior_rms_db']]
            ranks=page.locator('#ranks tr').all_inner_texts();assert len(ranks)==3
            for key,actual in zip(('fixed_Fourier8','learned_geometry8','effective_local16'),ranks):
                assert f"{row[key]['participation_rank_centered']:.4f}" in actual
                if not active:assert 'Unused in global-eight prediction' in actual
            details=open_details(page,'gate-products');products=page.locator('#gate-products tr').all_inner_texts();assert len(products)==16
            for i,actual in enumerate(products):
                assert all(v in actual for v in (f"{d['gates'][i]:.6f}",displayed_signed(d['effective_gated_prior_mean_db'][i]),
                    f"{d['effective_gated_prior_std_db'][i]:.6f}",f"{d['effective_gated_prior_rms_db'][i]:.6f}"))
            details.locator('summary').first.click();diagnostic_rows+=1;product_rows+=16
        assertions.append('all-fifteen-member-effective-prior-and-inactive-kernel-diagnostics')
        for id in ('layout','state','ap','band'):page.locator('#'+id).select_option('0')
        page.locator('#diagnostic-member').select_option(data['selected_member'])
        page.screenshot(path=str(HERE/'desktop.png'),full_page=True)
        page.locator('#scene-hier').scroll_into_view_if_needed();page.screenshot(path=str(HERE/'geometry.png'))
        page.set_viewport_size({'width':390,'height':844});page.locator('#colour').select_option('mean')
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'Mobile document overflows horizontally'
        page.screenshot(path=str(HERE/'mobile.png'),full_page=True)
        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(HERE/'mobile_hero.png'))
        page.locator('#scene-hier').scroll_into_view_if_needed();page.screenshot(path=str(HERE/'mobile_geometry.png'))
        browser.close()
    if errors or console or network:raise AssertionError({'page_errors':errors,'console_errors':console,'external_requests':network})
    after={name:sha(ROOT/name) for name in frozen}
    if frozen!=after:raise RuntimeError('Browser work changed a scientific input')
    result={'schema':1,'passes':True,'browser':'Playwright Chromium','assertions':assertions,
        'geometry_AP_band_combinations':576,'primary_contrasts':4,'RF_control_means_checked':9,
        'descriptive_stress_control_rows_checked':45,'fixed_epoch_curve_points_checked':20,
        'fit_members_checked':15,'checkpoint_scores_checked':60,'gate_members_checked':diagnostic_rows,
        'gated_prior_product_rows_checked':product_rows,'native_simulations':0,
        'desktop_viewport':[1440,1000],'mobile_viewport':[390,844],'page_errors':errors,'console_errors':console,
        'external_requests':network,'served_only_over_ephemeral_loopback':True,'scientific_calls':0,
        'scientific_input_hashes_unchanged':True,'input_files_checked':len(frozen),
        'provenance_sha256':sha(HERE/'provenance.json'),
        'scientific_seal_sha256':provenance['inputs_sha256']['artifacts/nested_radio/verification/manifest.json'],
        'browser_verifier_sha256':sha(__file__),
        'generator_sha256':provenance['generator_sha256'],'index_sha256':sha(HERE/'index.html'),
        'screenshots_sha256':{name:sha(HERE/name) for name in
            ('desktop.png','geometry.png','mobile.png','mobile_hero.png','mobile_geometry.png')}}
    with (HERE/'browser_review.json').open('x') as handle:
        json.dump(result,handle,sort_keys=True,indent=2);handle.write('\n')
    print(json.dumps({'passes':True,'combinations':576,'control_means':9,'checkpoints':60,'gated_products':product_rows,'page_errors':0}))


if __name__=='__main__':main()
