"""Build the concise ten-slide deck with actual Blender-rendered 3D models.

Text, metrics, callouts and notes remain editable in PowerPoint. The rendered
models are schematic, not photographs or measured RF traces. Performance
numbers are computed from the archived campaign, never the scene geometry.
"""
from __future__ import annotations
import argparse,base64,hashlib,importlib.util,json,os,subprocess,zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from PIL import Image
import pymupdf
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/slides/concise';OUT=BASE/'generated';NAME='wirelessjev_10_slides'
BG='#081923';INK='#F0F3EB';MUTED='#A6BCC2';TEAL='#45D9C5';AMBER='#F7AA67';RED='#FF9486'
W,H=16.,9.


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rgb(hexcode):return tuple(int(hexcode[i:i+2],16)/255 for i in [1,3,5])


def helpers():
    spec=importlib.util.spec_from_file_location('research_deck_exports',ROOT/'scripts/build_research_deck.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.BG=BG;m.INK=INK;return m


def text(value,x,y,w,size=18,colour=INK,bold=False,leading=1.2):
    font='DeckBold' if bold else 'DeckSans'
    rows=[]
    for paragraph in value.split('\n'):
        row=''
        for word in paragraph.split():
            candidate=(row+' '+word).strip()
            if row and pdfmetrics.stringWidth(candidate,font,size)>w*72:rows.append(row);row=word
            else:row=candidate
        rows.append(row)
    if any(pdfmetrics.stringWidth(r,font,size)>w*72 for r in rows):raise ValueError('Word overflow: '+value)
    return {'lines':rows,'x':x,'y':y,'w':w,'size':size,'font':font,'colour':colour,'leading':leading,'h':len(rows)*size*leading/72+.05}


def elements(s,i,data,model):
    """One visual hierarchy; numbers and relationships stay native, editable text."""
    blocks=[text(s['kicker'],.72,.52,14,10,TEAL,True),text(f'{i:02d} / 10',14.35,.52,.95,10,MUTED)]
    rules=[]
    footer='SIMULATION EVIDENCE  /  SCHEMATIC 3D MODEL'
    if s['id'] in {'mission','architecture','stack','validation'}:footer='IMPLEMENTED PROTOTYPE  /  SCHEMATIC 3D MODEL  /  FIELD QUALIFICATION PENDING'
    blocks.extend([text('PRECEPTUAL AI   ·   RADIOMAP + WIRELESSJEV',.72,8.52,7.5,8.5,MUTED),text(footer,8.,8.52,7.3,7.3,MUTED)])
    if s['layout']=='left':
        size=43.
        while max(pdfmetrics.stringWidth(line,'DeckBold',size) for line in s['title'].split('\n'))>5.15*72:size-=.5
        title=text(s['title'],.72,1.36,5.2,size,INK,True,1.1);blocks.append(title)
        by=title['y']+title['h']+.36
        blocks.append(text(s['body'],.76,by,4.85,16.5,MUTED,False,1.42))
        for k,label in enumerate(s['labels']):
            y=6.12+k*.45
            rules.append((.77,y+.12,.98,y+.12,AMBER if k==2 else TEAL,1.7))
            blocks.append(text(label,1.13,y,4.5,11.5,INK if k<2 else MUTED))
        calls={
            'architecture':[('decide','Mission → policy',10.75,1.50),('learn','Mean + uncertainty',6.25,4.00),('recover','Queues → retries → observe',10.45,7.27)],
            'radiomap':[('map_peak','Predicted RSSI · 5 GHz',10.25,1.62)],
            'mission':[('knobs','Gain · load · video share',9.2,1.65),('focus_robot','Protect amr-017',10.65,6.90)],
            'gate':[('planned','Planned action',11.5,3.85),('reactive','Reactive fallback',10.6,7.05)],
            'diversity':[('ap_a','AP A · blocked',6.65,1.8),('ap_b','AP B · independent path',11.0,2.05),('robot','Two client radios',10.9,7.05)],
            'stack':[('layer_3','APPLICATION\nDeadlines',13.5,1.6),('layer_2','QUEUES\nTraffic share',13.5,3.0),('layer_1','LINKS\nMap + retries',13.5,4.4),('layer_0','ATTACHMENT\nAP paths',13.5,5.8)],
            'validation':[('stage_0','Probe',6.5,2.8),('stage_1','Time',8.4,3.6),('stage_2','Calibrate',10.2,4.6),('stage_3','Qualify',12.4,5.5)]
        }
        for key,label,x,y in calls.get(s['id'],[]):
            px,py=model['markers'][key];px*=W;py*=H
            width=min(4.1,15.5-x);b=text(label,x,y,width,10.5,AMBER if key in {'ap_a','layer_3','planned','decide'} else TEAL,True)
            blocks.append(b);rules.append((px,py,x,y+b['h']+.09,MUTED,.5))
    elif s['layout']=='results':
        blocks.extend([text(s['title'],.72,1.05,14.8,41,INK,True),text(s['body'],.77,2.00,14.4,17,MUTED)])
        for x,arm,label in [(1.2,'vertical_client','TWO AP PATHS'),(8.8,'nr_30ms','CELLULAR MODEL · 30 ms HANDOVER')]:
            ppm=data['results']['main'][arm]['ppm'];blocks.append(text(label,x,6.00,6.,11.5,TEAL if arm=='vertical_client' else AMBER,True))
            blocks.append(text(f'{ppm:.1f}',x,6.33,4.0,65,INK,True))
            blocks.append(text('misses per million commands',x,7.43,6.,12,MUTED))
        blocks.append(text('6 paired scenes  ·  3.6M commands per arm  ·  modelled second client radio',.78,8.02,14.5,11,MUTED))
    else:
        blocks.extend([text(s['title'],.72,1.05,14.8,41,INK,True),text(s['body'],.77,2.00,14.5,15.8,MUTED)])
        for x,key,label in [(1.0,'lenient','CLEAN RADIO'),(6.15,'main','MAIN CONDITION'),(11.3,'stress','STRESS')]:
            ppm=data['results'][key]['vertical_client']['ppm'];c=RED if key=='stress' else TEAL
            blocks.extend([text(label,x,6.0,4.5,11.5,c,True),text(f'{ppm:.1f}',x,6.3,4.5,53,INK,True),text('Wi-Fi misses / million',x,7.22,4.5,11.5,MUTED)])
        blocks.append(text('Cellular model: 52.8 misses / million in all three conditions  ·  Wi-Fi stress reverses the ranking',.78,8.0,14.7,11.1,AMBER))
    for b in blocks:
        if b['y']+b['h']>8.9:raise ValueError('Text below slide: '+str(b))
    return blocks,rules


def pptx_text(slide,b):
    shape=slide.shapes.add_textbox(Inches(b['x']),Inches(b['y']),Inches(b['w']),Inches(b['h']))
    tf=shape.text_frame;tf.clear();tf.word_wrap=False;tf.vertical_anchor=MSO_ANCHOR.TOP
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
    for j,line in enumerate(b['lines']):
        p=tf.paragraphs[0] if j==0 else tf.add_paragraph();p.text=line;p.font.name='DejaVu Sans';p.font.size=Pt(b['size']);p.font.bold=b['font']=='DeckBold';p.font.color.rgb=RGBColor.from_string(b['colour'][1:]);p.space_before=p.space_after=Pt(0);p.line_spacing=Pt(b['size']*b['leading'])


def build(source,data,out):
    prs=Presentation();prs.slide_width=Inches(W);prs.slide_height=Inches(H)
    prs.core_properties.title='RadioMap + WirelessJEV — ten-slide visual system';prs.core_properties.author='Preceptual AI'
    pdf=canvas.Canvas(str(out/f'{NAME}.pdf'),pagesize=(W*72,H*72));pdf.setTitle(prs.core_properties.title);pdf.setAuthor('Preceptual AI')
    for i,s in enumerate(source['slides'],1):
        slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string(BG[1:])
        pdf.setFillColorRGB(*rgb(BG));pdf.rect(0,0,W*72,H*72,fill=1,stroke=0)
        # Preserve lossless RGBA render masters; compact composites keep the deck practical to share.
        master=Image.open(out/f'{s["id"]}.png').convert('RGBA');composite=Image.new('RGB',master.size,BG);composite.paste(master,mask=master.getchannel('A'))
        dest=out/'composites';dest.mkdir(exist_ok=True);path=dest/f'{s["id"]}.jpg';composite.save(path,quality=94,subsampling=0)
        slide.shapes.add_picture(str(path),0,0,width=Inches(W),height=Inches(H));pdf.drawImage(str(path),0,0,W*72,H*72)
        model=json.loads((out/f'{s["id"]}.model.json').read_text());blocks,rules=elements(s,i,data,model)
        from pptx.enum.shapes import MSO_CONNECTOR
        for x1,y1,x2,y2,c,width in rules:
            connector=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));connector.line.color.rgb=RGBColor.from_string(c[1:]);connector.line.width=Pt(width)
            pdf.setStrokeColorRGB(*rgb(c));pdf.setLineWidth(width);pdf.line(x1*72,(H-y1)*72,x2*72,(H-y2)*72)
        for b in blocks:
            pptx_text(slide,b);pdf.setFont(b['font'],b['size']);pdf.setFillColorRGB(*rgb(b['colour']))
            for j,line in enumerate(b['lines']):pdf.drawString(b['x']*72,(H-b['y'])*72-b['size']-j*b['size']*b['leading'],line)
        slide.notes_slide.notes_text_frame.text=s['notes']+'\n\nSOURCES:\n'+'\n'.join(s['sources'])
        pdf.showPage()
    prs.save(out/f'{NAME}.pptx');pdf.save()


def validate(source,out):
    prs=Presentation(out/f'{NAME}.pptx');document=pymupdf.open(out/f'{NAME}.pdf');seen=set();checks=[]
    assert len(prs.slides)==document.page_count==len(source['slides'])==10
    for s,slide,page in zip(source['slides'],prs.slides,document):
        images=[sh for sh in slide.shapes if sh.shape_type==MSO_SHAPE_TYPE.PICTURE];assert len(images)==1
        seen.add(hashlib.sha256(images[0].image.blob).hexdigest());assert s['notes'] in slide.notes_slide.notes_text_frame.text
        assert ' '.join(s['title'].split()) in ' '.join(page.get_text().split())
        model=json.loads((out/f'{s["id"]}.model.json').read_text());assert model['mesh_objects']>=20 and model['schematic_3d_model'] and not model['measured_hardware']
        assert model['width']>=2304 and model['samples']>=48
        checks.append({'id':s['id'],'real_3d_geometry':True,'mesh_objects':model['mesh_objects'],'editable_text_and_notes':True})
    document.close();assert len(seen)==10;return checks


def source_hashes(source):
    paths={Path(__file__),BASE/'deck.json',BASE/'render_models.py',BASE/'assets/scene_data.json',ROOT/'scripts/prepare_visual_deck_data.py',ROOT/'scripts/build_research_deck.py'}
    paths.update(p for p in (BASE/'vendor').iterdir() if p.is_file())
    for s in source['slides']:paths.update(ROOT/p for p in s['sources'])
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def model_viewer(out):
    """An offline GLB explorer: real geometry, drag to orbit, scroll to zoom."""
    vendor=BASE/'vendor/model-viewer.min.js'
    if not vendor.exists():raise FileNotFoundError('Vendor the pinned Apache-2.0 model-viewer 4.0.0 bundle first')
    models={name:'data:model/gltf-binary;base64,'+base64.b64encode((out/'models'/f'{name}.glb').read_bytes()).decode() for name in ['warehouse','architecture','diversity']}
    module=vendor.read_text().replace('</script','<\\/script')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WirelessJEV · 3D model explorer</title><style>*{box-sizing:border-box}body{margin:0;background:#081923;color:#f0f3eb;font:16px system-ui}header{padding:22px 28px}h1{font-size:24px;margin:0 0 10px}p{color:#a6bcc2;margin:4px 0}nav{display:flex;gap:10px;flex-wrap:wrap;padding:0 28px}button{background:#142e3c;border:1px solid #45d9c5;border-radius:18px;padding:10px 18px;color:white;cursor:pointer}model-viewer{width:100%;height:calc(100vh - 190px);min-height:400px;--progress-bar-color:#45d9c5}footer{padding:0 28px 18px;font-size:12px;color:#a6bcc2}</style><header><h1>Explore the system in 3D</h1><p>Drag to orbit · scroll / pinch to zoom · double tap to reset</p></header><nav><button data-model="warehouse">Warehouse</button><button data-model="architecture">Exploded architecture</button><button data-model="diversity">Cross-AP paths</button></nav><model-viewer id="model" camera-controls auto-rotate shadow-intensity="1" exposure="1.4" environment-image="neutral" interaction-prompt="auto" alt="Schematic 3D model of the RadioMap and WirelessJEV system"></model-viewer><footer>Conceptual models, not field measurements. Apache-2.0 model-viewer 4.0.0, license included with assets.</footer><script type="module">VENDOR_MODULE</script><script>const models=MODEL_DATA;const m=document.getElementById('model');document.querySelectorAll('button').forEach(b=>b.onclick=()=>{m.src=models[b.dataset.model];m.setAttribute('alt',b.textContent+' — schematic model')});m.src=models.warehouse;</script></html>'''
    html=html.replace(' · double tap to reset','').replace('camera-controls auto-rotate','camera-controls auto-rotate camera-orbit="35deg 55deg auto"')
    licenses='\n\n'.join((BASE/'vendor'/p).read_text() for p in ['LICENSE','lit.LICENSE','three.LICENSE','gainmap.LICENSE'])
    html=html.replace('</html>','<script type="text/plain" id="third-party-licenses">'+licenses+'</script></html>')
    (out/'explore_3d.html').write_text(html.replace('VENDOR_MODULE',module).replace('MODEL_DATA',json.dumps(models)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);source=json.loads((BASE/'deck.json').read_text());data=json.loads((BASE/'assets/scene_data.json').read_text())
    if args.render:
        env=os.environ.copy();env['BLENDER_USER_CONFIG']=str(ROOT.parent/'.cache/blender/config');env['BLENDER_USER_EXTENSIONS']=str(ROOT.parent/'.cache/blender/extensions')
        subprocess.run(['blender','--background','--factory-startup','--threads','4','--python',str(BASE/'render_models.py'),'--','--out',str(OUT),'--samples','64'],check=True,env=env)
    h=helpers();font,_=h.register_fonts();build(source,data,OUT)
    # Shared exports are pure helpers; the 32-slide evidence appendix is untouched.
    preview_source={**source,'slides':[{**s,'title':s['title'].replace('\n',' '),'insight':s['body'],'evidence':'Simulation evidence / schematic 3D model'} for s in source['slides']]}
    paths=h.previews(OUT/f'{NAME}.pdf',preview_source,OUT,font);old=h.build_html(preview_source,paths,OUT);old.rename(OUT/f'{NAME}.html')
    html=(OUT/f'{NAME}.html').read_text().replace('System Architecture','10-slide Visual System').replace('#152E3D',BG).replace('#152e3d',BG)
    (OUT/f'{NAME}.html').write_text(html);model_viewer(OUT)
    notes=['# RadioMap + WirelessJEV: ten-slide presenter notes','', 'Schematic 3D models; all reported performance comes from simulation. No physical fleet is qualified.','']
    for i,s in enumerate(source['slides'],1):notes.extend([f'## {i}. '+s['title'].replace('\n',' '),'',s['notes'],'','Sources: '+', '.join('`'+p+'`' for p in s['sources']),''])
    (BASE/'NOTES.md').write_text('\n'.join(notes));checks=validate(source,OUT)
    excluded={'manifest.json','wirelessjev_3d_assets.zip','browser_check.json','browser_preview.png','model_browser_check.json','model_browser_preview.png'}
    outputs={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded and not p.name.endswith('.blend1')}
    manifest={'title':source['title'],'base_commit':source['base_commit'],'built_at':datetime.now(ZoneInfo(source['timezone'])).isoformat(),'slide_count':10,'source_sha256':source_hashes(source),'outputs_sha256':outputs,'quality_checks':checks,'main_results':data['results']['main'],'main_ratio':data['ratio'],'models_are_schematic':True,'external_claude_deck_republished':False,'limitations':['Simulation evidence only, six paired scenes','Successful arm assumes a second client radio','Scaled cellular comparator, not a real 3GPP deployment','Stress condition favors the cellular model','The GLB explorer uses real geometry, not measurements']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    with zipfile.ZipFile(OUT/'wirelessjev_3d_assets.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('*.png'))+sorted(OUT.glob('*.model.json'))+sorted((OUT/'models').glob('*')):
            if not p.name.endswith('.blend1'):z.write(p,str(p.relative_to(OUT)))
        for p in [BASE/'deck.json',BASE/'render_models.py',BASE/'assets/scene_data.json',OUT/'manifest.json']:z.write(p,p.name)
        for p in (BASE/'vendor').iterdir():
            if p.is_file():z.write(p,'vendor/'+p.name)
    print('Built and verified 10 slides: PDF, editable PPTX, HTML, 3D explorer and Blender/GLB assets.',flush=True)


if __name__=='__main__':main()
