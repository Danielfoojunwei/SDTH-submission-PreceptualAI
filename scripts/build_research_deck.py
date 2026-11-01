"""Build the current-system research deck from source and archived evidence.

Outputs: editable PowerPoint text/notes, vector PDF, standalone browser deck,
slide previews, research figures in SVG/PDF/PNG, and a reproducibility manifest.
No campaign is rerun: map/trace pictures are labelled simulator illustrations.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import textwrap
import zipfile
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import pymupdf
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/slides/deck.json'
W,H=16.,9.
BG='#F5F5EF';INK='#152E3D';MUTED='#607581';ORANGE='#D46B3C';BLUE='#3278A3';LINE='#D5DCD9'
POSITIONS={'section':(.65,.25,13,.23,10.5),'title':(.65,.65,14.7,1.07,31.5),
 'subtitle':(.65,1.73,14.7,.5,15.8),'insight':(1.86,7.91,13.35,.5,17.),
 'source':(.65,8.57,14.15,.24,9.),'page':(14.85,8.57,.5,.24,10.)}
VISUAL=(.65,2.25,14.7,5.25)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def color(value):return tuple(int(value[i:i+2],16)/255 for i in [1,3,5])


def register_fonts():
    regular=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
    bold=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
    pdfmetrics.registerFont(TTFont('DeckSans',regular));pdfmetrics.registerFont(TTFont('DeckBold',bold))
    return regular,bold


def lines(text,width_pt,font,size):
    result=[]
    for paragraph in text.split('\n'):
        row=''
        for word in paragraph.split():
            test=(row+' '+word).strip()
            if row and pdfmetrics.stringWidth(test,font,size)>width_pt:
                result.append(row);row=word
            else:row=test
        result.append(row)
    return result


def block_text(slide,key,page,total):
    if key=='section':return f'PRECEPTUAL AI   /   {slide["section"]}'
    if key=='source':return slide['evidence']+'   ·   '+ ' / '.join(Path(s).name for s in slide['sources'][:2])
    if key=='page':return f'{page:02d}/{total}'
    return slide[key]


def typography(slide,page,total):
    blocks=[]
    for key,(x,y,w,h,size) in POSITIONS.items():
        font='DeckBold' if key in ['title','section','insight'] else 'DeckSans'
        value=block_text(slide,key,page,total)
        # References may use a slightly smaller footer, never the core message.
        while key=='source' and pdfmetrics.stringWidth(value,font,size)>w*72 and size>7.5:size-=.25
        wrapped=lines(value,w*72,font,size)
        used=len(wrapped)*size*1.15
        if used>h*72+1:raise ValueError(f'Text overflows: {slide["id"]}/{key}: {used:.1f}>{h*72:.1f}')
        blocks.append(dict(key=key,x=x,y=y,w=w,h=h,size=size,font=font,lines=wrapped,
            color=INK if key in ['title','insight'] else ORANGE if key=='section' else MUTED))
    return blocks


def source_inputs(source):
    paths={SOURCE,Path(__file__),ROOT/'docs/slides/figures.py',ROOT/'pyproject.toml',ROOT/'uv.lock'}
    for slide in source['slides']:
        for relative in slide['sources']:
            p=ROOT/relative
            if not p.exists():raise FileNotFoundError(p)
            if p.is_dir():paths.update(p.rglob('*.py'))
            else:paths.add(p)
    for p in (ROOT/'artifacts/native').glob('*.json'):paths.add(p)
    paths.add(ROOT/'artifacts/warehouse_v2/studies/wifi_link_curves.json')
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def figure_module():
    path=ROOT/'docs/slides/figures.py';spec=importlib.util.spec_from_file_location('sdth_deck_figures',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def add_pptx_text(slide,b):
    shape=slide.shapes.add_textbox(Inches(b['x']),Inches(b['y']),Inches(b['w']),Inches(b['h']))
    frame=shape.text_frame;frame.clear();frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
    frame.word_wrap=False;frame.vertical_anchor=MSO_ANCHOR.TOP
    for i,line in enumerate(b['lines']):
        paragraph=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        paragraph.text=line;paragraph.font.name='DejaVu Sans';paragraph.font.size=Pt(b['size'])
        paragraph.font.bold=b['font']=='DeckBold';paragraph.font.color.rgb=RGBColor.from_string(b['color'][1:])
        paragraph.space_before=Pt(0);paragraph.space_after=Pt(0);paragraph.line_spacing=Pt(b['size']*1.15)
        if b['key']=='page':paragraph.alignment=PP_ALIGN.RIGHT


def notes_text(slide):
    return slide['notes']+'\n\nTAKEAWAY: '+slide['insight']+'\n\nEVIDENCE: '+slide['evidence']+'\nSOURCES:\n'+'\n'.join(slide['sources'])


def build_pptx(source,out):
    prs=Presentation();prs.slide_width=Inches(W);prs.slide_height=Inches(H)
    prs.core_properties.title=source['title']+' — current system architecture'
    prs.core_properties.author='Preceptual AI';prs.core_properties.subject=source['subtitle']
    prs.core_properties.keywords='RadioMap, WirelessJEV, architecture, warehouse connectivity, simulation'
    for i,s in enumerate(source['slides'],1):
        slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string(BG[1:])
        # Native text stays editable; each research figure also ships as SVG.
        for b in typography(s,i,len(source['slides'])):add_pptx_text(slide,b)
        from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
        band=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,Inches(.65),Inches(7.72),Inches(14.7),Inches(.73))
        band.fill.solid();band.fill.fore_color.rgb=RGBColor.from_string('E8EEEA');band.line.fill.background()
        # Send the band behind already placed native text.
        tree=band._element.getparent();tree.remove(band._element);tree.insert(2,band._element)
        label=dict(key='label',x=.85,y=7.94,w=.92,h=.25,size=10.5,font='DeckBold',lines=['INSIGHT'],color=ORANGE)
        add_pptx_text(slide,label)
        x,y,w,h=VISUAL
        slide.shapes.add_picture(str(out/'figures'/f'{s["id"]}.png'),Inches(x),Inches(y),width=Inches(w),height=Inches(h))
        slide.notes_slide.notes_text_frame.text=notes_text(s)
    path=out/'wirelessjev_system_architecture.pptx';prs.save(path);return path


def build_pdf(source,out):
    base=out/'_layout.pdf';c=canvas.Canvas(str(base),pagesize=(W*72,H*72))
    c.setTitle(source['title']+' — current system architecture');c.setAuthor('Preceptual AI')
    for i,s in enumerate(source['slides'],1):
        c.setFillColorRGB(*color(BG));c.rect(0,0,W*72,H*72,fill=1,stroke=0)
        c.setStrokeColorRGB(*color(LINE));c.setLineWidth(.6);c.line(.65*72,8.47*72,15.35*72,8.47*72)
        c.setFillColorRGB(*color('#E8EEEA'));c.roundRect(.65*72,(H-8.45)*72,14.7*72,.73*72,9,fill=1,stroke=0)
        c.setFillColorRGB(*color(ORANGE));c.setFont('DeckBold',10.5);c.drawString(.85*72,(H-7.94)*72-10.5,'INSIGHT')
        for b in typography(s,i,len(source['slides'])):
            c.setFont(b['font'],b['size']);c.setFillColorRGB(*color(b['color']))
            for j,line in enumerate(b['lines']):
                x=b['x']*72;y=(H-b['y'])*72-b['size']-j*b['size']*1.15
                if b['key']=='page':c.drawRightString((b['x']+b['w'])*72,y,line)
                else:c.drawString(x,y,line)
        c.showPage()
    c.save();document=pymupdf.open(base)
    for page,s in zip(document,source['slides']):
        visual=pymupdf.open(out/'figures'/f'{s["id"]}.pdf');x,y,w,h=VISUAL
        page.show_pdf_page(pymupdf.Rect(x*72,y*72,(x+w)*72,(y+h)*72),visual,0);visual.close()
    output=out/'wirelessjev_system_architecture.pdf';document.save(output,garbage=4,deflate=True)
    document.close();base.unlink();return output


def previews(pdf,source,out,font):
    dest=out/'slides';dest.mkdir(exist_ok=True);document=pymupdf.open(pdf);paths=[]
    for i,page in enumerate(document,1):
        path=dest/f'{i:02d}.png';page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(path);paths.append(path)
    document.close()
    thumb_w,thumb_h=480,270;pad=16;label_h=36;cols=4;rows=math_ceil(len(paths)/cols)
    sheet=Image.new('RGB',(cols*(thumb_w+pad)+pad,rows*(thumb_h+label_h+pad)+pad),BG)
    draw=ImageDraw.Draw(sheet);tf=ImageFont.truetype(font,15)
    for i,path in enumerate(paths):
        im=Image.open(path).convert('RGB');im.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
        x=pad+(i%cols)*(thumb_w+pad);y=pad+(i//cols)*(thumb_h+label_h+pad)
        sheet.paste(im,(x,y));draw.text((x,y+thumb_h+4),f'{i+1:02d}  {source["slides"][i]["title"][:48]}',font=tf,fill=INK)
    sheet.save(out/'contact_sheet.jpg',quality=90)
    for group in range(math_ceil(rows/2)):
        y=group*2*(thumb_h+label_h+pad)
        sheet.crop((0,y,sheet.width,min(y+2*(thumb_h+label_h+pad)+pad,sheet.height))).save(out/f'contact_sheet_{group+1:02d}.jpg',quality=93)
    return paths


def math_ceil(value):return int(np.ceil(value))


def build_html(source,paths,out):
    slides=[]
    for s,path in zip(source['slides'],paths):
        slides.append({**s,'image':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()})
    data=json.dumps(slides,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RadioMap + WirelessJEV · System Architecture</title>
<style>*{box-sizing:border-box}body{margin:0;background:#152E3D;color:#f5f5ef;font:16px system-ui}main{height:calc(100vh - 56px);display:grid;place-items:center}#slide{max-width:100%;max-height:100%;object-fit:contain}nav{height:56px;display:flex;align-items:center;justify-content:center;gap:18px}button{background:#294553;color:white;border:1px solid #607581;border-radius:6px;padding:7px 14px;cursor:pointer}#notes{display:none;position:fixed;right:16px;top:16px;max-width:430px;max-height:80vh;overflow:auto;background:#fff;color:#152e3d;padding:22px;border-radius:8px;line-height:1.55;box-shadow:0 6px 36px #0007;white-space:pre-wrap}#overview{display:none;position:fixed;inset:0;background:#152e3df5;overflow:auto;padding:24px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}#overview img{width:100%;cursor:pointer;border:2px solid transparent}#overview img:hover{border-color:#D46B3C}a{color:#3278a3}@media print{nav,#notes,#overview{display:none!important}}</style>
<main><img id="slide" alt=""></main><nav><button id="previous" aria-label="Previous slide">←</button><span id="position"></span><button id="next" aria-label="Next slide">→</button><button id="noteButton">Notes · N</button><button id="gridButton">Overview · O</button><button id="fullButton">Fullscreen · F</button></nav><aside id="notes"></aside><section id="overview" aria-label="Slide overview"></section>
<script>const slides=DECK_DATA;let index=Number(location.hash.slice(1))||1;const image=document.getElementById('slide'),notes=document.getElementById('notes'),overview=document.getElementById('overview');function show(i){index=Math.max(1,Math.min(slides.length,i));const s=slides[index-1];image.src=s.image;image.alt=s.title+' — '+s.insight;document.getElementById('position').textContent=index+' / '+slides.length;notes.textContent=s.title+'\\n\\n'+s.notes+'\\n\\n'+s.evidence+'\\nSources: '+s.sources.join(', ');history.replaceState(null,'','#'+index)}function toggleNotes(){notes.style.display=notes.style.display==='block'?'none':'block'}function toggleGrid(){overview.style.display=overview.style.display==='grid'?'none':'grid'}slides.forEach((s,i)=>{const card=document.createElement('div'),thumb=document.createElement('img'),label=document.createElement('div');thumb.src=s.image;thumb.alt=s.title;label.textContent=(i+1)+'. '+s.title;card.append(thumb,label);thumb.onclick=()=>{toggleGrid();show(i+1)};overview.append(card)});document.getElementById('previous').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);document.getElementById('noteButton').onclick=toggleNotes;document.getElementById('gridButton').onclick=toggleGrid;function full(){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}document.getElementById('fullButton').onclick=full;document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}if(e.key==='Home')show(1);if(e.key==='End')show(slides.length);if(e.key.toLowerCase()==='n')toggleNotes();if(e.key.toLowerCase()==='o')toggleGrid();if(e.key.toLowerCase()==='f')full();if(e.key==='Escape'){overview.style.display='none';notes.style.display='none'}});show(index);</script></html>'''.replace('DECK_DATA',data)
    path=out/'wirelessjev_system_architecture.html';path.write_text(html);return path


def presenter_script(source,out):
    result=['# Presenter script: current RadioMap + WirelessJEV system','',
        'Canonical evidence base: PR #7, commit `'+source['base_commit']+'`. This deck explains the current architecture and design rationale; it does not present a development timeline or version comparisons.','',
        'Files: [PowerPoint](slides/generated/wirelessjev_system_architecture.pptx), [PDF](slides/generated/wirelessjev_system_architecture.pdf), [browser deck](slides/generated/wirelessjev_system_architecture.html). All headline measurements are simulation evidence, with explicitly separate software-tool and field-validation boundaries.','']
    for i,s in enumerate(source['slides'],1):
        result += [f'## {i}. {s["title"]}','',f'**Core insight:** {s["insight"]}','',s['notes'],'',f'**Evidence:** {s["evidence"]}',f'**Sources:** '+', '.join('`'+x+'`' for x in s['sources']),'']
    path=ROOT/'docs/PRESENTATION_SCRIPT.md';path.write_text('\n'.join(result));return path


def validate(source,out):
    ids=[s['id'] for s in source['slides']]
    if len(set(ids))!=len(ids):raise ValueError('Repeated visual IDs')
    prs=Presentation(out/'wirelessjev_system_architecture.pptx');pdf=pymupdf.open(out/'wirelessjev_system_architecture.pdf')
    if len(prs.slides)!=len(ids) or pdf.page_count!=len(ids):raise ValueError('Slide count mismatch')
    images=set();checks=[]
    for i,(s,slide,page) in enumerate(zip(source['slides'],prs.slides,pdf),1):
        pictures=[p for p in slide.shapes if p.shape_type==MSO_SHAPE_TYPE.PICTURE]
        if len(pictures)!=1:raise ValueError('Missing substantive visual on '+s['id'])
        images.add(hashlib.sha256(pictures[0].image.blob).hexdigest())
        if s['notes'] not in slide.notes_slide.notes_text_frame.text:raise ValueError('Missing speaker notes')
        extracted=' '.join(page.get_text().split())
        if ' '.join(s['title'].split()) not in extracted:raise ValueError('Missing PDF headline: '+s['id'])
        for ext in ['svg','pdf','png']:
            p=out/'figures'/f'{s["id"]}.{ext}'
            if not p.exists() or p.stat().st_size<1000:raise ValueError('Missing research figure: '+str(p))
        checks.append({'slide':i,'id':s['id'],'substantive_visual':True,'editable_title_and_notes':True,'pdf_title_verified':True})
    pdf.close()
    if len(images)!=len(ids):raise ValueError('Every slide needs its own substantive visual')
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'docs/slides/generated');p.add_argument('--reuse-figures',action='store_true');args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);source=json.loads(SOURCE.read_text());font,_=register_fonts()
    inputs=source_inputs(source);figdir=out/'figures';figdir.mkdir(exist_ok=True);cache=out/'figure_build.json'
    ids=[s['id'] for s in source['slides']]
    for path in figdir.iterdir():
        if path.suffix in ['.svg','.pdf','.png'] and path.stem not in ids:path.unlink()
    # Slide copy and page layout do not affect the independently drawn figure.
    figure_inputs={k:v for k,v in inputs.items() if k not in ['docs/slides/deck.json','scripts/build_research_deck.py']}
    if args.reuse_figures:
        if not cache.exists():raise RuntimeError('No verified figure cache; regenerate figures')
        previous=json.loads(cache.read_text())
        previous_inputs=previous.get('figure_source_sha256', {k:v for k,v in previous.get('source_sha256',{}).items() if k not in ['docs/slides/deck.json','scripts/build_research_deck.py']})
        if previous_inputs!=figure_inputs or previous['figures']!=ids:raise RuntimeError('Figure cache source mismatch; regenerate figures')
        facts=json.loads((out/'figure_facts.json').read_text())
    else:
        facts=figure_module().Figures(figdir).build(ids)
        (out/'figure_facts.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
        cache.write_text(json.dumps({'figure_source_sha256':figure_inputs,'figures':ids},indent=2)+'\n')
    build_pptx(source,out);pdf=build_pdf(source,out);paths=previews(pdf,source,out,font);build_html(source,paths,out);presenter_script(source,out)
    checks=validate(source,out)
    files={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ['manifest.json','wirelessjev_figure_assets.zip','browser_check.json','browser_preview.png']}
    manifest={'title':source['title'],'base_pr':source['base_pr'],'base_commit':source['base_commit'],
        'built_at':datetime.now(ZoneInfo(source['timezone'])).isoformat(),'slide_count':len(ids),
        'every_slide_has_unique_visual':True,'source_sha256':inputs,'outputs_sha256':files,
        'renderer_versions':{name:importlib.metadata.version(name) for name in ['matplotlib','python-pptx','reportlab','pymupdf']},
        'quality_checks':checks,'external_claude_deck_republished':False,'limitations':['All performance comparisons are simulations','PPTX text and notes are editable; diagrams also ship as SVG','Physical car and actual private-5G qualification remain pending']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    with zipfile.ZipFile(out/'wirelessjev_figure_assets.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ids:
            for ext in ['svg','pdf','png']:
                path=figdir/f'{name}.{ext}';archive.write(path,str(path.relative_to(out)))
        for relative in ['figure_facts.json','manifest.json']:archive.write(out/relative,relative)
        archive.write(SOURCE,'deck.json');archive.write(ROOT/'docs/slides/figures.py','figures.py')
    print('Built and verified',len(ids),'slides; PPTX, vector PDF, browser deck, previews and figure assets.',flush=True)


if __name__=='__main__':main()
