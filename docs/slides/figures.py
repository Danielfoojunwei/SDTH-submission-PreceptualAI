"""Research-deck figures: traceable charts and implementation diagrams.

Numbers come from immutable campaign reports unless explicitly labelled as a
published development diagnostic. Drawings describe code, never field hardware.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
C={'ink':'#152E3D','blue':'#3278A3','orange':'#D46B3C','teal':'#278D80',
   'muted':'#607581','grey':'#9BA8AD','bg':'#F5F5EF','white':'#FFFFFF',
   'pale_blue':'#E7EFF3','pale_orange':'#F5E9DE','pale_teal':'#E2EFEB','red':'#B94B48'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,'axes.labelsize':18,
 'axes.titlesize':22,'axes.titleweight':'bold','axes.labelcolor':C['ink'],
 'text.color':C['ink'],'xtick.color':C['muted'],'ytick.color':C['muted'],
 'axes.edgecolor':C['grey'],'axes.spines.top':False,'axes.spines.right':False,
 'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':C['bg']})


def load(path):return json.loads((ROOT/path).read_text())


def records(name,condition=None,arm=None):
    return [r for r in load(f'artifacts/native/{name}.json')['rows']
            if (condition is None or r.get('condition')==condition) and (arm is None or r.get('arm')==arm)]


def avg(name,condition,arm,key='control_miss'):
    rows=records(name,condition,arm)
    if not rows:raise ValueError((name,condition,arm))
    return float(np.mean([r[key] for r in rows]))


def canvas():
    fig,ax=plt.subplots(figsize=(17.5,6.25),facecolor=C['bg'])
    fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.97)
    ax.set(xlim=(0,16),ylim=(0,6));ax.axis('off')
    return fig,ax


def chart(ncols=1):
    fig,axes=plt.subplots(1,ncols,figsize=(17.5,6.25),facecolor=C['bg'],squeeze=False)
    fig.subplots_adjust(left=.095,right=.975,bottom=.19,top=.86,wspace=.4)
    for ax in axes.ravel():ax.set_facecolor(C['bg']);ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
    return fig,axes.ravel()


def wrap(value,width):return '\n'.join(textwrap.wrap(value,width=width,break_long_words=False,break_on_hyphens=False))


def text(ax,x,y,value,size=19,color=None,ha='left',va='center',weight='normal',width=None):
    if width and '\n' not in value:value=wrap(value,width)
    return ax.text(x,y,value,fontsize=size,color=color or C['ink'],ha=ha,va=va,
                   weight=weight,linespacing=1.3)


def box(ax,x,y,w,h,title,body='',color='blue',fill=None,dashed=False,fs=20):
    from matplotlib.font_manager import FontProperties
    col=C.get(color,color)
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035,rounding_size=.12',
        linewidth=1.8,edgecolor=col,facecolor=fill or C['white'],linestyle='--' if dashed else '-')
    ax.add_patch(patch)
    renderer=ax.figure.canvas.get_renderer()
    low=ax.transData.transform((x,y));high=ax.transData.transform((x+w,y+h))
    width=high[0]-low[0]-18;height=high[1]-low[1]-16;scale=ax.figure.dpi/72.
    def measured(value,size,bold=False):
        fp=FontProperties(family='DejaVu Sans',size=size,weight='bold' if bold else 'normal')
        rows=[]
        for para in value.split('\n'):
            line=''
            for word in para.split():
                trial=(line+' '+word).strip()
                if line and renderer.get_text_width_height_descent(trial,fp,False)[0]>width:
                    rows.append(line);line=word
                else:line=trial
            rows.append(line)
        return rows
    title_size=float(fs);body_size=16.
    for _ in range(20):
        head=measured(title,title_size,True);tail=measured(body,body_size) if body else []
        head_h=len(head)*title_size*1.2*scale;tail_h=len(tail)*body_size*1.2*scale
        total=head_h+tail_h+(8 if body else 0)
        widths=[renderer.get_text_width_height_descent(t,FontProperties(family='DejaVu Sans',size=title_size,weight='bold'),False)[0] for t in head]
        if total<=height and max(widths,default=0)<=width:break
        title_size=max(15.,title_size-.5);body_size=max(12.5,body_size-.25)
    top=(low[1]+high[1]+total)/2
    inv=ax.transData.inverted()
    tx,ty=inv.transform(((low[0]+high[0])/2,top))
    artist=text(ax,tx,ty,'\n'.join(head),title_size,col,'center','top',weight='bold')
    artist._deck_box=patch
    if tail:
        tx,ty=inv.transform(((low[0]+high[0])/2,top-head_h-8))
        artist=text(ax,tx,ty,'\n'.join(tail),body_size,C['muted'],'center','top')
        artist._deck_box=patch


def arrow(ax,start,end,label='',color='muted',rad=0.,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=19,
        lw=1.8,color=C.get(color,color),connectionstyle=f'arc3,rad={rad}',
        linestyle='--' if dashed else '-'))
    if label:
        x=(start[0]+end[0])/2;y=(start[1]+end[1])/2+.2
        text(ax,x,y,label,14,C['muted'],'center')


def callout(ax,x,y,value,label,color='orange',w=3.6):
    text(ax,x,y,value,42,C[color],weight='bold')
    text(ax,x,y-.65,label,17,C['muted'],width=int(w*10))


def save(fig,out,name):
    for extension in ['svg','pdf','png']:
        path=out/f'{name}.{extension}'
        fig.savefig(path,dpi=180,facecolor=C['bg'])
        if extension=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


class Figures:
    def __init__(self,out):
        self.out=out;out.mkdir(parents=True,exist_ok=True);self.facts={};self._world=None

    def world(self):
        if self._world is None:
            from wirelessjev.native.world import build_world
            self._world=build_world(530,duration_s=300.,n_robots=100,survey_s=60)
        return self._world

    def warehouse_background(self,ax,*,field=True):
        from wirelessjev.twin.warehouse import make_warehouse,AP_SITES,RACK_ROWS,RACK_SEGMENTS
        from wirelessjev.native.world import _path_losses,NOISE_DBM
        scene=make_warehouse(530,n_robots=100)
        a5=[a for a in scene.aps if a.band=='5g'];a2=[a for a in scene.aps if a.band=='2g']
        if field:
            x,y=np.meshgrid(np.linspace(.5,119.5,240),np.linspace(.5,79.5,160))
            snr=(15-_path_losses(scene,a5,a2,x,y)[...,1]-NOISE_DBM['5g']).max(2)
            ax.imshow(snr,origin='lower',extent=(0,120,0,80),cmap='Blues',vmin=-5,vmax=40,alpha=.88)
        for y in RACK_ROWS:
            for x0,x1 in RACK_SEGMENTS:ax.add_patch(Rectangle((x0,y-.65),x1-x0,1.3,color=C['ink'],alpha=.4,lw=0))
        site=np.asarray(AP_SITES);ax.scatter(site[:,0],site[:,1],s=65,c=C['orange'],edgecolors='white',linewidth=.8,zorder=4)
        ax.scatter([c.x for c in scene.cars],[c.y for c in scene.cars],s=12,c='white',edgecolors=C['ink'],linewidth=.25,zorder=5)
        ax.set(xlim=(0,120),ylim=(0,80),aspect='equal');ax.axis('off')

    def cover(self):
        fig=plt.figure(figsize=(17.5,6.25),facecolor=C['bg']);gs=fig.add_gridspec(1,2,width_ratios=[1.65,1],wspace=.15)
        ax=fig.add_subplot(gs[0]);self.warehouse_background(ax)
        ax.set_title('The test warehouse · 100 robots / 16 AP sites',loc='left',pad=18)
        a=fig.add_subplot(gs[1]);a.set(xlim=(0,6),ylim=(0,6));a.axis('off')
        for y,label,color in [(4.7,'LEARN THE SITE','blue'),(2.9,'COMPILE THE MISSION','orange'),(1.1,'RECOVER THROUGH ANOTHER AP','teal')]:
            box(a,.1,y-.5,5.7,1.05,label,color=color,fs=17)
        arrow(a,(2.95,4.18),(2.95,3.45));arrow(a,(2.95,2.38),(2.95,1.65))
        text(a,.1,.1,'Simulation evidence · hardware proof pending',15,C['muted'])
        return fig

    def problem(self):
        fig,ax=canvas();times=[.8,4.3,7.8,11.3]
        for x,t,b,c in zip(times,['CONTROL','VIDEO','BULK','THE REAL COST'],
                            ['600 B · 20 Hz\n50 ms deadline','150 ms modelled deadline\nOn-time delivery matters','Buffered application data\nRate is not continuity','Command silence\nA robot may stop'],
                            ['orange','blue','teal','ink']):
            box(ax,x,2.7,3.0,1.6,t,b,color=c,fs=19)
        # Traffic classes are parallel requirements, not a processing chain.
        ax.plot([.8,15],[1.6,1.6],color=C['grey'],lw=2)
        for x,label,c in [(1.2,'command',C['blue']),(3.2,'retry',C['orange']),(6.7,'deadline',C['red'])]:
            ax.vlines(x,1.4,1.85,color=c,lw=3);text(ax,x,1.05,label,18,c,'center')
        ax.add_patch(Rectangle((1.2,1.4),5.5,.42,color=C['pale_orange'],zorder=-1))
        text(ax,11.2,1.25,'Average Mbps can look healthy\nwhile short commands expire.',22,C['ink'],'center',weight='bold')
        return fig

    def design_rationale(self):
        fig,ax=canvas()
        pairs=[('KNOW COVERAGE','RadioMap','Spatial mean + uncertainty','blue'),
               ('EXPRESS INTENT','Mission compiler','Text → explicit policy knobs','orange'),
               ('BOUND AUTONOMY','Confidence gate','Plan or reactive fallback','orange'),
               ('SURVIVE PATH LOSS','Cross-AP executor','Independent retry opportunity','teal'),
               ('MEASURE OUTCOMES','Supervised adapter','Probe / request / reconcile','ink')]
        for i,(requirement,module,body,col) in enumerate(pairs):
            x=.2+i*3.15
            box(ax,x,4.25,2.8,1.0,requirement,color=col,fs=16)
            arrow(ax,(x+1.4,4.2),(x+1.4,3.1),color=col)
            box(ax,x,1.45,2.8,1.6,module,body,color=col,fs=21)
        text(ax,8,.5,'Five complementary responsibilities in one observable control system.',23,C['ink'],'center',weight='bold')
        return fig

    def responsibilities(self):
        fig,ax=canvas()
        items=[('RadioMap','INPUT: capped scans + geometry','OUTPUT: mean_dBm + std_dB','blue'),
               ('WirelessJEV','INPUT: mission + map + state','OUTPUT: explicit policy actions','orange'),
               ('Executor + adapters','INPUT: actions + queued traffic','OUTPUT: delivery + outcome state','teal')]
        for i,(title,inp,out,col) in enumerate(items):
            x=.25+i*5.25;box(ax,x,2.4,4.65,2.2,title,color=col,fs=27)
            text(ax,x+2.325,3.02,inp,15,C['muted'],'center');text(ax,x+2.325,1.2,out,16,C[col],'center',weight='bold')
            if i<2:arrow(ax,(x+4.7,3.5),(x+5.2,3.5),color=col)
        return fig

    def architecture(self):
        fig,ax=canvas()
        box(ax,.1,4.35,2.15,1.05,'CAD + APs',color='ink',fs=18)
        box(ax,.1,2.6,2.15,1.05,'Capped scans',color='blue',fs=18)
        box(ax,.1,.5,2.15,1.05,'Mission text',color='orange',fs=18)
        box(ax,2.85,2.55,2.7,1.25,'RadioMap','Mean / uncertainty',color='blue')
        box(ax,2.85,.4,2.7,1.25,'d3-lite reader','Weights → policy knobs',color='orange')
        box(ax,6.2,2.55,2.8,1.25,'Numerical policy','AP / links / timing',color='orange')
        box(ax,9.65,2.55,2.5,1.25,'Trust gate','Plan or react',color='orange')
        box(ax,12.8,2.55,2.95,1.25,'Slot executor','Retries / queues / delivery',color='teal')
        arrow(ax,(2.3,3.15),(2.8,3.15));arrow(ax,(5.6,3.15),(6.15,3.15));arrow(ax,(9.05,3.15),(9.6,3.15));arrow(ax,(12.2,3.15),(12.75,3.15))
        arrow(ax,(2.3,1.02),(2.8,1.02));arrow(ax,(5.6,1.02),(6.9,2.5),color='orange')
        arrow(ax,(2.3,4.75),(3.7,3.85),color='ink')
        box(ax,6.2,4.35,2.8,1.05,'Routes + demand',color='orange',fs=18)
        arrow(ax,(7.6,4.3),(7.6,3.85),color='orange')
        box(ax,6.2,.35,2.8,1.15,'Contract verifier','Separate proposal interface',color='grey',dashed=True,fs=18)
        box(ax,9.65,.35,2.5,1.15,'Reactive rule','Fallback when uncertain',color='teal',fs=18)
        box(ax,12.8,.35,2.95,1.15,'Supervised car tool','Fake-car tested; field pending',color='grey',dashed=True,fs=18)
        arrow(ax,(10.9,2.5),(10.9,1.55));arrow(ax,(12.2,1),(14.3,2.5),color='teal')
        arrow(ax,(14.25,2.5),(14.25,1.55),dashed=True)
        ax.plot([14.25,14.25,4.2,4.2],[3.85,5.65,5.65,3.85],c=C['blue'],lw=1.8)
        arrow(ax,(4.2,4.12),(4.2,3.85),color='blue')
        text(ax,9.4,5.9,'observed link measurements · periodic radio-map updates',17,C['blue'],'center')
        return fig

    def timescales(self):
        fig,ax=canvas();items=[('10 ms','Retry + deadline accounting','teal'),('100 ms','Measurement + link adaptation','blue'),('1 s','Native numerical planning','orange'),('10 s','Online RadioMap update','blue'),('Mission change','d3-lite interpretation','orange')]
        for i,(time,title,col) in enumerate(items):
            x=.25+i*3.15;box(ax,x,2.5,2.85,1.8,time,title,color=col,fs=24)
        ax.plot([.45,15.7],[1.6,1.6],c=C['ink'],lw=2)
        arrow(ax,(.45,1.6),(15.7,1.6),color='ink');text(ax,8,.8,'faster packet loop ←                       → slower decisions and learning',23,C['muted'],'center')
        return fig

    def warehouse(self):
        fig=plt.figure(figsize=(17.5,6.25),facecolor=C['bg']);gs=fig.add_gridspec(1,2,width_ratios=[1.35,1],wspace=.2)
        ax=fig.add_subplot(gs[0]);self.warehouse_background(ax)
        a=fig.add_subplot(gs[1]);a.set(xlim=(0,7),ylim=(0,6));a.axis('off')
        for i,(title,body,col) in enumerate([('SHARED','Routes · scene seed · fading · blockage','ink'),('OBSERVED','Capped scans · lagged link measurements','blue'),('HIDDEN','Instantaneous fades and blockage','orange')]):
            box(a,.1,4.5-i*1.65,6.7,1.25,title,body,color=col,fs=20)
        return fig

    def phy(self):
        from wirelessjev.native.phy import LinkCurves
        fig,axes=chart(2);curves=LinkCurves();ax=axes[0]
        snr,succ,rate=curves.tables['5g']
        for i in [0,3,5,8]:ax.plot(snr,succ[i],lw=2.5,label=f'MCS {i}')
        ax.set(xlim=(-5,35),ylim=(-.02,1.03),xlabel='SNR (dB)',ylabel='Packet success / attempt',title='Archived NIST VHT80 curves')
        ax.legend(frameon=False,fontsize=15,loc='lower right')
        a=axes[1];a.axis('off');a.set(xlim=(0,7),ylim=(0,6))
        box(a,.1,4.35,6.7,1.15,'SOURCE','VHT80 / HT20 · 2 streams · 1,200 B',color='blue')
        box(a,.1,2.35,6.7,1.15,'EXTRAPOLATION','6 GHz: double 5 GHz rate at 160 MHz',color='orange')
        box(a,.1,.35,6.7,1.15,'SYSTEM MODEL','10 ms slots · retry chains · simplified MAC',color='ink')
        arrow(a,(3.45,4.3),(3.45,3.55));arrow(a,(3.45,2.3),(3.45,1.55))
        return fig

    def scan_learning(self):
        fig=plt.figure(figsize=(17.5,6.25),facecolor=C['bg']);gs=fig.add_gridspec(1,2,width_ratios=[1.2,1],wspace=.25)
        from wirelessjev.native.radiomap import survey
        w=self.world();observed,censored=survey(w.rssi_dl[:1],w.pos[:1],np.random.default_rng(40530))
        values=sorted([float(v[2][np.flatnonzero(v[3]==0)[0]]) for v in observed.values() if (v[3]==0).any()],reverse=True)
        bounds=[float(v[2][np.flatnonzero(v[3]==0)[0]]) for v in censored.values() if (v[3]==0).any()]
        bound=bounds[0];values=np.array(values+[bound]*max(0,12-len(values)))[:12];heard=np.arange(12)<8
        ax=fig.add_subplot(gs[0])
        ax.bar(np.arange(12)[heard],values[heard]+100,bottom=-100,color=C['blue'],width=.7)
        ax.bar(np.arange(12)[~heard],np.full(4,bound+100),bottom=-100,color=C['grey'],alpha=.22,hatch='//',width=.7)
        ax.axhline(bound,c=C['orange'],ls='--',lw=2);ax.text(8,bound+2,'cap bound',fontsize=17,c=C['orange'])
        ax.set(ylim=(-100,-25),xlabel='BSS rank · simulated scan for robot 0',ylabel='RSSI (dBm)',title='Eight heard APs; others are censored')
        ax.set_xticks(np.arange(12));ax.set_xticklabels(np.arange(1,13));ax.grid(axis='y',alpha=.15)
        a=fig.add_subplot(gs[1]);a.set(xlim=(0,7),ylim=(0,6));a.axis('off')
        box(a,.1,4.4,6.7,1.1,'HEARD','Fit the noisy RSSI reading',color='blue')
        box(a,.1,2.4,6.7,1.1,'OMITTED','Fit P(signal < scan bound)',color='orange')
        box(a,.1,.4,6.7,1.1,'POSTERIOR MAP','Geometry + local structure + discrepancy',color='teal')
        arrow(a,(3.45,4.35),(3.45,3.55));arrow(a,(3.45,2.35),(3.45,1.55))
        fig.subplots_adjust(left=.08,right=.97,bottom=.16,top=.88)
        self.facts['scan_learning']={'scene':530,'robot':0,'source':'actual simulator survey() output','heard_rssi_dbm':values[heard].tolist(),'censor_bound_dbm':bound,'field_measurement':False}
        return fig

    def map_accuracy(self):
        from wirelessjev.native.world import _path_losses
        from wirelessjev.twin.warehouse import make_warehouse
        w=self.world();scene=make_warehouse(530,n_robots=100)
        a5=[a for a in scene.aps if a.band=='5g'];a2=[a for a in scene.aps if a.band=='2g']
        gx,gy=np.meshgrid(np.linspace(1,119,110),np.linspace(1,79,75))
        truth=(a5[0].tx_power_dbm-_path_losses(scene,a5,a2,gx,gy))[...,5,1]
        pred,std=w.rmap.predict(gx,gy)
        fig=plt.figure(figsize=(17.5,6.25),facecolor=C['bg']);gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.05],wspace=.32)
        for j,(z,title) in enumerate([(truth,'Hidden signal map'),(pred[...,5,1],'RadioMap after 60 s')]):
            ax=fig.add_subplot(gs[j]);im=ax.imshow(z,origin='lower',extent=(0,120,0,80),cmap='viridis',vmin=-95,vmax=-35)
            ax.set(title=title,xlabel='metres');ax.set_yticks([0,40,80]);ax.set_aspect('equal')
            if j==0:ax.set_ylabel('metres')
        ax=fig.add_subplot(gs[2]);cal=load('artifacts/native/learned_campaign.json')['calibration']
        cover=[np.mean([r['coverage_90'] for r in cal if r['condition']==c])*100 for c in ['survey60','survey300']]
        ax.bar([0,1],cover,color=[C['blue'],C['orange']],width=.55)
        ax.axhline(90,c=C['ink'],ls='--',lw=2);ax.set(xticks=[0,1],xticklabels=['60 s','300 s'],ylim=(0,105),ylabel='90% interval coverage (%)',title='Uncertainty undercovers')
        for i,v in enumerate(cover):ax.text(i,v+3,f'{v:.0f}%',ha='center',fontsize=21,weight='bold')
        fig.subplots_adjust(left=.055,right=.97,bottom=.18,top=.88)
        self.facts['map_accuracy']={'scene':530,'ap_site':5,'illustrative_model_render':True,'coverage_90_percent':cover}
        return fig

    def intent(self):
        d=load('artifacts/native/intent_flip.json');names=list(d['readings']);keys=['ho','fair','focus','viol','prot','W','V']
        values=np.array([[d['readings'][n]['weights'][k] for k in keys] for n in names])
        fig,axes=chart(2);ax=axes[0];im=ax.imshow(values,cmap='YlOrBr',aspect='auto',vmin=0,vmax=5)
        ax.set_yticks(range(4),['Protect teleop','Bulk upload','Protect amr-067','Quiet scanners'])
        ax.set_xticks(range(7),['roam','fair','focus','floor','protect','interf.','rate'],rotation=35,ha='right')
        for i in range(4):
            for j in range(7):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=16,c='white' if values[i,j]>3 else C['ink'])
        ax.set_title('Actual d3-lite mission readings');ax.grid(False)
        a=axes[1];a.axis('off');a.set(xlim=(0,7),ylim=(0,6))
        for i,(t,b) in enumerate([('TEXT → WEIGHTS','Inference on mission change'),('WEIGHTS → KNOBS','Thresholds / video share / focus'),('SEPARATE CONTRACT TOOL','Explicit groups / floors / proposal checks')]):
            box(a,.1,4.35-i*1.9,6.7,1.2,t,b,color='orange' if i<2 else 'grey',dashed=i==2,fs=18)
        return fig

    def allocation(self):
        fig,ax=canvas()
        box(ax,.2,2.15,3.15,2.2,'Decision inputs','Route forecast / AP load\nMission knobs / map posterior',color='blue',fs=22)
        box(ax,4.35,3.85,5.1,1.45,'AP choice','Near-term signal − load penalty\nGain threshold + confidence gate',color='orange',fs=21)
        box(ax,4.35,1.25,5.1,1.45,'Traffic-to-link mapping','Control: conservative near-term link\nBulk: predicted goodput',color='teal',fs=21)
        arrow(ax,(3.4,3.6),(4.3,4.4));arrow(ax,(3.4,2.8),(4.3,2.0))
        box(ax,10.55,2.15,5.15,2.2,'One policy output','AP / control band / data band\nRetry path / video share / data timing',color='ink',fs=22)
        arrow(ax,(9.5,4.45),(10.5,3.65));arrow(ax,(9.5,2.0),(10.5,2.75))
        text(ax,8,.45,'Explicit numerical heuristics, evaluated once per second—not a global optimum claim.',20,C['muted'],'center')
        return fig

    def trust_gate(self):
        fig=plt.figure(figsize=(17.5,6.25),facecolor=C['bg']);gs=fig.add_gridspec(1,2,width_ratios=[1,1.3],wspace=.25)
        a=fig.add_subplot(gs[0]);a.set(xlim=(0,7),ylim=(0,6));a.axis('off')
        box(a,.3,4.3,6.2,1.1,'Predicted AP gain / uncertainty',color='blue',fs=20)
        box(a,.3,2.25,2.65,1.1,'Confident','Planned action',color='orange',fs=19)
        box(a,3.85,2.25,2.65,1.1,'Uncertain','Reactive fallback',color='teal',fs=19)
        arrow(a,(2.1,4.25),(1.6,3.4));arrow(a,(4.7,4.25),(5.2,3.4))
        text(a,3.4,.9,r'$z=\Delta / \sqrt{\sigma_{old}^{2}+\sigma_{new}^{2}}$',24,C['ink'],'center')
        ax=fig.add_subplot(gs[1]);conds=['survey60','survey300','survey300_2g_noisy'];lab=['60 s survey','300 s survey','Noisy 2.4 GHz'];y=np.arange(3)
        for offset,arm,col in [(-.25,'mlo_split','grey'),(0,'native_learned','blue'),(.25,'gated_learned','orange')]:
            vals=[avg('learned_campaign',c,arm)*100 for c in conds];ax.barh(y+offset,vals,height=.23,color=C[col],label=arm.replace('_',' '))
        ax.set(yticks=y,yticklabels=lab,xlabel='Control misses (%)',title='Useful decisions despite imperfect intervals');ax.invert_yaxis();ax.legend(frameon=False,fontsize=14,loc='lower right');ax.grid(axis='x',alpha=.15)
        fig.subplots_adjust(left=.03,right=.98,bottom=.16,top=.88)
        return fig

    def handover(self):
        fig,ax=canvas()
        for y,title,parts,col in [(4.15,'NetworkManager profile activation',['Disconnect','Associate','Authenticate','DHCP','First packet'],'grey'),
                                  (1.8,'Within-connection roam target',['Request roam','Reassociate / FT','Keep IP','First packet'],'teal')]:
            text(ax,.3,y+1.05,title,21,C[col],weight='bold')
            widths=14.9/len(parts)
            for i,t in enumerate(parts):
                x=.35+i*widths;box(ax,x,y-.25,widths-.3,.95,t,color=col,fs=18)
                if i<len(parts)-1:arrow(ax,(x+widths-.25,y+.22),(x+widths-.04,y+.22),color=col)
        text(ax,8,.5,'Built: read-only probes + stage clock checks + supervised request.   Unmeasured: real interruption.',19,C['ink'],'center')
        return fig

    def path_diversity(self):
        fig,ax=canvas()
        box(ax,.4,2.45,2.1,1.1,'Robot',color='ink');box(ax,4.05,2.25,2.7,1.5,'One AP site','2.4 / 5 / 6 GHz',color='blue')
        for off in [-.25,0,.25]:arrow(ax,(2.55,3+off),(4,3+off),color='blue')
        ax.plot([6.1,6.9],[4.7,3.7],c=C['red'],lw=5);ax.plot([6.1,6.9],[3.7,4.7],c=C['red'],lw=5)
        text(ax,3.5,1.1,'Bands can share one blocked path.',20,C['muted'],'center')
        box(ax,8.25,2.45,2.1,1.1,'Robot',color='ink');box(ax,12,4.0,2.7,1.1,'AP A',color='blue');box(ax,12,1.65,2.7,1.1,'AP B',color='teal')
        arrow(ax,(10.4,3.3),(11.95,4.5),color='blue');arrow(ax,(10.4,2.8),(11.95,2.2),color='teal')
        text(ax,12.8,.6,'Different AP = another geometric path',20,C['teal'],'center')
        text(ax,8,5.55,'Frequency diversity                              Attachment-path diversity',24,C['ink'],'center',weight='bold')
        return fig

    def retry_trace(self):
        from wirelessjev.native.sim import hidden_draws,BLOCK_SLOTS,BLOCKAGE_DB
        from wirelessjev.native.phy import LinkCurves
        w=self.world();blocks=3000;fade,blocked,_=hidden_draws(w,blocks*BLOCK_SLOTS)
        kp=np.minimum(np.arange(blocks)*BLOCK_SLOTS//20,len(w.snr)-1);mean=w.snr[kp][...,1]
        inst=mean+fade[:blocks][...,1]-BLOCKAGE_DB*blocked[:blocks]
        order=np.argsort(-mean,axis=2);t=np.arange(blocks)[:,None];r=np.arange(w.n)[None,:]
        a=inst[t,r,order[...,0]];b=inst[t,r,order[...,1]]
        threshold=LinkCurves().threshold_db('5g',.99)[0];rescue=(a<threshold)&(b>=threshold);both=(a<threshold)&(b<threshold)
        best=(-1,0,0)
        for robot in range(w.n):
            rolling=np.convolve(rescue[:,robot],np.ones(200),'valid');start=int(rolling.argmax())
            if rolling[start]>best[0]:best=(float(rolling[start]),robot,start)
        _,robot,start=best;sl=slice(start,start+200);x=np.arange(200)*.1
        fig,axes=chart();ax=axes[0]
        ax.plot(x,a[sl,robot],c=C['blue'],lw=2.2,label='Primary AP path');ax.plot(x,b[sl,robot],c=C['orange'],lw=2.2,label='Alternative AP path')
        ax.axhline(threshold,c=C['ink'],ls='--',lw=1.4,label='99% single-attempt reference')
        yes=rescue[sl,robot];no=both[sl,robot]
        ax.scatter(x[yes],a[sl,robot][yes],s=75,facecolors='none',edgecolors=C['orange'],lw=2,label='Alternative above reference')
        ax.scatter(x[no],a[sl,robot][no],marker='x',s=65,c=C['red'],label='Both below reference')
        ax.set(xlabel='Seconds within selected illustrative window',ylabel='Uplink SNR (dB)',title=f'Scene 530 · robot {robot} · post-hoc selected window')
        ax.legend(ncols=3,frameon=False,fontsize=13,loc='upper center',bbox_to_anchor=(.5,-.19))
        fig.subplots_adjust(bottom=.28,top=.89)
        share=float(rescue.sum()/max(rescue.sum()+both.sum(),1));self.facts['retry_trace']={'scene':530,'robot':robot,'start_s':start*.1,'selected_for_many_opportunities':True,'fleet_alternative_available_fraction_when_primary_below_reference':share,'threshold_db':float(threshold),'not_packet_miss_counts':True}
        return fig

    def forensics(self):
        lines=(ROOT/'docs/CROSS_AP.md').read_text().split('## New arms')[0].splitlines();data=[]
        for line in lines:
            cells=[s.strip() for s in line.strip().strip('|').split('|')]
            if len(cells)!=6:continue
            try:rate=float(cells[1]);counts=np.array(list(map(float,cells[2:])))
            except ValueError:continue
            if 'onboard' not in cells[0]:data.append((cells[0],rate,counts))
        fig,axes=chart();ax=axes[0];labels=[];colors=[C['grey'],C['ink'],C['blue'],C['orange']]
        for i,(name,rate,counts) in enumerate(data):
            left=0
            for j,(v,col) in enumerate(zip(counts,colors)):
                width=rate*v/counts.sum();ax.barh(i,width,left=left,color=col,label=['No radio up','Blocked path','Weak link','Contention'][j] if i==0 else None,height=.65);left+=width
            ax.text(rate+8,i,f'{rate:g} / M',va='center',weight='bold',fontsize=18)
            labels.append(name.replace('Vertical,','Controller,'))
        ax.set(yticks=range(len(data)),yticklabels=labels,xlim=(0,570),xlabel='Misses per million · documented diagnostic scene 400',title='The dominant miss cause changes only with another AP path')
        ax.invert_yaxis();ax.legend(ncols=4,frameon=False,fontsize=15,loc='upper center',bbox_to_anchor=(.52,-.2));fig.subplots_adjust(left=.25,bottom=.26)
        self.facts['forensics']={'source':'docs/CROSS_AP.md development scene 400 table','not_a_fresh_test_result':True,'arms':len(data)}
        return fig

    def ablations(self):
        fig,axes=chart(2);ax=axes[0]
        arms=['vertical_single','vertical_client'];values=[avg('wifi_vs_5g','main',a)*1e6 for a in arms]
        ax.barh([1,0],values,color=[C['grey'],C['orange']],height=.6)
        ax.set(yticks=[1,0],yticklabels=['One client radio','Two AP paths'],xscale='log',xlim=(1,700),xlabel='Main-condition misses / M',title='The decisive architectural change')
        for y,v in zip([1,0],values):ax.text(v*1.14,y,f'{v:.1f}',va='center',weight='bold')
        a=axes[1];arms=['vertical_client','vertical_client-online','vertical_client-power','vertical_client-cadence','vertical_client-video','vertical_full'];vals=[avg('wifi_vs_5g','main',n)*1e6 for n in arms]
        a.barh(np.arange(6),vals,color=[C['orange']]+[C['blue']]*4+[C['teal']],height=.6)
        a.set(yticks=range(6),yticklabels=['Full dual-radio','No online learning','No power control','No cadence timing','No adaptive video','Add scheduled control'],xlim=(0,5.2),xlabel='Misses / M',title='Other levers affect the margin');a.invert_yaxis()
        for y,v in enumerate(vals):a.text(v+.1,y,f'{v:.1f}',va='center',fontsize=17)
        fig.subplots_adjust(left=.15,right=.98,wspace=.75)
        return fig

    def execution_safety(self):
        fig,ax=canvas()
        nodes=[(.2,'PROBE','Fixed read-only commands','blue'),(4.15,'CONFIRM','Operator + typed target','orange'),(8.1,'REQUEST ONCE','Customer interface only','teal')]
        for x,title,body,col in nodes:
            box(ax,x,3.8,3.35,1.3,title,body,color=col,fs=19)
            if x<8:arrow(ax,(x+3.4,4.45),(x+3.9,4.45))
        for y,title,col in [(4.7,'APPLIED','teal'),(2.95,'NOT APPLIED','grey'),(1.2,'UNKNOWN','orange')]:
            box(ax,12.4,y-.4,3.2,1.0,title,color=col,fs=20);arrow(ax,(11.5,4.35),(12.35,y+.1),color=col)
        box(ax,4.15,.55,6.95,1.15,'Read status to reconcile an uncertain outcome',color='orange',fs=20)
        arrow(ax,(12.35,1.3),(11.15,1.1),color='orange')
        box(ax,.2,.55,3.35,1.15,'Management wlan0','Excluded from automatic control',color='grey',fs=17)
        return fig

    def operator_view(self):
        fig=plt.figure(figsize=(17.5,6.25),facecolor=C['bg'])
        gs=fig.add_gridspec(1,2,width_ratios=[1.8,1],wspace=.12)
        ax=fig.add_subplot(gs[0]);ax.imshow(plt.imread(ROOT/'docs/slides/assets/studio_fleet_map.png'));ax.axis('off')
        a=fig.add_subplot(gs[1]);a.set(xlim=(0,7),ylim=(0,6));a.axis('off')
        for i,(title,body,col) in enumerate([('FLEET STATE','Robot position / service / link context','blue'),('DECISIONS + OUTCOMES','Inspect proposal and execution records','orange'),('EVIDENCE VIEWS','Keep simulation claims auditable','teal')]):
            box(a,.15,4.5-i*1.8,6.65,1.2,title,body,color=col,fs=19)
        fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
        return fig

    def cellular_model(self):
        fig,ax=canvas()
        stages=[('Same sites','16 warehouse sites\nShared routes / blockage','ink'),('Cellular assumptions','n78 · 100 MHz · 23 dBm\nLicensed / clean spectrum','blue'),('Link abstraction','Scaled Wi-Fi NIST curves\nNot a 3GPP NR PHY','orange'),('Access + mobility','Ideal control scheduling\n0 or 30 ms interruption','blue')]
        for i,(t,b,c) in enumerate(stages):
            x=.3+i*3.95;box(ax,x,2.7,3.4,1.8,t,b,color=c,fs=20)
            if i<3:arrow(ax,(x+3.45,3.6),(x+3.9,3.6))
        text(ax,8,1.35,'No inter-cell interference · no measured private network · no dual-cell packet diversity',21,C['muted'],'center')
        text(ax,8,.5,'Favorable assumptions do not make an abstract comparator a validated product benchmark.',19,C['orange'],'center',weight='bold')
        return fig

    def protocols(self):
        fig,ax=canvas()
        box(ax,.2,4.45,5.3,1.1,'100 robots × 300 s × 20 Hz',color='blue',fs=25)
        box(ax,.2,2.85,5.3,1.1,'3.6 million commands per arm',color='teal',fs=23)
        text(ax,2.85,1.8,'Six paired scenes',24,C['ink'],'center',weight='bold')
        for i,scene in enumerate(range(530,536)):
            x=.2+i*.9;ax.add_patch(Circle((x+.4,.75),.33,fc='white',ec=C['blue'],lw=1.8));text(ax,x+.4,.75,str(scene),14,C['blue'],'center')
        for i,(title,body,col) in enumerate([('CLEAN','λ = 0 · interference off','teal'),('MAIN','λ = 0.01 · interference on','orange'),('STRESS','λ = 0.03 · +10 dB at 2.4 GHz','blue')]):
            box(ax,7,4.3-i*1.65,8.5,1.2,title,body,color=col,fs=22)
        text(ax,8,5.85,'Same routes, fading and blockage for every compared policy.',20,C['ink'],'center',weight='bold')
        return fig

    def main_results(self):
        fig,axes=chart();ax=axes[0]
        choices=[('vertical_client','Controller · two AP paths','orange'),('nr_daps','Cellular model · 0 ms handover','blue'),('nr_30ms','Cellular model · 30 ms handover','blue'),('vertical_single','Controller · one client radio','grey'),('mlo_split','Wi-Fi · fixed split rule','grey')]
        for i,(arm,label,col) in enumerate(choices):
            rows=records('wifi_vs_5g','main',arm);values=np.array([r['control_miss']*1e6 for r in rows]);v=values.mean()
            ax.barh(i,v,color=C[col],height=.55,alpha=.82);ax.scatter(values,i+np.linspace(-.1,.1,len(values)),s=42,c=C['ink'],zorder=5)
            ax.text(v*1.22+1,i,f'{v:,.1f}',va='center',fontsize=20,weight='bold')
        ax.set_yticks(range(5),[c[1] for c in choices]);ax.invert_yaxis();ax.set_xscale('symlog',linthresh=1)
        ax.set(xlim=(0,15000),xlabel='Misses per million · linear below 1, logarithmic above',title='Bars: six-scene means. Dots: actual paired scene outcomes.')
        ax.set_xticks([0,1,10,100,1000,10000],['0','1','10','100','1,000','10,000']);fig.subplots_adjust(left=.31,right=.95)
        n=sum(r['control_sent'] for r in records('wifi_vs_5g','main','vertical_client'))
        self.facts['main_results']={'commands_per_arm_condition':n,'dual_ppm':avg('wifi_vs_5g','main','vertical_client')*1e6,'cellular_30ms_ppm':avg('wifi_vs_5g','main','nr_30ms')*1e6,'ratio':avg('wifi_vs_5g','main','nr_30ms')/avg('wifi_vs_5g','main','vertical_client'),'independent_scene_realisations':6,'packet_iid_intervals_used':False}
        return fig

    def sla_results(self):
        fig,axes=plt.subplots(2,2,figsize=(17.5,6.25),facecolor=C['bg']);fig.subplots_adjust(left=.18,right=.97,bottom=.14,top=.92,wspace=.45,hspace=.6)
        metrics=[('availability_1s','Robot-second availability (%)',100),('control_gap_p99_ms','p99 robot-longest gap (ms)',1),('control_miss_worst_decile','Worst-decile control misses (%)',100),('safety_stop_fraction','Modelled stop-time fraction (%)',100)]
        for ax,(key,title,mult) in zip(axes.ravel(),metrics):
            for i,(arm,color) in enumerate([('vertical_client','orange'),('nr_30ms','blue')]):
                values=np.array([r[key]*mult for r in records('wifi_vs_5g','main',arm)]);v=values.mean()
                ax.barh(i,v,color=C[color],height=.55,alpha=.85);ax.scatter(values,i+np.linspace(-.09,.09,6),c=C['ink'],s=20)
                label=f'{v:.3f}' if key=='availability_1s' else f'{v:.4f}' if 'fraction' in key or key=='control_miss_worst_decile' else f'{v:.0f}'
                ax.text(.98,.8-i*.55,label,transform=ax.transAxes,ha='right',fontsize=18,weight='bold')
            ax.set_yticks([0,1],['Two AP paths','Cellular model'],fontsize=15);ax.invert_yaxis();ax.set_title(title,fontsize=18);ax.set_facecolor(C['bg']);ax.grid(axis='x',alpha=.13)
            if key=='availability_1s':
                ax.set_xlim(99.85,100.);ax.set_xticks([99.85,99.90,99.95,100.],['99.85','99.90','99.95','100.00'])
            ax.tick_params(axis='x',labelsize=13)
        return fig

    def stress(self):
        fig,axes=chart(2);ax=axes[0];conds=['lenient','main','stress'];labels=['Clean','Main','Stress'];y=np.arange(3)
        for offset,arm,col,label in [(-.17,'vertical_client','orange','Two AP paths'),(.17,'nr_30ms','blue','Cellular model')]:
            vals=[avg('wifi_vs_5g',c,arm)*1e6 for c in conds];ax.barh(y+offset,vals,height=.3,color=C[col],label=label)
            for yy,v in zip(y+offset,vals):ax.text(v+2,yy,f'{v:.1f}',va='center',fontsize=17)
        ax.set(yticks=y,yticklabels=labels,xlim=(0,125),xlabel='Control misses / million',title='The ranking reverses under stress');ax.invert_yaxis();ax.legend(frameon=False,fontsize=15)
        a=axes[1];a.set_facecolor(C['bg']);a.grid(alpha=.15)
        a.scatter([0,.01],[0,0],c=C['orange'],s=260,edgecolors='white',lw=2);a.scatter([.03],[10],c=C['blue'],s=260,edgecolors='white',lw=2)
        for x,y,t in [(0,0,'Clean'),(.01,0,'Main'),(.03,10,'Stress')]:a.annotate(t,(x,y),xytext=(8,12),textcoords='offset points',fontsize=18)
        a.set(xlim=(-.004,.038),ylim=(-2.5,13),xlabel='Collision-model parameter λ',ylabel='Extra 2.4 GHz noise (dB)',title='Only these configurations were tested')
        return fig

    def hardware_options(self):
        fig,ax=canvas();items=[('ONE BE200','Two station associations\nRetune after primary retries','Support unverified','blue'),('ONBOARD + BE200','Use the Pi radio as path B\nMove management out of band','Weaker link; operational rework','teal'),('SECOND CLIENT RADIO','Independent AP attachment\nUSB adapter / second card','Assumed in completed main test','orange')]
        for i,(title,body,limit,col) in enumerate(items):
            x=.25+i*5.25;box(ax,x,2.8,4.75,1.8,title,body,color=col,fs=22)
            ax.add_patch(Circle((x+1.5,1.65),.3,ec=C[col],fc='white',lw=2));ax.add_patch(Circle((x+3.2,1.65),.3,ec=C[col],fc='white',lw=2))
            arrow(ax,(x+1.85,1.65),(x+2.85,1.65),color=col)
            text(ax,x+2.4,.6,limit,18,C['muted'],'center',width=35)
        return fig

    def wifi8(self):
        fig,ax=canvas();xmin,xmax=2.8,15.4
        for y,title,col in [(4.5,'100 ms AP selection','blue'),(2.1,'10 ms retry opportunity','orange')]:
            text(ax,.2,y,title,19,C[col],width=18);ax.plot([xmin,xmax],[y,y],c=C['grey'],lw=2)
        for t in [0,100,200]:
            x=xmin+t/200*(xmax-xmin);ax.vlines(x,4.25,4.75,color=C['blue'],lw=4);text(ax,x,5.1,f'{t} ms',16,C['muted'],'center')
        for t in range(0,201,10):
            x=xmin+t/200*(xmax-xmin);ax.vlines(x,1.95,2.25,color=C['orange'],lw=2)
        x=xmin+40/200*(xmax-xmin);ax.axvspan(x,xmin+100/200*(xmax-xmin),ymin=.53,ymax=.84,color=C['pale_orange']);text(ax,x+.1,3.35,'Path fails between steering epochs',18,C['red'])
        arrow(ax,(x,1.2),(x+.65,1.95),color='teal');text(ax,9.1,.75,'304 / M vs 12.5 / M in one diagnostic scene—not a Wi-Fi 8 standard limit',19,C['ink'],'center')
        self.facts['wifi8']={'illustrative_timing_not_measurement':True,'development_only':True,'fresh_cross_ap_results_archived':False}
        return fig

    def traffic_control(self):
        fig,ax=canvas()
        rows=[(4.45,'CONTROL','Robust retry + deadline','AP A → AP B on failure','orange'),
              (2.75,'VIDEO','Limit encoded demand','Chosen share of goodput','blue'),
              (1.05,'BULK','Queue when link improves','Buffered delivery on fast link','teal')]
        for y,title,rule,target,col in rows:
            box(ax,.25,y-.35,2.5,1.05,title,color=col,fs=21)
            for k in range(4):ax.add_patch(Rectangle((3.3+k*.35,y-.14),.28,.45,facecolor=C[col],alpha=1-k*.16))
            box(ax,5.35,y-.35,4.35,1.05,rule,color=col,fs=20)
            box(ax,11.1,y-.35,4.55,1.05,target,color=col,fs=18)
            arrow(ax,(4.8,y+.15),(5.3,y+.15),color=col);arrow(ax,(9.75,y+.15),(11.05,y+.15),color=col)
        text(ax,8,5.7,'Application objectives reach the link scheduler through explicit class-specific controls.',20,C['ink'],'center',weight='bold')
        return fig

    def capability_map(self):
        labels=['Spatial map + uncertainty','Mission → native policy knobs','AP planning + trust gate','Control retries + cross-AP path','Traffic-to-link mapping','Video / bulk demand control','Delivery feedback + online map','Rich contract proposal interface','Supervised handover diagnostics']
        values=np.zeros((9,3));values[:7,0]=1;values[7:,1]=1
        fig,axes=chart();ax=axes[0]
        from matplotlib.colors import ListedColormap
        ax.imshow(values,cmap=ListedColormap([C['white'],C['pale_blue']]),vmin=0,vmax=1,aspect='auto')
        cols=['Executed simulation','Separate software tools','Field qualified']
        ax.set_xticks(range(3),cols,fontsize=18);ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=13)
        ax.set_yticks(range(9),labels,fontsize=17);ax.tick_params(axis='y',length=0,pad=10);ax.grid(False)
        for i in range(9):
            for j in range(3):ax.text(j,i,'●' if values[i,j] else '—',ha='center',va='center',c=C['blue'] if values[i,j] else C['grey'],fontsize=24)
        for y in np.arange(-.5,9,1):ax.axhline(y,c=C['grey'],lw=.5,alpha=.3)
        fig.subplots_adjust(left=.36,right=.98,top=.88,bottom=.05)
        return fig

    def validation(self):
        fig,ax=canvas();items=[('1 · CAPABILITY','Read-only probes\nDual association / driver / AP support'),('2 · MEASURE','Supervised roam + scan\nClocked packet capture + power'),('3 · CALIBRATE','Fit the reality gap\nReplay frozen comparisons'),('4 · FIELD COMPARE','Small fleet → larger fleet\nActual private-5G test route')]
        for i,(t,b) in enumerate(items):
            x=.2+i*3.95;y=3.55-i*.55;box(ax,x,y,3.45,1.55,t,b,color='blue' if i<2 else 'orange',fs=19)
            if i<3:arrow(ax,(x+3.5,y+.6),(x+3.9,y+.05))
        ax.plot([.4,15.45],[.75,.75],c=C['grey'],ls='--');text(ax,8,.32,'No stage has yet produced physical fleet qualification.',20,C['muted'],'center')
        return fig

    def bom(self):
        fig,ax=canvas();items=[('4','Wi-Fi 7 APs','blue'),('10','Robot milestone','ink'),('12','Conditional adapters\n10 + 2 spares','orange'),('2','Capture adapters\n+ one laptop','teal'),('2','Power meters','blue')]
        for i,(number,label,col) in enumerate(items):
            x=.3+i*3.15;ax.add_patch(Circle((x+1.2,4.1),.7,ec=C[col],fc=C['white'],lw=2.5));text(ax,x+1.2,4.1,number,36,C[col],'center',weight='bold');text(ax,x+1.2,2.65,label,20,C['ink'],'center',width=20)
        box(ax,.3,.45,7.1,1.1,'Management: 10 out-of-band paths if onboard Wi-Fi is reused',color='grey',fs=18)
        box(ax,8,.45,7.25,1.1,'Partner lab: private-5G network + 2+ modems',color='blue',fs=19)
        return fig

    def close(self):
        fig=plt.figure(figsize=(17.5,6.25),facecolor=C['bg']);gs=fig.add_gridspec(1,2,width_ratios=[1.15,1.25],wspace=.15)
        a=fig.add_subplot(gs[0]);self.warehouse_background(a)
        ax=fig.add_subplot(gs[1]);ax.set(xlim=(0,8),ylim=(0,6));ax.axis('off')
        for i,(title,body,col) in enumerate([('LEARN','Site map + explicit uncertainty','blue'),('DECIDE','Mission-driven numerical policy + gate','orange'),('RECOVER','Another AP path when one fails','teal')]):
            box(ax,.2,4.45-i*1.8,7.5,1.2,title,body,color=col,fs=22)
        text(ax,4,.2,'Next: measured interruption and a controlled pilot.',18,C['ink'],'center',weight='bold')
        return fig

    def build(self,ids):
        for i,name in enumerate(ids,1):
            method=getattr(self,name,None)
            if method is None:raise ValueError('No substantive visual for '+name)
            figure=method();save(figure,self.out,name)
            print(f'Figure {i}/{len(ids)}: {name}',flush=True)
        return self.facts
