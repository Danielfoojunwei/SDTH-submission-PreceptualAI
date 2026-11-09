"""Export plots of actual archived v3 evidence with explicit simulation scope."""
from pathlib import Path
import argparse
import json
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=Path('artifacts/warehouse_v3'));args=p.parse_args()
    root=args.archive;snapshot=json.loads((root/'snapshot.json').read_text());core=json.loads((root/'core/results.json').read_text())
    matplotlib.rcParams['svg.hashsalt']='warehouse-v3'
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    domains=['ordinary','layout','fleet200','traffic'];x=np.arange(4)
    for arm,label in [('numerical','Numerical belief'),('mlp','MLP'),('graph_scratch','Graph from scratch'),('graph_masked','Masked pretraining'),('graph_jepa','JEPA-style pretraining')]:
        values=[];lower=[];upper=[]
        for domain in domains:
            rows=[r['summary']['service_rmse'] for r in core['results'] if r['domain']==domain and
                (r['method']=='numerical' if arm=='numerical' else r['method'] in [f'{arm}-{i}' for i in [0,1,2]])]
            values.append(np.mean(rows));lower.append(min(rows));upper.append(max(rows))
        axes[0].plot(x,values,'o-',label=label,linewidth=1.5)
        if arm!='numerical':axes[0].fill_between(x,lower,upper,alpha=.1)
    axes[0].set(xticks=x,xticklabels=['Ordinary','Layout','200 robots','3x bulk'],ylabel='Service-fraction RMSE',title='Held-out radio forecasts')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    base=snapshot['packet_persistence']['late_arrival_risk']['brier']
    values=[base]+[r['calibrated_late_event_brier'] for r in snapshot['packet_predictors']]
    bars=axes[1].bar(['Persistence','Instant model','Temporal model'],values,color=['#64748b','#c084fc','#0d9488'])
    for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value+.002,f'{value:.3f}',ha='center',fontsize=10)
    axes[1].set(ylabel='Late-arrival event Brier score',title='Actual native packet-event predictions',ylim=(0,max(values)*1.2))
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Wi-Fi warehouse simulation: representation and packet-event audits',fontsize=12)
    fig.savefig(root/'forecast_and_packet_events.png',dpi=180)
    vector=root/'forecast_and_packet_events.svg';fig.savefig(vector,metadata={'Date':None})
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n');plt.close(fig)
    print('Exported actual v3 forecast and native-event figures')


if __name__=='__main__':main()
