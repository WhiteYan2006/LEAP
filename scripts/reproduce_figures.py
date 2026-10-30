"""Regenerate paper aggregate plots without prompts, model weights or a GPU.

Values and cohort definitions are verified before rendering. Font availability
can change PDF bytes; exported plotted values must not change.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'paper_data'
NAMES={'LIE':'LEAP','LIE+RLS':'LEAP+RLS','SSJF-N':'SSJF','TRAIL-init':'TRAIL',
       'lie':'LEAP','lie_sqrt':'LEAP','tie':'TIE','tie_expectation':'TIE',
       'ltr':'LTR','ltr_cal':'LTR','ssjf':'SSJF','trail':'TRAIL',
       'egtp_hidden_fp32':'EGTP-P','egtp_logits':'EGTP-L','fcfs':'FCFS'}
COLORS={'LEAP':'#2878B5','TIE':'#DF8B33','LTR':'#3C8C60','SSJF':'#C86A7A',
        'TRAIL':'#8172B3','EGTP':'#777777','EGTP-P':'#777777','EGTP-L':'#777777','FCFS':'#7A838C'}
MARKERS=['o','^','D','s','v','P','X']


def load_verified():
    proof=json.loads((DATA/'PROVENANCE.json').read_bytes());rows={}
    for name,info in proof['files'].items():
        p=DATA/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'],name
        with p.open(encoding='utf8',newline='') as f:
            rows[name]=list(csv.DictReader(f))
        assert len(rows[name])==info['rows'],name
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    rows=load_verified();out=args.output;out.mkdir(parents=True,exist_ok=False);values=[]
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    def record(figure,panel,method,x,y):
        assert math.isfinite(float(x)) and math.isfinite(float(y))
        values.append(dict(figure=figure,panel=panel,method=method,x=float(x),y=float(y)))

    # Figure5: exact LMSYS deduplication and the original per-target cohort.
    r=rows['figure5_ood.csv'];methods=['LIE','TIE','TRAIL-init','LTR','SSJF-N','EGTP'];datasets=['lmsys','alpaca','math','aya']
    fig,ax=plt.subplots(figsize=(6.6,3));x=np.arange(6);width=.18
    for j,dataset in enumerate(datasets):
        y=[]
        for method in methods:
            matches=[v for v in r if v['method']==method and (v['condition']=='LMSYS ID' if dataset=='lmsys' else v['target']==dataset and v['condition']!='LMSYS ID')]
            assert len(matches)==(3 if dataset=='lmsys' else 1)
            assert len({(v['rho'],v['n']) for v in matches})==1
            score=float(matches[0]['rho']);y.append(score);record('figure5','spearman',NAMES.get(method,method),j,score)
        ax.bar(x+(j-1.5)*width,y,width,label=dataset.upper())
    for j,dataset in enumerate(datasets[1:],1):
        matches=[v for v in r if v['method']=='LIE+RLS' and v['target']==dataset and v['condition']!='LMSYS ID']
        assert len(matches)==1
        score=float(matches[0]['rho'])
        ax.scatter((j-1.5)*width,score,marker='_',s=140,color='#205830',zorder=5,
                   label='LEAP+RLS' if j==1 else None)
        record('figure5','spearman','LEAP+RLS',j,score)
    ax.set_xticks(x,[NAMES.get(m,m) for m in methods]);ax.set_ylabel('Spearman correlation')
    ax.legend(ncol=5,fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.01));save(fig,'figure5_ood')

    # Figure6: all20 points use the same first1024 requests and shared protocol.
    r=rows['figure6_shared1024.csv'];rates=[20,40,60,80,100];arms=['fcfs','lie_sqrt','tie_expectation','ltr_cal']
    index={(v['arm'],int(v['rate_rps'])):v for v in r};assert set(index)=={(m,k) for m in arms for k in rates}
    assert all(int(v['requests'])==1024 and int(v['actual_output_tokens'])==544047 for v in r)
    panels=[('mean_E2E_per_output_token_s','Mean PTL (s/token)'),('p90_E2E_per_output_token_s','P90 PTL (s/token)'),('mean_TTFT_s','Mean TTFT (s)'),('p90_TTFT_s','P90 TTFT (s)')]
    fig,axes=plt.subplots(2,2,figsize=(6.6,5.4))
    for ax,(key,title) in zip(axes.flat,panels):
        for i,m in enumerate(arms):
            y=[float(index[m,k][key]) for k in rates];name=NAMES[m]
            ax.plot(rates,y,marker=MARKERS[i],color=COLORS[name],label=name)
            for k,value in zip(rates,y):record('figure6',key,name,k,value)
        ax.set_title(title);ax.set_xlabel('Offered requests/s');ax.set_ylim(bottom=0);ax.set_xlim(0,105)
    axes[0,0].legend(fontsize=7);fig.tight_layout();save(fig,'figure6_shared1024')

    # Figure7: all42 newly uniform-clocked B8 cells; real nonmonotonicity retained.
    r=rows['figure7_cpu.csv'];methods=['lie','trail','tie','ltr','ssjf','egtp_hidden_fp32','egtp_logits'];rates=[1,10,100,1000,10000,100000]
    assert {(v['method'],int(v['offered_predictions_s'])) for v in r}=={(m,k) for m in methods for k in rates}
    assert all(int(v['batch'])==8 and int(v['unfinished_batches'])==0 and v['generator_limited'].lower()=='false' for v in r)
    fig,axes=plt.subplots(3,1,figsize=(4.2,5.3),sharex=True)
    panels=[('delivered_predictions_s','Delivered predictions/s'),('p95_delivery_ms','P95 delivery (ms)'),('predict_and_consume_cpu_us_per_prediction','CPU (us/score)')]
    for i,m in enumerate(methods):
        series=sorted((v for v in r if v['method']==m),key=lambda v:int(v['offered_predictions_s']));name=NAMES[m]
        x=[float(v['scheduled_effective_predictions_s']) for v in series]
        for ax,(key,label) in zip(axes,panels):
            y=[float(v[key]) for v in series];assert all(v>0 for v in y)
            ax.plot(x,y,color=COLORS[name],marker=MARKERS[i],label=name,ms=3,linestyle='--' if m=='egtp_logits' else '-')
            ax.set_xscale('log');ax.set_yscale('log');ax.set_ylabel(label)
            for xx,yy in zip(x,y):record('figure7',key,name,xx,yy)
    axes[0].legend(ncol=4,fontsize=6,loc='upper left',bbox_to_anchor=(0,1.38));axes[-1].set_xlabel('Offered predictions/s');fig.tight_layout();save(fig,'figure7_cpu')

    # Figure8: aggregate counts; no fabricated timing margin or confidence interval.
    r=rows['figure8_overlap.csv'];batches=[1,8,32];modes=['async_fp32','async_fp16'];fig,ax=plt.subplots(figsize=(4.2,2.7));x=np.arange(3)
    for j,mode in enumerate(modes):
        y=[]
        for batch in batches:
            row=next(v for v in r if int(v['batch'])==batch and v['mode']==mode);assert int(row['trials'])==24
            score=100*float(row['ready_before_final_norm_fraction']);y.append(score);record('figure8','ready_percent',mode,batch,score)
        ax.bar(x+(j-.5)*.30,y,width=.28,label=mode)
    ax.set_xticks(x,batches);ax.set_xlabel('Batch size');ax.set_ylabel('Ready at probe (%)');ax.set_ylim(0,115);ax.legend(fontsize=7);save(fig,'figure8_overlap')

    # Appendix training curves: only the151 recorded objectives; no smoothed epochs.
    r=rows['baseline_training.csv'];groups=[]
    for row in r:
        if row['method'] not in groups:groups.append(row['method'])
    fig,axes=plt.subplots(2,3,figsize=(8,4.6))
    assert len(groups)==6
    for ax,method in zip(axes.flat,groups):
        subset=[v for v in r if v['method']==method]
        for kind in dict.fromkeys(v['series'] for v in subset):
            series=sorted((v for v in subset if v['series']==kind),key=lambda v:int(v['epoch']))
            x=[int(v['epoch']) for v in series];y=[float(v['value']) for v in series];ax.plot(x,y,label=kind)
            for xx,yy in zip(x,y):record('training',kind,method,xx,yy)
        ax.set_title(method);ax.set_xlabel('Epoch');ax.set_ylabel('Recorded objective');ax.legend(fontsize=6)
    fig.tight_layout();save(fig,'baseline_training')
    with (out/'PLOT_VALUES.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    (out/'REPRODUCTION.json').write_text(json.dumps(dict(passed=True,aggregate_plots=5,
         figure5_values=27,figure6_points=20,figure7_cells=42,figure7_plot_values=126,training_epoch_values=151,
         model_or_GPU_run=False,input_provenance_sha256=hashlib.sha256((DATA/'PROVENANCE.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(passed=True,plots=5,output=str(out))))

if __name__=='__main__':main()
