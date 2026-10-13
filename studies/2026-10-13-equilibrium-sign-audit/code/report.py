"""Make tables and figures from the fixed saved results. No model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def table(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    assert np.__version__=='2.3.5' and matplotlib.__version__=='3.10.8'
    manifest=json.loads((args.results/'manifest.json').read_text())
    for name,value in manifest.items():assert hashlib.sha256((args.results/name).read_bytes()).hexdigest()==value
    source=json.loads((args.results/'summary.json').read_text());models=source['models']
    seeds=[];outcomes=[];costs=[];bins=[];reversals=[];steps_all=[];ordinary_all=[]
    for m in models:
        seed=m['seed'];p=args.results/str(seed)
        coarse=np.load(p/'coarse.npz');initial=np.load(p/'initial-bounds.npz');final=np.load(p/'refined.npz')
        naive=coarse['naive'];nonzero=naive!=0;unresolved=(initial['lower']<=0)&(initial['upper']>=0)
        fraction=float((nonzero&unresolved).sum()/nonzero.sum())
        assert fraction==m['primary_fraction']
        steps=final['extra_updates'];steps_all.extend(steps.tolist());ordinary_all.extend(naive.tolist())
        seeds.append(dict(seed=seed,accuracy=m['full_test_accuracy'],panel_accuracy=m['panel_accuracy'],
            first_accuracy=m['first_update_accuracy'],zero_accuracy=m['zero_state_accuracy'],
            first_disagreement=m['first_update_disagreement'],zero_disagreement=m['zero_state_disagreement'],
            nonzero_ordinary=int(nonzero.sum()),unresolved_nonzero=int((nonzero&unresolved).sum()),
            primary_fraction=fraction,ordinary_zero=int((~nonzero).sum()),unresolved_all=int(unresolved.sum()),
            certified_reversals=m['certified_sign_reversals']))
        outcomes.append(dict(seed=seed,**m['final_counts'],extra_updates_total=int(steps.sum()),
            extra_updates_mean=float(steps.mean()),extra_updates_max=int(steps.max())))
        costs.append({key:m[key] for key in ['seed','training_updates','training_examples','training_state_update_examples',
            'ordinary_map_calls','ordinary_map_examples','certificate_calls','certificate_examples',
            'certificate_map_passes','certificate_map_examples','input_affine_enclosure_calls',
            'input_affine_enclosure_examples','extra_paired_updates','refinement_rows']})
        for row in m['bins']:
            bins.append(dict(seed=seed,bin=row['bin'],count=row['count'],ordinary_mean=row['ordinary_mean'],
                initial_width_mean=row['initial_width_mean'],final_width_mean=row['final_width_mean'],**row['final_counts']))
        reversed_ids=np.flatnonzero(((naive<0)&(final['status']=='positive'))|((naive>0)&(final['status']=='negative')))
        assert len(reversed_ids)==m['certified_sign_reversals']
        for i in reversed_ids:
            reversals.append(dict(seed=seed,pair=int(i),test_row=int(coarse['panel_rows'][coarse['pair_image'][i]]),
                unit=int(coarse['pair_unit'][i]),winner=int(coarse['winner'][i]),runner=int(coarse['runner'][i]),
                ordinary_effect=float(naive[i]),initial_lower=float(initial['lower'][i]),initial_upper=float(initial['upper'][i]),
                final_lower=float(final['lower'][i]),final_upper=float(final['upper'][i]),extra_updates=int(steps[i])))
    for name,rows in [('seeds',seeds),('outcomes',outcomes),('costs',costs),('bins',bins),('reversals',reversals)]:
        table(args.output/(name+'.csv'),rows)
    steps_all=np.array(steps_all);ordinary_all=np.array(ordinary_all)
    totals=dict(primary_mean=float(np.mean([r['primary_fraction'] for r in seeds])),
        effects=len(steps_all),nonzero_ordinary=sum(r['nonzero_ordinary'] for r in seeds),
        unresolved_nonzero=sum(r['unresolved_nonzero'] for r in seeds),
        unresolved_all=sum(r['unresolved_all'] for r in seeds),ordinary_zero=sum(r['ordinary_zero'] for r in seeds),
        final_counts={key:sum(r[key] for r in outcomes) for key in ('positive','negative','small','unresolved')},
        extra_updates_total=int(steps_all.sum()),extra_updates_mean=float(steps_all.mean()),
        extra_updates_max=int(steps_all.max()),zero_extra_updates=int((steps_all==0).sum()),
        certified_reversals=len(reversals),costs={key:sum(r[key] for r in costs) for key in costs[0] if key!='seed'},
        table_rows=sum(map(len,[seeds,outcomes,costs,bins,reversals])),scientific_model_calls=0)
    dump(args.output/'report.json',totals)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained');x=np.arange(4);labels=[str(r['seed']) for r in seeds]
    a=axes[0,0];values=[100*r['accuracy'] for r in seeds]
    a.bar(x,values,color='#377a9c');a.axhline(80,color='#555555',ls='--',lw=1,label='Fixed ability gate')
    a.set(ylim=(0,100),title='A. All four models pass the task check',ylabel='Test accuracy (%)',xticks=x,xticklabels=labels)
    for i,v in enumerate(values):a.text(i,v+1,f'{v:.2f}',ha='center',fontsize=9)
    a.legend(frameon=False,loc='lower right')
    a=axes[0,1];values=[100*r['primary_fraction'] for r in seeds]
    a.bar(x,values,color='#b87437');a.set(ylim=(0,25),title='B. Some nonzero signs are not established',
        ylabel='Unresolved at coarse stop (%)',xticks=x,xticklabels=labels)
    for i,v in enumerate(values):a.text(i,v+.4,f'{v:.2f}',ha='center',fontsize=9)
    a=axes[1,0];bottom=np.zeros(4)
    for key,color,label in [('positive','#377a9c','Positive'),('negative','#98485d','Negative'),('small','#bfbfbf','Within ±10⁻⁷')]:
        values=np.array([r[key] for r in outcomes]);a.bar(x,values,bottom=bottom,color=color,label=label);bottom+=values
    a.set(title='C. Every case gets a sign\nor small-effect bound',ylabel='Number of effects',xticks=x,xticklabels=labels,ylim=(0,4100))
    a.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=9)
    a=axes[1,1];a.hist(steps_all,bins=np.arange(-.5,int(steps_all.max())+1.5),color='#5c8c61')
    a.set(title='D. Most effects need few extra steps',xlabel='Extra paired updates',ylabel='Number of effects')
    fig.suptitle('A small state residual does not establish every effect sign',fontsize=14)
    fig.savefig(args.output/'equilibrium-signs.png',dpi=160)
    fig.savefig(args.output/'equilibrium-signs.pdf')
    dump(args.output/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir()) if p.name!='manifest.json'})
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
