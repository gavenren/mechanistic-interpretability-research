#!/usr/bin/env python3
"""Plot all six seed outcomes in every fixed factorial cell."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'results'/'selected-latents.csv').open()))
assert len(rows)==72
labels=('Local','Downstream','Balanced')
methods=('local','downstream','balanced')
colors=('#4279a6','#df8c32','#448a68')
fig,axes=plt.subplots(2,2,figsize=(9.4,6.8),sharey=True)
for ax,(p,a) in zip(axes.flat,((.01,.25),(.01,1.),(.16,.25),(.16,1.))):
    for j,method in enumerate(methods):
        selected=sorted([r for r in rows if float(r['p'])==p and float(r['amplitude'])==a and r['objective']==method],key=lambda r:int(r['seed']))
        assert len(selected)==6
        scores=np.array([float(r['test_native_error']) for r in selected])
        ax.scatter(j+np.linspace(-.07,.07,6),scores,color=colors[j],s=36,alpha=.9,zorder=3)
        ax.hlines(scores.mean(),j-.22,j+.22,color=colors[j],linewidth=3,zorder=4)
    ax.axhline(.9640275800758169,color='#777777',linestyle=':',linewidth=1)
    ax.set_title(f'Frequency {100*p:g}% | amplitude {a:g}',fontsize=11)
    ax.set_xticks(range(3),labels,fontsize=10)
    ax.set_ylim(-.035,1.02)
    ax.set_yticks((0,.25,.5,.75,1))
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
for ax in axes[:,0]: ax.set_ylabel('Single-feature transport error\n(lower is better)',fontsize=10)
fig.suptitle('Single-feature transport across frequency and amplitude',fontsize=14,y=.99)
fig.text(.5,.014,'Dots: six fixed seeds. Bars: mean. Dotted line: no intervention.\nAll panels use the same scale. The balanced objective uses true event labels.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.95))
out=root/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'feature-transport.png',dpi=180)
fig.savefig(out/'feature-transport.pdf')
