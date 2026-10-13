#!/usr/bin/env python3
"""Report the post-failure follow-up. Read saved summaries; do not call models."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'results-float64/model-summaries.json').read_text())
summary = json.loads((ROOT / 'results-float64/summary.json').read_text())
assert [(r['seed'], r['architecture']) for r in rows] == [(s,a) for s in range(8) for a in ['direct','filtered']]
figdir = ROOT / 'figures'; figdir.mkdir(exist_ok=True)
tabledir = ROOT / 'tables'; tabledir.mkdir(exist_ok=True)

def write_csv(name, items):
    with (tabledir / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0]))
        writer.writeheader(); writer.writerows(items)

scalar_keys = [k for k,v in rows[0].items() if not isinstance(v,list)]
write_csv('model-summaries.csv', [{k:r[k] for k in scalar_keys} for r in rows])
for group in ['channel','digit','shift']:
    write_csv(f'by-{group}.csv', [dict(seed=r['seed'], architecture=r['architecture'], **g)
                               for r in rows for g in r['by_'+group]])
write_csv('paired-seed-differences.csv', [dict(seed=s, direct_D=rows[2*s]['D'],
           filtered_D=rows[2*s+1]['D'], difference=summary['paired_seed_differences'][s]) for s in range(8)])
metric_keys = [k for k in scalar_keys if k not in ['seed','architecture','initial_parameter_sha256','reused_training']]
write_csv('architecture-means.csv', [dict(architecture=a, **{k:float(np.mean([r[k] for r in rows if r['architecture']==a]))
                                                          for k in metric_keys}) for a in ['direct','filtered']])

colors = {'direct':'#B65B30','filtered':'#176E9B'}
labels = {'direct':'Direct sampling','filtered':'Filtered sampling'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'shift-ablation-followup'})
fig, axs = plt.subplots(2,2,figsize=(11.8,8.6))

def values(arch,key):
    return np.array([r[key] for r in rows if r['architecture']==arch])

def grouped(ax,keys,category_labels,scale=1):
    for j,arch in enumerate(colors):
        xx=np.arange(len(keys))+(j-.5)*.3
        ax.bar(xx,[scale*values(arch,key).mean() for key in keys],width=.28,
               color=colors[arch],label=labels[arch],alpha=.85)
        for x,key in zip(xx,keys):
            ax.scatter(x+np.linspace(-.07,.07,8),scale*values(arch,key),s=15,color='#333333',zorder=3)
    ax.set_xticks(range(len(keys)),category_labels)

ax=axs[0,0]
grouped(ax,['accuracy_original','accuracy_shifted'],['Original images','One-pixel shifts'],scale=100)
ax.set_ylim(0,100); ax.set_ylabel('Digit accuracy (%)')
ax.set_title('A  Low digit accuracy limits the claim',loc='left',fontweight='bold')
ax.text(.02,.95,'Original-image accuracy across all models:\n14.91% to 35.95%',
        transform=ax.transAxes,va='top',fontsize=10)
ax.legend(frameon=False,fontsize=9,loc='upper right',bbox_to_anchor=(1,.77))

ax=axs[0,1]
a,b=values('direct','D'),values('filtered','D')
for i in range(8): ax.plot([0,1],[a[i],b[i]],color='#BBBBBB',lw=1,zorder=1)
for x,arch,v in [(0,'direct',a),(1,'filtered',b)]:
    ax.scatter(np.full(8,x),v,color=colors[arch],s=34,zorder=2)
    ax.scatter([x],[v.mean()],marker='_',s=330,color='black',linewidths=2,zorder=3)
ax.set_xticks([0,1],[labels['direct'],labels['filtered']]); ax.set_xlim(-.3,1.3)
ax.set_ylim(0,.14); ax.set_ylabel('Change in signed removal effect (D)')
ax.set_title('B  Removal effects change less after filtering',loc='left',fontweight='bold')
lo,hi=summary['percentile_95_interval']
ax.text(.02,.97,f'Mean paired difference: {summary["mean"]:.4f}\n95% seed interval: [{lo:.4f}, {hi:.4f}]',
        transform=ax.transAxes,va='top',fontsize=9)

ax=axs[1,0]
grouped(ax,['baseline_shift_tv','ablated_shift_tv'],['Intact output','After channel removal'])
ax.set_ylabel('Output change under one-pixel shifts (TV)')
ax.set_title('C  Both output distributions change less',loc='left',fontweight='bold')

ax=axs[1,1]
for x,arch in enumerate(colors):
    v=values(arch,'effect_size_original')
    ax.bar(x,v.mean(),width=.5,color=colors[arch],alpha=.85)
    ax.scatter(x+np.linspace(-.12,.12,8),v,s=23,color='#333333',zorder=3)
ax.set_xticks([0,1],[labels['direct'],labels['filtered']])
ax.set_ylabel('Effect size on original images (TV)')
ax.set_title('D  Mean channel effects are not weaker',loc='left',fontweight='bold')
for ax in axs.flat:
    ax.grid(axis='y',color='#E6E6E6',linewidth=.6); ax.set_axisbelow(True)
fig.suptitle('Weak digit classifiers: stable removal effects do not establish good explanations',
             fontsize=14,fontweight='bold',y=.99)
fig.text(.5,.018,'Post-failure float64 evaluation of float32-trained models. Points show all eight models per architecture.\nLines in B join the same seed. All 1,797 test images and eight channels are included.',
         ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.96),h_pad=2,w_pad=2)
fig.savefig(figdir/'shift-ablation.png',dpi=180,bbox_inches='tight')
fig.savefig(figdir/'shift-ablation.svg',bbox_inches='tight',metadata={'Date':None})
plt.close(fig)
print('Saved the figure and six descriptive tables. No model was called.')
