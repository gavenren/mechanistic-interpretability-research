#!/usr/bin/env python3
"""Build report tables and a figure from saved results. No model calls."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT/'results/summary.json').read_text())
seeds = summary['seeds']
stages = ['immediate','copy_write_first','copy_write_final','write_copy_first','write_copy_final']
(ROOT/'tables').mkdir(exist_ok=True)

with (ROOT/'tables/stage-scores.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['seed','route','stage','accuracy','mean_target_margin','mean_target_probability'])
    for row in seeds:
        for route, scores in row['routes'].items():
            for stage in stages:
                writer.writerow([row['seed'],route,stage,scores[stage+'_accuracy'],
                                 scores[stage+'_margin'],scores[stage+'_probability']])
with (ROOT/'tables/route-scores.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['seed','route','primary_success','immediate_A_accuracy','immediate_B_accuracy',
                     'mean_edit_norm','same_donor_B_success','different_donor_B_success'])
    for row in seeds:
        for route, scores in row['routes'].items():
            writer.writerow([row['seed'],route,scores['contract_success'],scores['immediate_A_accuracy'],
                             scores['immediate_B_accuracy'],scores['mean_edit_norm'],
                             scores['donor_B_strata']['same']['contract_success'],
                             scores['donor_B_strata']['different']['contract_success']])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none'})
fig, axes = plt.subplots(1,2,figsize=(11.8,4.8),gridspec_kw={'width_ratios':[1,1.3]})
colors = ['#3a6b99','#8b6c8d','#207b63','#b27843']
for row, color in zip(seeds, colors):
    values = [100*row['routes'][arm]['contract_success'] for arm in ('immediate','transition')]
    label = str(row['seed']) + (' (ability pass)' if row['ability']['passes'] else ' (ability fail)')
    axes[0].plot([0,1],values,'o-',color=color,label=label,linewidth=1.6)
axes[0].set_xticks([0,1],['Immediate loss','Immediate +\none-step loss'])
axes[0].set_xlim(-.15,1.15)
axes[0].set_ylim(86,95)
axes[0].set_ylabel('Primary test success (%)')
axes[0].set_title('A. Four model seeds',loc='left')
axes[0].legend(loc='lower right',fontsize=8,frameon=False)
metrics = ['immediate_accuracy','copy_write_final_accuracy','write_copy_final_accuracy','contract_success']
labels = ['Immediate\nstate','Copy then\nwrite','Write then\ncopy','Primary\nconjunction']
values = np.array([[100*(row['routes']['transition'][metric]-row['routes']['immediate'][metric])
                    for metric in metrics] for row in seeds])
axes[1].bar(np.arange(4),values.mean(0),color='#8aacbf',width=.55,alpha=.6)
for i,color in enumerate(colors):
    axes[1].scatter(np.arange(4)+(i-1.5)*.055,values[i],color=color,s=30,zorder=3)
axes[1].axhline(0,color='#444444',linewidth=.8)
axes[1].set_xticks(range(4),labels)
axes[1].set_ylim(-2.3,2.15)
axes[1].set_ylabel('Gain from one-step loss (percentage points)')
axes[1].set_title('B. Gains and losses by test',loc='left')
fig.suptitle('A future-step loss gives a small, uncertain gain on the full test',fontsize=14,y=1.01)
fig.text(.02,.005,'Each dot is one trained GRU. Bars in B show the four-seed mean. Left axis starts at 86%.\n'
         'Primary mean gain: +0.21 points; descriptive seed interval: −1.02 to +1.13. Three models fail the fixed ability rule.',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.12,1,.97])
(ROOT/'figures').mkdir(exist_ok=True)
for suffix in ('png','svg'):
    fig.savefig(ROOT/'figures'/('write-order-results.'+suffix),dpi=180,bbox_inches='tight')
