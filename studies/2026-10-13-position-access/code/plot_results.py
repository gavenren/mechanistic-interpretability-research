#!/usr/bin/env python3
"""Build the declared-condition figure from saved result summaries."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
summary = json.loads((root/'results/summary.json').read_text())
cells = {(r['gap'],r['position'],r['access']):r for r in summary['cells'] if r['mode']=='intact'}
styles = [(1,1,'Native positions, readable fillers','#253858','-'),
          (1,0,'Native positions, blocked fillers','#b46b27','--'),
          (0,1,'Compressed positions, readable fillers','#8560aa','-.'),
          (0,0,'Compressed positions, blocked fillers','#177d6b',':')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes = plt.subplots(1,2,figsize=(11.8,6.1),gridspec_kw={'width_ratios':[1.1,1]})
for p,a,label,color,style in styles:
    axes[0].plot([0,32,128],[cells[g,p,a]['margin'] for g in (0,32,128)],
                 label=label,color=color,linestyle=style,linewidth=2.3,marker='o',markersize=5)
axes[0].set(xlabel='Added tokens per gap (two gaps)',ylabel='Mean correct-minus-other score (logits)',
            xticks=[0,32,128],ylim=(-.3,6.5),title='The four conditions have different joint effects')
axes[0].grid(alpha=.18)
native = cells[128,1,1]['margin']
repairs = [cells[128,1,0]['margin']-native,cells[128,0,1]['margin']-native]
axes[1].bar([0,1],repairs,color=['#b46b27','#8560aa'],width=.6)
axes[1].axhline(0,color='#555555',linewidth=.8)
axes[1].set(xticks=[0,1],xticklabels=['Block fillers\n(native positions)','Compress positions\n(readable fillers)'],
            ylabel='Score change from native-readable input',ylim=(-5.9,.55),
            title='At gap 128, each single change lowers the score')
for x,value in enumerate(repairs):
    axes[1].text(x,value-.18,f'{value:+.3f}',ha='center',va='top',fontweight='bold')
primary = summary['primary']
lo,hi = primary['interval_95']
axes[1].text(.5,.87,f'Fixed difference: {primary["mean"]:+.3f} logits\n95% group interval: [{lo:+.3f}, {hi:+.3f}]',
             transform=axes[1].transAxes,ha='center',va='top',fontsize=10,
             bbox={'facecolor':'white','edgecolor':'#dddddd','boxstyle':'round,pad=.5'})
fig.suptitle('Position spacing and access to extra tokens',fontsize=16,x=.51,y=.97)
fig.text(.51,.915,'Pythia-14M · 32 token groups · all 256 core cases retained',ha='center',color='#555555')
handles,labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=9)
fig.subplots_adjust(left=.075,right=.975,top=.81,bottom=.23,wspace=.32)
out = root/'figures'
out.mkdir(exist_ok=True)
fig.savefig(out/'position-access.png',dpi=180)
fig.savefig(out/'position-access.svg')
plt.close(fig)
