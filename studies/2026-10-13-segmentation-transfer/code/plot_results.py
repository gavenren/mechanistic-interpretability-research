#!/usr/bin/env python3
"""Plot the fixed study's saved estimates; does not call the model."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'results/summary.json').read_text())['estimates']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1.35,1]})
labels=['Matching splits\nAA and BB','Different splits\nAB and BA'];x=np.arange(2);width=.32
for offset,metric,color,label in [(-width/2,'intact_pair_correct','#285880','Correct choice from 2 values'),(width/2,'intact_full_correct','#ca753b','Correct top prediction')]:
 vals=[100*s[t][metric]['mean'] for t in ('matched','mismatched')];low=[vals[i]-100*s[t][metric]['lower'] for i,t in enumerate(('matched','mismatched'))];high=[100*s[t][metric]['upper']-vals[i] for i,t in enumerate(('matched','mismatched'))]
 ax[0].bar(x+offset,vals,width,color=color,label=label,yerr=np.array([low,high]),capsize=4)
 for j,(pos,val) in enumerate(zip(x+offset,vals)):ax[0].text(pos,max(val+high[j]+2.5,3),f'{val:.1f}%',ha='center',va='bottom',fontsize=10)
ax[0].set_xticks(x,labels);ax[0].set_ylabel('Intact model accuracy (%)');ax[0].set_ylim(0,114);ax[0].legend(loc='upper right',fontsize=9,frameon=False,bbox_to_anchor=(1.01,1.13));ax[0].set_title('A. Binding ability',loc='left',pad=30)
vals=[s[t]['effect']['mean'] for t in ('matched','mismatched')];err=np.array([[vals[i]-s[t]['effect']['lower'] for i,t in enumerate(('matched','mismatched'))],[s[t]['effect']['upper']-vals[i] for i,t in enumerate(('matched','mismatched'))]])
ax[1].bar(x,vals,.5,color=['#285880','#ca753b'],yerr=err,capsize=5)
for i,val in enumerate(vals):ax[1].text(i,val+err[1,i]+.12,f'{val:.2f}',ha='center',fontsize=11)
ax[1].set_xticks(x,labels);ax[1].set_ylabel('Binding-score drop after head removal (logits)');ax[1].set_ylim(0,5.8);ax[1].set_title('B. Effect of the fixed copying head',loc='left',pad=30)
fig.suptitle('Binding use falls when source and query use different token splits',fontsize=14,y=.99)
fig.text(.02,.015,'One Pythia-14M checkpoint; 2,048 prompts in 32 string groups. Error bars show 95% group-bootstrap intervals.\nThe different-split condition fails the fixed capability rule; its positive score effect is not successful retrieval.',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.12,1,.93]);(root/'figures').mkdir(exist_ok=True)
for suffix in ('png','svg'):fig.savefig(root/'figures'/('segmentation-transfer.'+suffix),dpi=180,bbox_inches='tight')
