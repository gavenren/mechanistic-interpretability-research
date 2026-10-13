#!/usr/bin/env python3
"""Plot the fixed primary comparisons and all 24 descriptive head results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
result = json.loads((ROOT/'results/bootstrap.json').read_text())
with (ROOT/'results/head-summary.csv').open() as f:
    rows = [r for r in csv.DictReader(f) if r['subset']=='all']
metrics = ['route_drop','value_drop','joint_rescue_from_route']
labels = ['Route change:\nloss of answer score','Value change:\nloss of answer score',
          'Joint change:\nrecovery from route change']
fig,(left,right) = plt.subplots(1,2,figsize=(12,9),gridspec_kw={'width_ratios':[1.2,1]})
for i,(metric,label) in enumerate(zip(metrics,labels)):
    r = result['intervals'][metric]
    left.errorbar(r['mean'],i,xerr=[[r['mean']-r['lower']],[r['upper']-r['mean']]],
                  fmt='o',markersize=7,capsize=5,color='#16657c')
left.axvline(0,color='gray',linewidth=1)
left.set_yticks(range(3),labels)
left.set_ylim(2.6,-.6)
left.set_xlabel('Mean change in logit units; 95% group bootstrap interval')
head=result['head']
left.set_title(f"Selected on calibration data: layer {head['layer']}, head {head['head']}")
left.grid(axis='x',alpha=.2)
data=np.array([[float(r[m]) for m in metrics] for r in rows])
bound=max(float(np.abs(data).max()),1e-8)
im=right.imshow(data,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
right.set_yticks(range(24),[f"L{r['layer']} H{r['head']}" for r in rows],fontsize=8)
right.set_xticks(range(3),['Route loss','Value loss','Joint recovery'],rotation=20,ha='right')
right.set_title('All heads: descriptive test means')
fig.colorbar(im,ax=right,label='Logit units',fraction=.04,pad=.04)
fig.suptitle('Which attention changes alter use of the supplied key–value binding?',fontsize=15)
fig.text(.5,.018,'Pythia-14M, one fixed checkpoint; 16 held-out token groups. Layer and head numbers start at zero.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.045,1,.95))
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/copy-routing.png',dpi=180)
fig.savefig(ROOT/'figures/copy-routing.svg')
