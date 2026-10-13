#!/usr/bin/env python3
"""Draw the fixed endpoint from saved results; no model calls."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / 'results/summary.json').read_text())
with (ROOT / 'results/set-means.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
seeds = [row['seed'] for row in summary['per_seed']]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1.05, 1.0]})
fig.suptitle('Very small order effects after 2,000 balanced updates', fontsize=15, fontweight='bold')
colors = {'reset': '#2166ac', 'keep': '#c26922'}
for condition, offset in [('reset', -.12), ('keep', .12)]:
    values = [100 * row[condition + '_BA_minus_AB'] for row in summary['per_seed']]
    axes[0].scatter(values, np.arange(8) + offset, color=colors[condition], s=35,
                    label='Momentum reset' if condition == 'reset' else 'Momentum kept')
axes[0].axvline(0, color='#333333', linewidth=.8)
axes[0].axvline(100 * summary['primary']['mean'], color=colors['reset'], linestyle='--', linewidth=1,
               label='Reset mean')
axes[0].set(yticks=np.arange(8), yticklabels=[str(s) for s in seeds], ylabel='Initialization seed',
            xlabel='B→A minus A→B effect (percentage points)', title='Each paired seed: selected eight paths')
axes[0].invert_yaxis()
axes[0].grid(axis='x', color='#dddddd', linewidth=.5)
axes[0].legend(frameon=False, fontsize=9, loc='lower left')
for k, (kind, label, color) in enumerate([('selected', 'Selected eight paths', '#2166ac'),
                                         ('full', 'Whole shortcut input', '#5a8b55')]):
    for i, schedule in enumerate(('AB', 'BA', 'ALT')):
        values = [100 * float(row['mean_oriented_probability']) for row in rows
                  if row['tail'] == '2000' and row['condition'] == 'reset' and row['split'] == 'test'
                  and row['schedule'] == schedule and row['kind'] == kind]
        x = i + (-.13 if k == 0 else .13)
        axes[1].scatter(np.repeat(x, len(values)), values, s=25, alpha=.55, color=color,
                        label=label if i == 0 else None)
        axes[1].plot([x-.08, x+.08], [np.mean(values)] * 2, color=color, linewidth=3)
axes[1].axhline(0, color='#333333', linewidth=.8)
axes[1].set(xticks=range(3), xticklabels=['A then B', 'B then A', 'Alternating'],
            ylabel='Oriented probability effect (percentage points)', title='Reset condition: paths and whole input')
axes[1].grid(axis='y', color='#dddddd', linewidth=.5)
axes[1].legend(frameon=False, fontsize=9, loc='upper right')
for ax in axes:
    ax.spines[['right', 'top']].set_visible(False)
fig.text(.5, .025, 'Dots show all eight seeds. Bars show means. All 48 endpoint models and all recorded endpoint patches keep the correct class.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, .05, 1, .93])
(ROOT / 'figures').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figures/order-effects.png', dpi=180)
fig.savefig(ROOT / 'figures/order-effects.pdf')
