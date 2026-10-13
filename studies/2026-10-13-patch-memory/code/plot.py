#!/usr/bin/env python3
"""Draw the two fixed comparison horizons from complete held-out tables."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / 'results/group-summary.csv').open()))
selection = json.loads((root / 'results/selection.json').read_text())
colors = ['#277DA1', '#43AA8B', '#F8961E', '#7B2CBF', '#F94144', '#777777']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
for ax, horizon, title in zip(axes, (0, 8), ('Immediately after the edit', 'After eight fixed tokens')):
    values = [float(next(r['mean_tv'] for r in rows if r['split'] == 'test'
                         and r['group'] == 'all' and int(r['layer']) == layer
                         and int(r['horizon']) == horizon)) for layer in range(6)]
    ax.bar(range(6), values, color=colors, width=.7)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(6))
    ax.set_xlabel('Edited block (index starts at 0)')
    ax.set_ylabel('Mean output change (total variation)')
    ax.set_ylim(0, max(max(values) * 1.3, 1e-8))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for layer, value in enumerate(values):
        ax.annotate(f'{value:.3g}', (layer, value), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8)
fig.suptitle('Immediate and later output changes in the held-out prompts', fontsize=14)
immediate, delayed = selection['immediate_selected_layer'], selection['delayed_selected_layer']
fig.supxlabel(f'Calibration choices: immediate rule → block {immediate}; delayed rule → block {delayed}. '
              'Block 5 is a known zero-future control.\n'
              'Panels use separate vertical scales. More output change does not mean better output.', fontsize=9)
(root / 'figures').mkdir(exist_ok=True)
fig.savefig(root / 'figures/layer-comparison.png', dpi=180)
plt.close(fig)
