#!/usr/bin/env python3
"""Draw the fixed three-way transfer comparison from saved result tables."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root/'results/float32/summary.csv').open() as f:
    rows = list(csv.DictReader(f))
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout='constrained')
for ax, split, title in zip(axes,
                           ('new_words', 'new_frames', 'new_words_and_frames'),
                           ('New words', 'New prompt frames', 'New words and frames')):
    selected = sorted((r for r in rows if r['split'] == split and r['group_type'] == 'layer'),
                      key=lambda r: int(r['group_id']))
    x = np.arange(5)
    ax.bar(x-.18, [float(r['uncorrected_mae']) for r in selected], .36,
           color='#557a95', label='Sum of separate effects')
    ax.bar(x+.18, [float(r['corrected_mae']) for r in selected], .36,
           color='#c27935', label='Sum plus calibration median')
    ax.set_xticks(x, [str(r['group_id']) for r in selected])
    ax.set_xlabel('Layer (zero-based)')
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Mean absolute error (logit units)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
fig.suptitle('Does a measured interaction correction transfer?', fontsize=14)
(root/'figures').mkdir(exist_ok=True)
fig.savefig(root/'figures/transfer-error.png', dpi=180)
plt.close(fig)
