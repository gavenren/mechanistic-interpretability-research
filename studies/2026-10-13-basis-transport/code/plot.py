#!/usr/bin/env python3
"""Plot the fixed descriptive comparisons. Does not train or select models."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parents[1]
with (STUDY/"results"/"per-seed-summary.csv").open() as f:
    rows = list(csv.DictReader(f))

kinds = ("identity", "permutation", "rotation", "condition_4", "condition_16")
labels = ("Original", "Signed\npermutation", "Rotation", "Scaled\n(condition 4)", "Scaled\n(condition 16)")
colors = plt.colormaps["tab10"](np.arange(8))
plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False,
                     "savefig.dpi":180, "figure.facecolor":"white", "axes.facecolor":"white"})
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=.10, right=.99, bottom=.23, top=.89)
for seed in range(8):
    selected = [next(r for r in rows if int(r["seed"])==seed and r["kind"]==kind) for kind in kinds]
    values = np.array([float(r["coordinate_tv_mean"]) for r in selected])
    lower = values-np.array([float(r["coordinate_tv_min"]) for r in selected])
    upper = np.array([float(r["coordinate_tv_max"]) for r in selected])-values
    offset = (seed-3.5)*.055
    ax.errorbar(np.arange(5)+offset, values, yerr=[lower,upper], fmt="o", ms=4.5,
                capsize=2, alpha=.85, color=colors[seed], label=f"Seed {seed}", linewidth=1)
ax.set_xticks(np.arange(5), labels)
ax.set_ylabel("Counterfactual error (mean total variation)")
ax.set_ylim(-.025, 1.025)
ax.grid(axis="y", alpha=.2)
ax.set_title("Coordinate masks after rotations that preserve model outputs", loc="left", fontsize=13)
ax.legend(ncol=4, loc="upper left", frameon=False, fontsize=9)
fig.text(.01, .025, "Each point is one trained seed. Rotated/scaled points average eight fixed maps; bars show their range.\nMasks use discovery pairs. These errors use 250 test pairs. The source factorization is designed.", fontsize=9)
output = STUDY/"figures"
output.mkdir(exist_ok=True)
fig.savefig(output/"coordinate-mask-errors.png", bbox_inches="tight")
fig.savefig(output/"coordinate-mask-errors.svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=.10, right=.99, bottom=.21, top=.89)
methods = (("transported_tv_mean", "Same transported intervention", "#167D8D", -.14),
           ("orthogonal_image_tv_mean", "Euclidean projector onto the same subspace", "#B6512E", .14))
for method, label, color, offset in methods:
    for i, kind in enumerate(("rotation", "condition_4", "condition_16")):
        values = [float(r[method]) for r in rows if r["kind"] == kind]
        ax.scatter(i+offset+np.linspace(-.04,.04,len(values)), values, color=color,
                   s=28, alpha=.85, label=label if i==0 else None)
        ax.plot([i+offset-.065,i+offset+.065], [np.mean(values)]*2, color=color, linewidth=2)
ax.set_xticks(range(3), ("Rotation", "Scaled (condition 4)", "Scaled (condition 16)"))
ax.set_ylabel("Counterfactual error (mean total variation)")
ax.set_ylim(-.025, 1.025)
ax.set_title("The same subspace can use a different replacement rule", loc="left", fontsize=13)
ax.grid(axis="y", alpha=.2)
ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.text(.01, .025, "Each point averages the eight fixed maps for one trained seed; short lines mark the eight-seed mean.\nThe transported operator can be oblique. Both operators have the same image subspace.", fontsize=9)
fig.savefig(output/"operator-comparison.png", bbox_inches="tight")
fig.savefig(output/"operator-comparison.svg", bbox_inches="tight")
plt.close(fig)
