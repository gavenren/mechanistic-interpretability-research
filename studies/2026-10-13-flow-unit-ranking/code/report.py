#!/usr/bin/env python3
"""Build fixed descriptive tables and a figure from saved results; no model calls."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def write_csv(name, rows):
    path = ROOT / 'tables' / name
    path.parent.mkdir(exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    receipt = json.loads((RESULTS / 'manifest.json').read_text())
    assert receipt['status'] == 'complete'
    models = json.loads((RESULTS / 'model-summary.json').read_text())
    panel = json.loads((RESULTS / 'summary.json').read_text())
    selected, units, quadrant_rows, unit_summaries = [], [], [], []
    test_scores = []
    for m in models:
        seed, fit = m['seed'], m['fit']
        row = dict(seed=seed, fit_gate=fit['gate'], test_excess_nll=fit['test_excess_nll'],
                   test_generator_quadrant_tv=fit['generated_quadrant_tv'], selector_agreement=m['selector_agreement'],
                   calibration_tau_b=m['calibration_tau_b'], test_tau_b=m['test_tau_b'], primary_difference=m['primary_difference'])
        for key in ('L', 'M'):
            score = m['selected_test'][key]
            for name in ('condition', 'block', 'unit', 'nll_damage', 'gaussian_damage', 'logjac_damage', 'quadrant_tv',
                         'mean_squared_displacement', 'crossing_fraction'):
                row[key + '_' + name] = score[name]
        selected.append(row)
        for phase in ('cal', 'test'):
            scores = json.loads((RESULTS / f'scores-{seed}-{phase}.json').read_text())
            unit_summary = dict(seed=seed, phase=phase)
            for key in ('nll_damage', 'quadrant_tv'):
                values = [score[key] for score in scores[1:]]
                unit_summary[key + '_mean'] = float(np.mean(values))
                unit_summary[key + '_minimum'] = min(values)
                unit_summary[key + '_maximum'] = max(values)
            unit_summaries.append(unit_summary)
            if phase == 'test':
                test_scores.append(scores)
                for q in range(4):
                    quadrant_rows.append(dict(seed=seed, quadrant=q, exact_target=fit['exact_target_quadrants'][q],
                                              generated_count=scores[0]['quadrant_counts'][q],
                                              generated_frequency=scores[0]['quadrant_frequencies'][q]))
            ranks = {key: {condition: i + 1 for i, condition in enumerate(m['rankings'][phase][key])}
                     for key in ('nll_damage', 'quadrant_tv')}
            for score in scores[1:]:
                unit = {key: score[key] for key in ('seed', 'phase', 'condition', 'block', 'unit', 'mean_nll', 'nll_damage',
                                                    'gaussian_damage', 'logjac_damage', 'quadrant_tv', 'mean_squared_displacement',
                                                    'squared_displacement_sum', 'crossing_fraction', 'crossing_count')}
                unit['nll_rank'] = ranks['nll_damage'][score['condition']]
                unit['quadrant_tv_rank'] = ranks['quadrant_tv'][score['condition']]
                for q in range(4):
                    unit[f'quadrant_{q}_frequency'] = score['quadrant_frequencies'][q]
                    unit[f'quadrant_{q}_change'] = score['quadrant_changes'][q]
                units.append(unit)
    write_csv('selector-comparison.csv', selected)
    write_csv('unit-rankings.csv', units)
    write_csv('generated-quadrants.csv', quadrant_rows)
    write_csv('all-unit-summary.csv', unit_summaries)
    moment_rows = []
    for model_label, overall, regions in ([(str(m['seed']), m['fit']['generated'], m['fit']['generated_by_quadrant']) for m in models]
                                           + [('test_data', panel['test_data'], panel['test_data_by_quadrant'])]):
        for label, stats in [('all', overall)] + [(str(i), r) for i, r in enumerate(regions)]:
            moment_rows.append(dict(model=model_label, quadrant=label, count=stats['count'],
                                   mean_x0=stats['mean'][0], mean_x1=stats['mean'][1],
                                   variance_x0=stats['covariance'][0][0], covariance=stats['covariance'][0][1], variance_x1=stats['covariance'][1][1]))
    write_csv('intact-moments.csv', moment_rows)
    assert len(moment_rows) == 25
    assert len(selected) == 4 and len(units) == 1024 and len(quadrant_rows) == 16
    colors = ['#185C8A', '#DF6B31', '#3C8C70', '#9365A6']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    x = np.arange(4)
    ax = axes[0, 0]
    ax.bar(x - .18, [m['selected_test']['L']['quadrant_tv'] for m in models], .35, color='#185C8A', label='Selected by NLL damage (L)')
    ax.bar(x + .18, [m['selected_test']['M']['quadrant_tv'] for m in models], .35, color='#DF6B31', label='Selected by quadrant change (M)')
    ax.set(xticks=x, xticklabels=[str(m['seed']) for m in models], xlabel='Model seed', ylabel='Held-out quadrant-frequency TV',
           title='A. Fixed selectors on separate test latents')
    ax.set_ylim(0, .082)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 1]
    for color, m, scores in zip(colors, models, test_scores):
        ax.scatter([s['nll_damage'] for s in scores[1:]], [s['quadrant_tv'] for s in scores[1:]], s=13, alpha=.45, color=color, label=str(m['seed']))
        for key, marker in [('L', 's'), ('M', '*')]:
            s = m['selected_test'][key]
            ax.scatter(s['nll_damage'], s['quadrant_tv'], s=85 if key == 'M' else 45, marker=marker, color=color, edgecolor='black', linewidth=.7)
    ax.set(xlabel='Held-out mean NLL increase (nats)', ylabel='Held-out quadrant-frequency TV', title='B. All 128 removals in each model')
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.text(.02, .98, 'Square: L; star: M', transform=ax.transAxes, va='top', fontsize=8)
    ax = axes[1, 0]
    ax.bar(x, [m['fit']['test_excess_nll'] for m in models], color=colors, width=.55)
    ax.axhline(.20, color='#8C2525', linestyle='--', linewidth=1.2, label='Fixed NLL fit limit')
    ax.set(xticks=x, xticklabels=[str(m['seed']) for m in models], xlabel='Model seed', ylabel='Intact test excess NLL (nats)', title='C. Intact density fit against the known law')
    ax.set_ylim(0, .31)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]
    for i, (color, m, scores) in enumerate(zip(colors, models, test_scores)):
        ax.bar(x + (i - 1.5) * .16, scores[0]['quadrant_frequencies'], .15, color=color, label=str(m['seed']))
    target = models[0]['fit']['exact_target_quadrants']
    for q in range(4):
        ax.plot([q - .36, q + .36], [target[q], target[q]], color='black', linewidth=1.4, label='Exact target' if q == 0 else None)
    ax.set(xticks=x, xticklabels=['(-, -)', '(-, +)', '(+, -)', '(+, +)'], xlabel='Signs of the two output coordinates', ylabel='Intact generated frequency',
           title='D. Coarse generated frequencies before removal')
    ax.set_ylim(0, .52)
    ax.legend(frameon=False, fontsize=8, ncols=3)
    gates = sum(m['fit']['gate'] for m in models)
    fig.suptitle('Likelihood damage and generated quadrant change', x=.05, ha='left', fontsize=17, weight='bold')
    fig.text(.05, .924, f'Four small Real NVP models; {gates}/4 pass both fixed fit checks. Quadrant TV is not full-density TV.', fontsize=10)
    fig.text(.05, .025, 'Fixed calibration selectors; all units retained. Four models, not 512 independent models.\nConstructed two-dimensional data. AI-authored study; no human peer review.', fontsize=9, color='#444444')
    fig.tight_layout(rect=(.03, .075, .99, .9), h_pad=2.2, w_pad=2.0)
    (ROOT / 'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'figures/flow-unit-ranking.png', dpi=180)
    fig.savefig(ROOT / 'figures/flow-unit-ranking.svg')
    print(json.dumps(dict(table_rows=len(selected) + len(units) + len(quadrant_rows) + len(moment_rows) + len(unit_summaries), primary_mean=panel['primary_mean'], fit_gates=gates)))


if __name__ == '__main__':
    main()
