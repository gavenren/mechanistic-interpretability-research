#!/usr/bin/env python3
"""Build descriptive tables and a figure from the fixed study outputs. No model calls."""
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_csv(name, rows):
    path = ROOT / 'tables' / name
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    (ROOT / 'tables').mkdir(exist_ok=True)
    (ROOT / 'figures').mkdir(exist_ok=True)
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    with gzip.open(ROOT / 'results/group-scores.csv.gz', 'rt') as stream:
        rows = list(csv.DictReader(stream))
    score_rows, ability_rows, named_rows, effects = [], [], [], []
    for model in summary['seeds']:
        seed = model['seed']
        score_rows.append({key: model[key] for key in [
            'seed', 'raw_selected_selectivity', 'random_selectivity', 'primary_difference',
            'ability_passes', 'per_model_interpretation_conditions_pass', 'positive_raw_names',
            'all_graph_cross_entropy']})
        for category, accuracy in model['ability'].items():
            ability_rows.append({'seed': seed, 'category': category, 'accuracy': accuracy,
                                 'gate': .99, 'passes': accuracy >= .99})
        for name in model['named_obstructions']:
            named_rows.append({'seed': seed, 'name': name['name'],
                               **{phase + '_' + field: name[phase][field]
                                  for phase in ['selection', 'test']
                                  for field in ['selected', 'random_mean', 'selected_minus_random', 'full_layer']}})
        for family in ['selected', 'random', 'full']:
            part = [row for row in rows if int(row['seed']) == seed and row['phase'] == 'test'
                    and (row['mask'].startswith(family + '_') if family != 'full' else row['mask'] == 'full')]
            assert len(part) == (1152 if family == 'random' else 144)
            for category in ['target', 'positive', 'hall3']:
                values = {field: float(np.mean([float(row[category + '_' + field]) for row in part]))
                          for field in ['logit_change', 'intact_logit', 'ablated_logit', 'probability_change',
                                        'intact_acceptance', 'ablated_acceptance', 'removed_norm']}
                effects.append({'seed': seed, 'family': family, 'category': category, **values})
    write_csv('model-scores.csv', score_rows)
    write_csv('ability.csv', ability_rows)
    write_csv('named-scores.csv', named_rows)
    write_csv('category-effects.csv', effects)

    # Equal weights for seeds, names, degree classes, and random masks match the plan.
    pooled = []
    for family in ['selected', 'random', 'full']:
        for category in ['target', 'positive', 'hall3']:
            part = [row for row in effects if row['family'] == family and row['category'] == category]
            pooled.append({'family': family, 'category': category,
                           **{key: float(np.mean([row[key] for row in part]))
                              for key in effects[0] if key not in ['seed', 'family', 'category']}})
    write_csv('pooled-category-effects.csv', pooled)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'hall-obstruction-fixed-study'})
    colors = {'selected': '#087f8c', 'random': '#a85d30', 'full': '#756189', 'intact': '#77838d'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7), layout='constrained')
    ax = axes[0, 0]
    categories = ['all_graphs', 'no_isolated_negatives', 'test_target', 'test_positive', 'test_hall3']
    ability = np.array([[100 * model['ability'][key] for key in categories] for model in summary['seeds']])
    ax.imshow(ability, vmin=80, vmax=100, cmap='YlGn', aspect='auto')
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f'{ability[i,j]:.2f}', ha='center', va='center', color='#16251a', fontsize=10)
    ax.set_xticks(range(5), ['All\ngraphs', 'No isolated\nnegative', 'Target\nnegative', 'Positive', 'Hall-3\nnegative'])
    ax.set_yticks(range(4), [str(model['seed']) for model in summary['seeds']])
    ax.set_ylabel('Model seed')
    ax.set_title('A. Intact accuracy (%): no model passes all five 99% gates', loc='left', fontsize=11, fontweight='bold', pad=12)

    ax = axes[0, 1]
    x = np.arange(4)
    for i, family in enumerate(['selected', 'random', 'full']):
        data = []
        for model in summary['seeds']:
            if family == 'selected':
                data.append(model['raw_selected_selectivity'])
            elif family == 'random':
                data.append(model['random_selectivity'])
            else:
                data.append(np.mean([name['test']['full_layer'] for name in model['named_obstructions']]))
        ax.bar(x + (i - 1) * .24, data, width=.23, color=colors[family], label=family.title())
    ax.axhline(0, color='#222222', linewidth=.8)
    ax.set_xticks(x, [str(model['seed']) for model in summary['seeds']])
    ax.set_xlabel('Model seed')
    ax.set_ylabel('Selectivity (logit units)')
    ax.legend(frameon=False, ncols=3, fontsize=9)
    ax.set_title('B. Selected units beat random units on the fixed score', loc='left', fontsize=11, fontweight='bold', pad=12)

    def value(family, category, field):
        return next(row[field] for row in pooled if row['family'] == family and row['category'] == category)

    ax = axes[1, 0]
    cats = ['target', 'positive', 'hall3']
    labels = ['Target\n(correct: reject)', 'Positive\n(correct: accept)', 'Hall-3 control\n(correct: reject)']
    x = np.arange(3)
    for i, family in enumerate(['intact', 'selected', 'random', 'full']):
        data = [100 * value('selected' if family == 'intact' else family, c,
                            'intact_acceptance' if family == 'intact' else 'ablated_acceptance') for c in cats]
        ax.bar(x + (i - 1.5) * .19, data, width=.18, color=colors[family], label=family.title())
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Graph acceptance (%)')
    ax.legend(frameon=False, ncols=2, fontsize=9, loc='upper center')
    ax.set_title('C. Selected edits also make Hall-3 controls fail', loc='left', fontsize=11, fontweight='bold', pad=12)

    ax = axes[1, 1]
    for i, family in enumerate(['selected', 'random', 'full']):
        ax.bar(x + (i - 1) * .24, [value(family, c, 'logit_change') for c in cats], width=.23,
               color=colors[family], label=family.title())
    ax.axhline(0, color='#222222', linewidth=.8)
    ax.set_xticks(x, ['Target', 'Positive', 'Hall-3 control'])
    ax.set_ylabel('Mean logit change toward acceptance')
    ax.legend(frameon=False, ncols=3, fontsize=9)
    ax.set_title('D. The target effect is larger, but is not exclusive', loc='left', fontsize=11, fontweight='bold', pad=12)
    fig.suptitle('Hall-obstruction test: large effects, failed ability gates, changed control decisions', fontsize=14, fontweight='bold')
    fig.savefig(ROOT / 'figures/hall-obstruction.png', dpi=180)
    fig.savefig(ROOT / 'figures/hall-obstruction.svg', metadata={'Date': None})
    plt.close(fig)
    print(json.dumps({'model_rows': len(score_rows), 'ability_rows': len(ability_rows),
                      'named_rows': len(named_rows), 'category_rows': len(effects), 'pooled_rows': len(pooled)}))


if __name__ == '__main__':
    main()
