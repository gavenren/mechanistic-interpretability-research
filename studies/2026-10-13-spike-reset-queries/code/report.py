"""Build descriptive tables and a figure from the fixed, saved study results.

No model calls, training, or changes to scientific results occur here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_table(path, values):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh output directory.'
    manifest = json.loads((args.results/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, value in manifest['files'].items():
        assert digest(args.results/name) == value
    models = json.loads((args.results/'model-summaries.json').read_text())
    summary = json.loads((args.results/'summary.json').read_text())
    cases, events, regrets, joints = [rows(args.results/f'{name}.csv')
                                     for name in ('cases', 'events', 'regrets', 'joints')]
    assert [len(x) for x in (cases, events, regrets, joints)] == [480, 30720, 1920, 2880]
    args.output.mkdir(parents=True)
    seed_table, coverage, signs, regret_table, joint_table = [], [], [], [], []
    for model in models:
        seed = model['seed']
        own = [row for row in cases if int(row['seed']) == seed]
        with np.load(args.results/f'panel-{seed}.npz') as panel:
            ordered_logits = np.sort(panel['logits'][0], axis=-1)
            baseline_margin = (ordered_logits[:, -1]-ordered_logits[:, -2]).mean()
        seed_table.append(dict(seed=seed, test_accuracy=model['test_accuracy'],
            panel_accuracy=model['panel_accuracy'], primary=model['primary'],
            gain_message=model['gain_message'], gain_node=model['gain_node'],
            correct_baseline_n=model['correct_baseline_n'],
            correct_baseline_primary=model['correct_baseline_primary'],
            baseline_margin_mean=float(baseline_margin)))
        coverage.append(dict(seed=seed, candidate_count=120*32,
            active_count=sum(int(row['active_candidates']) for row in own),
            same_selection_count=sum(row['selection_same']=='True' for row in own),
            positive_primary_count=sum(float(row['primary'])>0 for row in own),
            negative_primary_count=sum(float(row['primary'])<0 for row in own),
            zero_primary_count=sum(float(row['primary'])==0 for row in own),
            any_positive_message_count=sum(row['any_positive_message']=='True' for row in own),
            any_positive_node_count=sum(row['any_positive_node']=='True' for row in own),
            zero_message_effects=sum(int(row['zero_effects_message']) for row in own),
            zero_node_effects=sum(int(row['zero_effects_node']) for row in own),
            exact_query_mean_absolute_difference=model['exact_query_difference_mean_absolute']))
        for query in ('message', 'node'):
            for graph in ('message', 'node'):
                rr = [r for r in regrets if int(r['seed'])==seed and r['query']==query and r['graph']==graph]
                regret_table.append(dict(seed=seed, query=query, graph=graph,
                    mean_regret=float(np.mean([float(r['regret']) for r in rr])),
                    mean_selected_effect=float(np.mean([float(r['exact_selected_effect']) for r in rr])),
                    mean_best_candidate_effect=float(np.mean([float(r['oracle_effect']) for r in rr])),
                    selected_negative_count=sum(float(r['exact_selected_effect'])<0 for r in rr),
                    selected_zero_count=sum(float(r['exact_selected_effect'])==0 for r in rr)))
                ee = [r for r in events if int(r['seed'])==seed and r['query']==query
                      and int(r['active']) and float(r['exact_effect']) and float(r[f'predicted_{graph}'])]
                wrong = sum(np.sign(float(r['exact_effect']))!=np.sign(float(r[f'predicted_{graph}'])) for r in ee)
                signs.append(dict(seed=seed, query=query, graph=graph, eligible=len(ee), wrong=int(wrong),
                    fraction=wrong/len(ee) if ee else None))
            for selector in ('message', 'node', 'random'):
                jj = [r for r in joints if int(r['seed'])==seed and r['query']==query and r['selector']==selector]
                joint_table.append(dict(seed=seed, query=query, selector=selector,
                    mean_effect=float(np.mean([float(r['effect']) for r in jj]))))
    tables = {'seeds': seed_table, 'coverage': coverage, 'regrets': regret_table,
              'wrong-sign': signs, 'joint-effects': joint_table}
    for name, table in tables.items():
        write_table(args.output/f'{name}.csv', table)
    rt = np.mean([m['regret_table'] for m in models], axis=0)
    selected = np.mean([m['selected_effect_table'] for m in models], axis=0)
    pooled_sign = []
    for query in ('message', 'node'):
        for graph in ('message', 'node'):
            rr = [r for r in signs if r['query']==query and r['graph']==graph]
            n, wrong = sum(r['eligible'] for r in rr), sum(r['wrong'] for r in rr)
            pooled_sign.append(dict(query=query, graph=graph, eligible=n, wrong=wrong, fraction=wrong/n))
    report = dict(note='Descriptive report from fixed results; no new model calls.',
        result_manifest_sha256=digest(args.results/'manifest.json'),
        report_code_sha256=digest(__file__), table_row_count=sum(map(len, tables.values())),
        means={key:float(np.mean([r[key] for r in seed_table])) for key in seed_table[0] if key!='seed'},
        regret_table=rt.tolist(), selected_effect_table=selected.tolist(),
        mean_diagonal_regret=float(np.trace(rt)/2), mean_off_diagonal_regret=float((rt[0,1]+rt[1,0])/2),
        joint_means=np.mean([m['joint_means'] for m in models], axis=0).tolist(),
        coverage_totals={key:sum(r[key] for r in coverage) for key in coverage[0]
                         if key not in ('seed','exact_query_mean_absolute_difference')},
        pooled_wrong_sign=pooled_sign,
        exact_query_mean_absolute_difference=float(np.mean([r['exact_query_mean_absolute_difference'] for r in coverage])))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout='constrained')
    seed_labels = [str(m['seed']) for m in models]
    x = np.arange(4)
    axes[0,0].bar(x, [100*m['test_accuracy'] for m in models], color='#326c91')
    axes[0,0].axhline(80, ls='--', color='#555', label='Fixed 80% gate')
    axes[0,0].set(xticks=x, xticklabels=seed_labels, ylim=(0,100), ylabel='Test accuracy (%)',
                  title='All four models passed the ability check', xlabel='Model seed')
    axes[0,0].legend(frameon=False, loc='lower left')
    axes[0,1].plot(x, [m['gain_message'] for m in models], 'o-', color='#326c91', label='Message query')
    axes[0,1].plot(x, [m['gain_node'] for m in models], 's-', color='#c26835', label='Node query')
    axes[0,1].plot(x, [m['primary'] for m in models], 'D', color='#222', label='Primary: mean of both')
    axes[0,1].axhline(0, color='#999', lw=1)
    axes[0,1].set(xticks=x, xticklabels=seed_labels, ylabel='Matching gain (margin units)',
                  title='One query loses in seed 317', xlabel='Model seed')
    axes[0,1].legend(frameon=False, fontsize=9)
    axes[1,0].bar(np.arange(2)-.17, rt[:,0], width=.34, color='#326c91', label='Message gradient')
    axes[1,0].bar(np.arange(2)+.17, rt[:,1], width=.34, color='#c26835', label='Node gradient')
    axes[1,0].set(xticks=np.arange(2), xticklabels=['Message deletion', 'Node deletion'],
                  ylabel='Single-event regret (margin units)', title='Matching still leaves selection error')
    axes[1,0].legend(frameon=False, fontsize=9)
    jm=np.array(report['joint_means']).reshape(2,3)
    for j,(name,color) in enumerate(zip(['Message ranking','Node ranking','Random four'],['#326c91','#c26835','#909090'])):
        axes[1,1].bar(np.arange(2)+(j-1)*.23, jm[:,j], width=.23, label=name, color=color)
    axes[1,1].set(xticks=np.arange(2), xticklabels=['Message deletion','Node deletion'],
                  ylabel='Direct joint margin drop', title='Secondary: delete four events together')
    axes[1,1].legend(frameon=False, fontsize=9)
    fig.suptitle('Matching spike explanations to two meanings of deletion', fontsize=15)
    fig.savefig(args.output/'spike-queries.png', dpi=180)
    fig.savefig(args.output/'spike-queries.pdf')
    plt.close(fig)
    (args.output/'manifest.json').write_text(json.dumps({p.name:digest(p) for p in sorted(args.output.iterdir())}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__=='__main__':
    main()
