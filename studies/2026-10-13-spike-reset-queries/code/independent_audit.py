#!/usr/bin/env python3
"""Saved-weight replay and reverse-adjoint audit; does not import the runner."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [311, 313, 317, 331]
QUERIES = ['message', 'node']
FIELDS = ['u1', 's1', 'm1', 'v1', 'u2', 's2', 'v2']
WEIGHTS = ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias', 'readout.weight', 'readout.bias']
TOL = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_hash(values):
    h = hashlib.sha256()
    for name in WEIGHTS:
        h.update(name.encode())
        h.update(np.ascontiguousarray(values[name]).tobytes())
    return h.hexdigest()


def initial_hash(seed):
    torch.manual_seed(seed)
    values = {}
    for name, inputs, outputs in [('layer1', 64, 32), ('layer2', 32, 32), ('readout', 32, 10)]:
        layer = torch.nn.Linear(inputs, outputs, dtype=torch.float64)
        values[name+'.weight'] = layer.weight.detach().numpy()
        values[name+'.bias'] = layer.bias.detach().numpy()
    return weight_hash(values)


def replay(x, weights, query=None, mask=None):
    """Explicit hard recurrence. There is no surrogate or autograd function here."""
    current = F.linear(x, weights['layer1.weight'], weights['layer1.bias'])
    state1 = torch.zeros((len(x), 32), dtype=torch.float64)
    state2 = torch.zeros_like(state1)
    rows = {k: [] for k in FIELDS}
    for t in range(8):
        potential1 = state1*.9 + current
        reset1 = (potential1 >= 1).double()
        outgoing = reset1
        if t == 3 and query is not None:
            outgoing = reset1.masked_fill(mask, 0.)
            if query == 'node':
                reset1 = outgoing
        state1 = potential1 - reset1
        potential2 = state2*.9 + F.linear(outgoing, weights['layer2.weight'], weights['layer2.bias'])
        reset2 = (potential2 >= 1).double()
        state2 = potential2-reset2
        for key, value in zip(FIELDS, [potential1, reset1, outgoing, state1, potential2, reset2, state2]):
            rows[key].append(value)
    traces = {k: torch.stack(v, 1) for k, v in rows.items()}
    logits = F.linear(traces['s2'].mean(1), weights['readout.weight'], weights['readout.bias'])
    return logits.numpy(), {k: v.numpy() for k, v in traces.items()}


def adjoint(weights, trace, winner, runner_up, query):
    """Reverse recurrence; separate from the runner's forward tangent directions."""
    w2 = weights['layer2.weight']
    read = (weights['readout.weight'][winner]-weights['readout.weight'][runner_up])/8
    post1 = np.zeros((len(winner), 32))
    post2 = np.zeros_like(post1)
    for t in range(7, 2, -1):
        slope2 = 1 / np.square(1+10*np.abs(trace['u2'][:, t]-1))
        before2 = post2 + (read-post2)*slope2
        outgoing = before2 @ w2
        post2 = .9*before2
        if t == 3:
            return outgoing-post1 if query == 'node' else outgoing
        slope1 = 1 / np.square(1+10*np.abs(trace['u1'][:, t]-1))
        before1 = post1 + (outgoing-post1)*slope1
        post1 = .9*before1
    raise AssertionError('Missing intervention step')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    start = time.monotonic()
    receipt = {'status': 'running', 'repeated_training': False, 'tolerance': TOL,
               'audit_sha256': sha(__file__), 'gradient_method': 'manual reverse adjoint, no autograd',
               'forward_method': 'explicit hard recurrence, no runner imports'}
    maxima = {}
    checks = calls = examples = 0
    context = {}

    def check(a, b, name, tolerance=TOL):
        nonlocal checks
        aa, bb = np.asarray(a), np.asarray(b)
        assert aa.shape == bb.shape, (name, aa.shape, bb.shape)
        assert np.isfinite(aa).all() and np.isfinite(bb).all(), name
        error = float(np.abs(aa-bb).max()) if aa.size else 0.
        maxima[name] = max(maxima.get(name, 0.), error)
        checks += 1
        assert error <= tolerance, (name, error, tolerance)

    def forward(x, weights, **kwargs):
        nonlocal calls, examples
        if time.monotonic()-start > 1200:
            raise TimeoutError('Audit 20-minute cap')
        calls += 1
        examples += len(x)
        with torch.no_grad():
            return replay(x, weights, **kwargs)

    try:
        meta = json.loads((args.results/'metadata.json').read_text())
        assert meta['mode'] == 'fixed-study'
        manifest = json.loads((args.results/'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        for name, digest in manifest['files'].items():
            assert sha(args.results/name) == digest, name
        receipt['manifest_sha256'] = sha(args.results/'manifest.json')
        repo = ROOT.parents[1]
        for name in ['analysis-plan.md', 'code/run.py', 'inputs.json', 'sources.md', 'requirements.txt',
                     'data/optdigits.tra', 'data/optdigits.tes']:
            rel = (ROOT/name).relative_to(repo).as_posix()
            assert subprocess.check_output(['git', 'show', meta['lock_commit']+':'+rel], cwd=repo) == (ROOT/name).read_bytes()
        ledger = json.loads((ROOT/'inputs.json').read_text())
        for name, digest in ledger['data_hashes'].items():
            assert sha(ROOT/'data'/name) == digest
        raw = np.loadtxt(ROOT/'data/optdigits.tes', delimiter=',', dtype=np.int64)
        assert raw.shape == (1797, 65)
        panel_ids = np.array(sorted(np.concatenate([np.flatnonzero(raw[:, -1] == digit)[:12] for digit in range(10)])))
        assert ledger['panel'] == [{'row': int(i), 'label': int(raw[i, -1])} for i in panel_ids]
        x = torch.from_numpy(raw[:, :64]/16.)
        y = raw[:, -1]
        tables = {name: list(csv.DictReader((args.results/(name+'.csv')).open()))
                  for name in ['training', 'controls', 'cases', 'events', 'regrets', 'joints']}
        assert {k: len(v) for k, v in tables.items()} == dict(training=4800, controls=1688, cases=480,
                                                           events=30720, regrets=1920, joints=2880)
        for row in tables['controls']:
            assert row['passed'] == 'True'
            assert 0 <= float(row['error']) <= float(row['tolerance']) <= TOL
        indexes = {}
        for name, fields in [('cases', ['seed', 'row']), ('events', ['seed', 'row', 'query', 'unit']),
                             ('regrets', ['seed', 'row', 'query', 'graph']), ('joints', ['seed', 'row', 'query', 'selector'])]:
            indexes[name] = {tuple(row[f] for f in fields): row for row in tables[name]}
            assert len(indexes[name]) == len(tables[name])
        model_summaries = json.loads((args.results/'model-summaries.json').read_text())
        assert [r['seed'] for r in model_summaries] == SEEDS
        seed_scores = []
        for si, (seed, summary) in enumerate(zip(SEEDS, model_summaries)):
            context = {'seed': seed}
            weights = dict(np.load(args.results/f'model-{seed}.npz'))
            assert list(weights) == WEIGHTS
            assert sum(v.size for v in weights.values()) == 3466
            assert all(np.isfinite(v).all() for v in weights.values())
            assert weight_hash(weights) == summary['final_parameter_sha256']
            assert initial_hash(seed) == summary['initial_parameter_sha256']
            wt = {k: torch.from_numpy(v) for k, v in weights.items()}
            rng = np.random.default_rng(10000+seed)
            orders = np.stack([rng.permutation(3823) for _ in range(40)])
            assert np.array_equal(orders, np.load(args.results/f'schedule-{seed}.npz')['orders'])
            training = [r for r in tables['training'] if int(r['seed']) == seed]
            for i, row in enumerate(training):
                assert [int(row[k]) for k in ['epoch', 'batch', 'n', 'global_update']] == [i//30, i % 30, 111 if i % 30 == 29 else 128, si*1200+i+1]
                assert np.isfinite(float(row['loss'])) and float(row['loss']) >= 0
            full = np.concatenate([forward(x[i:i+128], wt)[0] for i in range(0, len(x), 128)])
            full_saved = np.load(args.results/f'test-{seed}.npz')
            check(full, full_saved['logits'], 'full_test_logits')
            assert np.array_equal(y, full_saved['labels'])
            accuracy = float((full.argmax(1) == y).mean())
            check(accuracy, summary['test_accuracy'], 'ability')
            assert summary['ability_gate'] == (accuracy >= .8)
            saved = np.load(args.results/f'panel-{seed}.npz')
            assert np.array_equal(saved['row_indices'], panel_ids) and np.array_equal(saved['labels'], y[panel_ids])
            baseline, traces = forward(x[panel_ids], wt)
            order = np.argsort(-baseline, axis=1, kind='stable')
            winner, runner_up = order[:, 0], order[:, 1]
            axis = np.arange(120)
            margin = baseline[axis, winner]-baseline[axis, runner_up]
            gradients = np.stack([adjoint(weights, traces, winner, runner_up, q) for q in QUERIES])
            check(gradients, saved['gradients'], 'reverse_adjoint_gradients')
            spike = traces['s1'][:, 3]
            predicted = gradients*spike[None]
            check(predicted, saved['predictions'], 'predicted_effects')
            # Rebuild selectors from saved gradients too, so tiny replay error cannot hide a tie issue.
            selected = saved['predictions'].argmax(-1)
            top4 = np.argsort(-saved['predictions'], axis=-1, kind='stable')[:, :, :4]
            assert np.array_equal(selected, saved['selected']) and np.array_equal(top4, saved['top4'])
            assert np.array_equal(predicted.argmax(-1), selected)
            assert np.array_equal(np.argsort(-predicted, axis=-1, kind='stable')[:, :, :4], top4)
            rng = np.random.default_rng(20000+seed)
            random = np.stack([rng.choice(32, 4, replace=False) for _ in range(120)])
            assert np.array_equal(random, saved['random_top4'])
            names = ['baseline', 'leaf_message', 'leaf_node']
            names += [f'single_{q}_{u}' for q in QUERIES for u in range(32)]
            names += [f'joint_{q}_{g}' for q in QUERIES for g in [*QUERIES, 'random']]
            names += ['self_message', 'self_node']
            assert saved['condition_names'].tolist() == names
            exact = np.zeros((2, 120, 32))
            joint = []
            for ci, condition in enumerate(names):
                context = {'seed': seed, 'condition': condition}
                query = None
                mask = None
                if condition.startswith('single_'):
                    _, query, unit = condition.split('_')
                    mask = torch.zeros((120, 32), dtype=torch.bool)
                    mask[:, int(unit)] = True
                elif condition.startswith('joint_'):
                    _, query, selector = condition.split('_')
                    chosen = random if selector == 'random' else top4[QUERIES.index(selector)]
                    mask = torch.zeros((120, 32), dtype=torch.bool)
                    mask[torch.arange(120)[:, None], torch.from_numpy(chosen)] = True
                z, tr = (baseline, traces) if ci == 0 else forward(x[panel_ids], wt, query=query, mask=mask)
                check(z, saved['logits'][ci], 'panel_logits')
                for field in ['s1', 'm1', 's2']:
                    check(tr[field], saved[field][ci], 'binary_'+field, 0.)
                if ci < 3:
                    for field in ['u1', 'u2', 'v1', 'v2']:
                        check(tr[field], saved[condition+'_'+field], 'voltage_'+field)
                drop = margin-(z[axis, winner]-z[axis, runner_up])
                if condition.startswith('single_'):
                    exact[QUERIES.index(query), :, int(unit)] = drop
                elif condition.startswith('joint_'):
                    joint.append(drop)
                    for j, rid in enumerate(panel_ids):
                        row = indexes['joints'][(str(seed), str(rid), query, selector)]
                        assert [int(v) for v in row['units'].split()] == chosen[j].tolist()
                        check(drop[j], float(row['effect']), 'joint_rows')
            check(exact, saved['exact_effects'], 'exact_effects')
            regrets = np.empty((120, 2, 2))
            effects = np.empty_like(regrets)
            for qi, q in enumerate(QUERIES):
                for gi, g in enumerate(QUERIES):
                    effects[:, qi, gi] = exact[qi, axis, selected[gi]]
                    regrets[:, qi, gi] = exact[qi].max(-1)-effects[:, qi, gi]
                    for j, rid in enumerate(panel_ids):
                        row = indexes['regrets'][(str(seed), str(rid), q, g)]
                        assert int(row['selected_unit']) == selected[gi, j]
                        check([effects[j, qi, gi], exact[qi, j].max(), regrets[j, qi, gi]],
                              [float(row[k]) for k in ['exact_selected_effect', 'oracle_effect', 'regret']], 'regret_rows')
                for j, rid in enumerate(panel_ids):
                    for unit in range(32):
                        row = indexes['events'][(str(seed), str(rid), q, str(unit))]
                        assert int(row['active']) == spike[j, unit]
                        check([exact[qi, j, unit], predicted[0, j, unit], predicted[1, j, unit]],
                              [float(row[k]) for k in ['exact_effect', 'predicted_message', 'predicted_node']], 'event_rows')
            gains = np.stack([effects[:, 0, 0]-effects[:, 0, 1], effects[:, 1, 1]-effects[:, 1, 0]])
            primary = gains.mean(0)
            check(primary, .5*(regrets[:, 1, 0]+regrets[:, 0, 1]-regrets[:, 1, 1]-regrets[:, 0, 0]), 'primary_identity')
            check(primary, saved['primary'], 'primary_array')
            check(gains[0], saved['gain_message'], 'message_gain')
            check(gains[1], saved['gain_node'], 'node_gain')
            check(regrets, saved['regrets'], 'regret_array')
            correct = winner == y[panel_ids]
            for j, rid in enumerate(panel_ids):
                row = indexes['cases'][(str(seed), str(rid))]
                expected_ints = dict(label=int(y[rid]), winner=int(winner[j]), runner_up=int(runner_up[j]),
                                     active_candidates=int(spike[j].sum()), zero_effects_message=int((exact[0, j] == 0).sum()),
                                     zero_effects_node=int((exact[1, j] == 0).sum()))
                assert all(int(row[k]) == v for k, v in expected_ints.items())
                expected_bools = dict(correct=bool(correct[j]), any_positive_message=bool((exact[0, j] > 0).any()),
                                      any_positive_node=bool((exact[1, j] > 0).any()), selection_same=bool(selected[0, j] == selected[1, j]))
                assert all(row[k] == str(v) for k, v in expected_bools.items())
                check([gains[0, j], gains[1, j], primary[j]], [float(row[k]) for k in ['gain_message', 'gain_node', 'primary']], 'case_rows')
            rebuilt = dict(panel_n=120, panel_accuracy=correct.mean(), primary=primary.mean(), gain_message=gains[0].mean(),
                           gain_node=gains[1].mean(), regret_table=regrets.mean(0), selected_effect_table=effects.mean(0),
                           active_candidates_mean=spike.sum(-1).mean(), any_positive_fraction=[(exact[q] > 0).any(-1).mean() for q in range(2)],
                           zero_effect_counts=[(exact[q] == 0).sum() for q in range(2)], selection_agreement=(selected[0] == selected[1]).mean(),
                           correct_baseline_n=correct.sum(), joint_means=np.array(joint).mean(-1),
                           exact_query_difference_mean=(exact[1]-exact[0]).mean(),
                           exact_query_difference_mean_absolute=np.abs(exact[1]-exact[0]).mean())
            for key, value in rebuilt.items():
                check(value, summary[key], 'summary_'+key)
            if correct.any():
                check(primary[correct].mean(), summary['correct_baseline_primary'], 'correct_subset')
            else:
                assert summary['correct_baseline_primary'] is None
            for record in summary['wrong_sign']:
                qi, gi = QUERIES.index(record['query']), QUERIES.index(record['graph'])
                valid = (spike != 0) & (predicted[gi] != 0) & (exact[qi] != 0)
                wrong = valid & (np.sign(predicted[gi]) != np.sign(exact[qi]))
                assert record['n'] == int(valid.sum()) and record['wrong'] == int(wrong.sum())
                if valid.any():
                    check(wrong.sum()/valid.sum(), record['fraction'], 'sign_fraction')
                else:
                    assert record['fraction'] is None
            seed_scores.append(float(primary.mean()))
        result = json.loads((args.results/'summary.json').read_text())
        rng = np.random.default_rng(2026101501)
        values = np.array(seed_scores)
        boot = values[rng.integers(0, 4, (10000, 4))].mean(1)
        check(values, result['seed_values'], 'seed_scores')
        check(values.mean(), result['mean'], 'primary_mean')
        check(np.quantile(boot, [.025, .975], method='linear'), result['percentile_95_interval'], 'interval')
        assert result['ability_gate_all_seeds'] == all(s['ability_gate'] for s in model_summaries)
        assert result['directional_hypothesis_mean_positive'] == (values.mean() > 0)
        assert (result['calls'], result['examples'], result['training_updates'], result['control_rows']) == (5160, 654868, 4800, 1688)
        assert (calls, examples) == (360, 43188)
        receipt.update(status='passed', seed_scores=seed_scores, primary_mean=float(values.mean()),
                       replayed_test_logits=71880, replayed_panel_logits=360000,
                       replayed_binary_trace_values=27648000, training_record_rows=4800)
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error), context=context)
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start, comparisons=checks, maxima=maxima,
                       full_forward_calls=calls, forwarded_examples=examples)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
