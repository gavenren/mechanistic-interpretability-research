#!/usr/bin/env python3
"""Fixed finite matching study; model outcomes require a public byte lock."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [19121, 19139, 19157, 19181]
TOL = 1e-10
CAP = 1200


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def array_identity(x):
    return {'dtype': x.dtype.str, 'shape': list(x.shape),
            'sha256': hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}


def configure():
    assert platform.python_version_tuple()[:2] == ('3', '12')
    assert np.__version__ == '2.3.5' and torch.__version__ == '2.8.0'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)


def data_and_ledger():
    ids = np.arange(65536, dtype=np.uint32)
    graphs = ((ids[:, None] >> np.arange(16)) & 1).astype(np.uint8).reshape(-1, 4, 4)
    labels = np.zeros(len(graphs), dtype=bool)
    for permutation in itertools.permutations(range(4)):
        labels |= graphs[:, np.arange(4), permutation].all(axis=1)
    failures, pair_failure = [], np.zeros(len(graphs), dtype=bool)
    for subset in range(1, 16):
        rows = [i for i in range(4) if subset & (1 << i)]
        fail = graphs[:, rows].any(axis=1).sum(axis=1) < len(rows)
        failures.append(fail)
        if len(rows) == 2:
            pair_failure |= fail
    assert np.array_equal(labels, ~np.stack(failures).any(axis=0))
    degrees = np.concatenate([graphs.sum(axis=2), graphs.sum(axis=1)], axis=1).astype(np.uint8)
    no_isolated = degrees.min(axis=1) > 0
    other_cause = no_isolated & ~labels & ~pair_failure
    degree_groups = {}
    for graph_id, degree in enumerate(degrees.tolist()):
        degree_groups.setdefault(tuple(degree), []).append(graph_id)
    names, unions = [], {phase: {kind: set() for kind in ('target', 'positive', 'hall3')}
                        for phase in ('selection', 'test')}
    for pair in itertools.combinations(range(4), 2):
        for right in range(4):
            target = (graphs[:, pair[0], right] == 1) & (graphs[:, pair[1], right] == 1)
            target &= (degrees[:, pair[0]] == 1) & (degrees[:, pair[1]] == 1) & no_isolated
            name = f'L{pair[0]}L{pair[1]}_R{right}'
            groups = []
            for key in sorted(set(tuple(degrees[i]) for i in np.flatnonzero(target))):
                count = sum(key[:4])
                if count not in (6, 7):
                    continue
                members = np.array(degree_groups[key], dtype=np.int64)
                t = members[target[members]].tolist()
                p = members[labels[members] & no_isolated[members]].tolist()
                c = members[other_cause[members]].tolist()
                if t and p and c:
                    phase = 'selection' if count == 6 else 'test'
                    groups.append({'degree': [int(v) for v in key], 'phase': phase,
                                   'target': t, 'positive': p, 'hall3': c})
                    for kind, values in [('target', t), ('positive', p), ('hall3', c)]:
                        unions[phase][kind].update(values)
            for phase, expected in [('selection', (3, 6, 78, 12)), ('test', (6, 18, 48, 12))]:
                selected = [g for g in groups if g['phase'] == phase]
                counts = (len(selected), *[len({i for g in selected for i in g[k]})
                                          for k in ('target', 'positive', 'hall3')])
                assert counts == expected
            names.append({'name': name, 'left_pair': list(pair), 'right': right, 'groups': groups})
    random = np.random.default_rng(94301)
    masks = [sorted(random.choice(64, size=4, replace=False).tolist()) for _ in range(8)]
    random = np.random.default_rng(94210)
    epochs = np.stack([random.permutation(65536).astype(np.int32) for _ in range(64)])
    union_lists = {phase: {kind: sorted(values) for kind, values in kinds.items()}
                   for phase, kinds in unions.items()}
    selection = set().union(*unions['selection'].values())
    test = set().union(*unions['test'].values())
    assert not selection.intersection(test)
    ledger = {'encoding': 'bit 4*left+right gives edge; vertex IDs start at zero',
              'graph_count': 65536, 'positive_count': int(labels.sum()),
              'no_isolated_negative_count': int((no_isolated & ~labels).sum()),
              'strict_hall3_negative_count': int(other_cause.sum()),
              'names': names, 'category_unions': union_lists,
              'phase_unique_graph_counts': {'selection': len(selection), 'test': len(test)},
              'random_masks': masks, 'random_mask_seed': 94301,
              'epoch_seed': 94210, 'epoch_draw_rule': '64 sequential generator.permutation(65536) calls; cast each to int32',
              'arrays': {'graph_ids': array_identity(ids), 'labels': array_identity(labels),
                         'epoch_orders': array_identity(epochs)}}
    return graphs, labels, no_isolated, epochs, ledger


def verify_lock(commit):
    repo = ROOT.parents[1]
    exact = subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'], cwd=repo).decode().strip()
    if exact != commit:
        raise RuntimeError('Use a full immutable commit hash')
    for name in ('analysis-plan.md', 'code/run.py', 'inputs.json', 'sources.md', 'requirements.txt'):
        path = ROOT / name
        public = subprocess.check_output(['git', 'show', commit + ':' + path.relative_to(repo).as_posix()], cwd=repo)
        if public != path.read_bytes():
            raise RuntimeError('Lock mismatch: ' + name)


def budget(start):
    if time.monotonic() - start > CAP:
        raise RuntimeError('The fixed 20-minute time limit was reached')


class Table:
    def __init__(self, path):
        self.stream = gzip.open(path, 'wt', newline='') if str(path).endswith('.gz') else Path(path).open('w', newline='')
        self.writer, self.count = None, 0

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count += 1

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(16, 64, dtype=torch.float64)
        self.second = nn.Linear(64, 64, dtype=torch.float64)
        self.last = nn.Linear(64, 1, dtype=torch.float64)

    def hidden(self, x):
        return self.first(x).relu()

    def finish(self, h):
        return self.last(self.second(h).relu()).squeeze(-1)

    def forward(self, x):
        return self.finish(self.hidden(x))


def save_weights(path, model):
    np.savez_compressed(path, **{key: value.detach().numpy() for key, value in model.state_dict().items()})


def fit(model, x, y, epochs, seed, output, trace, start):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    save_weights(output / f'model-{seed}-initial.npz', model)
    step = 0
    for epoch, order in enumerate(epochs):
        for first in range(0, 65536, 512):
            budget(start)
            indices = order[first:first + 512].astype(np.int64)
            logits = model(x[indices])
            loss = F.binary_cross_entropy_with_logits(logits, y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            trace.add({'seed': seed, 'update': step, 'epoch': epoch,
                       'first_in_epoch': first, 'loss': loss.item()})
            if step % 1024 == 0:
                save_weights(output / f'model-{seed}-step-{step}.npz', model)
                trace.flush()
    assert step == 8192
    model.eval()


def mask_library(ledger):
    names = ['empty', 'full'] + [f'unit_{i}' for i in range(64)] + [f'random_{i}' for i in range(8)]
    masks = np.zeros((74, 64), dtype=bool)
    masks[1] = True
    masks[2:66] = np.eye(64, dtype=bool)
    for i, units in enumerate(ledger['random_masks']):
        masks[66 + i, units] = True
    return names, masks


def group_effect(logit, intact, lookup, group):
    differences = logit - intact
    effects = [float(differences[lookup[group[kind]]].mean()) for kind in ('target', 'positive', 'hall3')]
    score = effects[0] - max(0.0, effects[1], effects[2])
    return score, effects


def control(controls, seed, mask, kind, error):
    controls.add({'seed': seed, 'mask': mask, 'kind': kind, 'max_error': float(error)})
    controls.flush()
    if not np.isfinite(error) or error > TOL:
        raise RuntimeError(f'Numerical check failed: {kind} {error}')


def evaluate(model, seed, x, labels, no_isolated, ledger, output, controls, scores, start):
    graph_ids = np.array(sorted({i for name in ledger['names'] for group in name['groups']
                                 for kind in ('target', 'positive', 'hall3') for i in group[kind]}), dtype=np.int64)
    lookup = np.full(65536, -1, dtype=np.int64)
    lookup[graph_ids] = np.arange(len(graph_ids))
    with torch.inference_mode():
        intact_parts = []
        for first in range(0, 65536, 512):
            budget(start)
            intact_parts.append(model(x[first:first + 512]).numpy())
        intact_all = np.concatenate(intact_parts)
        base = intact_all[graph_ids]
        h = model.hidden(x[graph_ids])
        control(controls, seed, 'all', 'cached_forward', np.max(np.abs(model.finish(h).numpy() - base)))
        mask_names, masks = mask_library(ledger)
        outputs, norms = [], []

        def one_mask(name, mask):
            budget(start)
            changed = h.clone()
            changed[:, mask] = 0
            control(controls, seed, name, 'untouched_entries',
                    (changed[:, ~mask] - h[:, ~mask]).abs().max().item() if (~mask).any() else 0.0)
            control(controls, seed, name, 'selected_zero',
                    changed[:, mask].abs().max().item() if mask.any() else 0.0)
            restored = changed.clone()
            restored[:, mask] = h[:, mask]
            control(controls, seed, name, 'restore_hidden', (restored - h).abs().max().item())
            values = []
            for first in range(0, len(h), 512):
                budget(start)
                values.append(model.finish(changed[first:first + 512]).numpy())
            return np.concatenate(values), (changed - h).norm(dim=-1).numpy()

        for name, mask in zip(mask_names, masks):
            values, removed = one_mask(name, mask)
            outputs.append(values)
            norms.append(removed)
        control(controls, seed, 'empty', 'output_identity', np.max(np.abs(outputs[0] - base)))
        full_expected = (model.second.bias.relu() @ model.last.weight[0] + model.last.bias[0]).item()
        control(controls, seed, 'full', 'bias_only_output', np.max(np.abs(outputs[1] - full_expected)))
        selections = []
        for name in ledger['names']:
            cal = [group for group in name['groups'] if group['phase'] == 'selection']
            unit_scores = [float(np.mean([group_effect(outputs[2 + unit], base, lookup, group)[0]
                                          for group in cal])) for unit in range(64)]
            ranking = sorted(range(64), key=lambda unit: (-unit_scores[unit], unit))
            chosen = sorted(ranking[:4])
            mask = np.zeros(64, dtype=bool)
            mask[chosen] = True
            mask_name = 'selected_' + name['name']
            values, removed = one_mask(mask_name, mask)
            mask_names.append(mask_name)
            masks = np.concatenate([masks, mask[None]], axis=0)
            outputs.append(values)
            norms.append(removed)
            selections.append({'name': name['name'], 'calibration_unit_scores': unit_scores,
                               'ranking': ranking, 'selected_units': chosen})
        outputs, norms = np.stack(outputs), np.stack(norms)
        np.savez_compressed(output / f'ablations-{seed}.npz', graph_ids=graph_ids, masks=masks,
                            logits=outputs, removed_norm=norms, intact_logits=base)
        np.savez_compressed(output / f'intact-{seed}.npz', graph_ids=np.arange(65536, dtype=np.int64),
                            logits=intact_all, labels=labels)
        dump(output / f'masks-{seed}.json', {'mask_names': mask_names, 'selections': selections})
        named = []
        for n, name in enumerate(ledger['names']):
            phase_results = {}
            applicable = list(range(74)) + [74 + n]
            for phase in ('selection', 'test'):
                phase_groups = [g for g in name['groups'] if g['phase'] == phase]
                means = {}
                for mask_id in applicable:
                    group_scores = []
                    for group_index, group in enumerate(phase_groups):
                        value, effects = group_effect(outputs[mask_id], base, lookup, group)
                        group_scores.append(value)
                        row = {'seed': seed, 'name': name['name'], 'phase': phase,
                               'degree_group': group_index, 'degree': ','.join(map(str, group['degree'])),
                               'mask_id': mask_id, 'mask': mask_names[mask_id], 'selectivity': value}
                        for k, kind in enumerate(('target', 'positive', 'hall3')):
                            idx = lookup[group[kind]]
                            before, after = base[idx], outputs[mask_id, idx]
                            row.update({kind + '_count': len(idx), kind + '_logit_change': effects[k],
                                        kind + '_intact_logit': float(before.mean()), kind + '_ablated_logit': float(after.mean()),
                                        kind + '_probability_change': float((torch.from_numpy(after).sigmoid() - torch.from_numpy(before).sigmoid()).mean()),
                                        kind + '_intact_acceptance': float((before >= 0).mean()),
                                        kind + '_ablated_acceptance': float((after >= 0).mean()),
                                        kind + '_removed_norm': float(norms[mask_id, idx].mean())})
                        row['target_minus_positive'] = effects[0] - effects[1]
                        row['target_minus_hall3'] = effects[0] - effects[2]
                        scores.add(row)
                        budget(start)
                    means[mask_names[mask_id]] = float(np.mean(group_scores))
                selected_score = means['selected_' + name['name']]
                random_score = float(np.mean([means[f'random_{i}'] for i in range(8)]))
                phase_results[phase] = {'selected': selected_score, 'random_mean': random_score,
                                         'selected_minus_random': selected_score - random_score,
                                         'full_layer': means['full'], 'mask_scores': means}
            named.append({'name': name['name'], **phase_results})
        scores.flush()
        correct = (intact_all >= 0) == labels
        gates = {'all_graphs': float(correct.mean()),
                 'no_isolated_negatives': float(correct[no_isolated & ~labels].mean())}
        for kind, ids in ledger['category_unions']['test'].items():
            gates['test_' + kind] = float(correct[ids].mean())
        all_gates = all(value >= .99 for value in gates.values())
        selected = float(np.mean([n['test']['selected'] for n in named]))
        random_score = float(np.mean([n['test']['random_mean'] for n in named]))
        result = {'seed': seed, 'ability': gates, 'ability_passes': all_gates,
                  'all_graph_cross_entropy': float(F.binary_cross_entropy_with_logits(torch.from_numpy(intact_all), torch.from_numpy(labels.astype(np.float64)))),
                  'raw_selected_selectivity': selected, 'random_selectivity': random_score,
                  'primary_difference': selected - random_score,
                  'positive_raw_names': sum(n['test']['selected'] > 0 for n in named),
                  'raw_name_range': [min(n['test']['selected'] for n in named), max(n['test']['selected'] for n in named)],
                  'per_model_interpretation_conditions_pass': bool(all_gates and selected > 0 and selected > random_score),
                  'named_obstructions': named, 'unique_intervention_graphs': len(graph_ids),
                  'mask_count': len(mask_names), 'saved_ablation_outputs': int(outputs.size)}
        dump(output / f'summary-{seed}.json', result)
    return result


def run(args):
    verify_lock(args.lock)
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Refusing a nonempty result directory')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    graphs, labels, no_isolated, epochs, ledger = data_and_ledger()
    if ledger != json.loads((ROOT / 'inputs.json').read_text()):
        raise RuntimeError('Inputs differ from the fixed ledger')
    np.savez_compressed(output / 'data.npz', graph_ids=np.arange(65536, dtype=np.uint32),
                        labels=labels, epoch_orders=epochs)
    metadata = {'status': 'running', 'lock_commit': args.lock, 'seeds': SEEDS, 'threads': 2,
                'dtype': 'float64', 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__},
                'locked_sha256': {name: sha(ROOT / name) for name in ('analysis-plan.md', 'code/run.py', 'inputs.json', 'sources.md', 'requirements.txt')}}
    dump(output / 'run-metadata.json', metadata)
    training, controls, scores = Table(output / 'training.csv.gz'), Table(output / 'controls.csv'), Table(output / 'group-scores.csv.gz')
    results = []
    try:
        x = torch.from_numpy(graphs.reshape(-1, 16).astype(np.float64))
        y = torch.from_numpy(labels.astype(np.float64))
        for seed in SEEDS:
            budget(started)
            torch.manual_seed(seed)
            model = Model()
            assert sum(p.numel() for p in model.parameters()) == 5313
            t = time.monotonic()
            fit(model, x, y, epochs, seed, output, training, started)
            train_seconds = time.monotonic() - t
            t = time.monotonic()
            result = evaluate(model, seed, x, labels, no_isolated, ledger, output, controls, scores, started)
            result['seconds'] = {'training': train_seconds, 'evaluation': time.monotonic() - t}
            dump(output / f'summary-{seed}.json', result)
            results.append(result)
            print(json.dumps({'completed_seed': seed, 'primary_difference': result['primary_difference'],
                              'raw_selected': result['raw_selected_selectivity'], 'ability': result['ability'],
                              'seconds': result['seconds']}), flush=True)
        differences = np.array([r['primary_difference'] for r in results])
        random = np.random.default_rng(94302)
        boot = differences[random.integers(0, 4, size=(10000, 4))].mean(axis=1)
        raw = float(np.mean([r['raw_selected_selectivity'] for r in results]))
        summary = {'seeds': results, 'primary_mean_selected_minus_random': float(differences.mean()),
                   'descriptive_seed_interval': np.quantile(boot, [.025, .975]).tolist(),
                   'raw_selected_mean': raw, 'random_mean': float(np.mean([r['random_selectivity'] for r in results])),
                   'all_models_pass_ability': all(r['ability_passes'] for r in results),
                   'panel_interpretation_conditions_pass': all(r['per_model_interpretation_conditions_pass'] for r in results)}
        dump(output / 'summary.json', summary)
        budget(started)
        verify_lock(args.lock)
        assert training.count == 32768 and controls.count == 1188 and scores.count == 64800
        metadata.update({'status': 'complete', 'training_rows': training.count, 'control_rows': controls.count,
                         'group_score_rows': scores.count, 'training_examples': 16777216,
                         'intact_outputs': 4 * 65536,
                         'ablation_outputs': sum(r['saved_ablation_outputs'] for r in results),
                         'elapsed_seconds': time.monotonic() - started, 'scientific_model_outputs_collected': True})
    except BaseException as error:
        metadata.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                         'completed_models': len(results), 'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        for table in (training, controls, scores):
            table.close()
        dump(output / 'run-metadata.json', metadata)
        dump(output / 'manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'})


def preflight(args):
    graphs, labels, no_isolated, epochs, ledger = data_and_ledger()
    assert ledger == json.loads((ROOT / 'inputs.json').read_text())
    assert labels.sum() == 37823 and (no_isolated & ~labels).sum() == 3680
    assert all(np.array_equal(np.sort(order), np.arange(65536)) for order in epochs)
    torch.manual_seed(90351)
    model = Model()
    assert sum(p.numel() for p in model.parameters()) == 5313
    # These artificial nonbinary vectors cannot be any natural task graph.
    dummy = torch.linspace(-.31, .73, 512 * 16, dtype=torch.float64).reshape(512, 16)
    assert not bool(((dummy == 0) | (dummy == 1)).all(dim=1).any())
    _, masks = mask_library(ledger)
    with torch.inference_mode():
        h = model.hidden(dummy)
        errors = {'cached_forward': (model(dummy) - model.finish(h)).abs().max().item()}
        for mask in masks:
            changed = h.clone()
            changed[:, mask] = 0
            assert bool((changed[:, mask] == 0).all())
            assert torch.equal(changed[:, ~mask], h[:, ~mask])
            restored = changed.clone()
            restored[:, mask] = h[:, mask]
            assert torch.equal(restored, h)
        full = model.finish(torch.zeros_like(h))
        expected = model.second.bias.relu() @ model.last.weight[0] + model.last.bias[0]
        errors['full_layer_bias'] = (full - expected).abs().max().item()
        errors['batch_split'] = (model(dummy) - torch.cat([model(dummy[:256]), model(dummy[256:])])).abs().max().item()
    assert all(np.isfinite(v) and v <= TOL for v in errors.values())
    # Train-batch gradient shape check only, on dummy data. No optimizer step.
    F.binary_cross_entropy_with_logits(model(dummy), torch.arange(512, dtype=torch.float64) % 2).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    receipt = {'status': 'passed', 'scientific_model_outputs_collected': False, 'model_training_performed': False,
               'dummy_batch_size': 512, 'dummy_inputs_nonbinary': True, 'errors': errors,
               'graph_count': len(graphs), 'phase_unique_graph_counts': ledger['phase_unique_graph_counts'],
               'plan_sha256': sha(ROOT / 'analysis-plan.md'), 'runner_sha256': sha(__file__), 'inputs_sha256': sha(ROOT / 'inputs.json')}
    path = Path(args.out)
    if path.exists():
        raise RuntimeError('Refusing to overwrite a dummy receipt')
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inputs')
    check = sub.add_parser('preflight')
    check.add_argument('--out', required=True)
    scientific = sub.add_parser('run')
    scientific.add_argument('--lock', required=True)
    scientific.add_argument('--out', required=True)
    args = parser.parse_args()
    configure()
    if args.command == 'inputs':
        path = ROOT / 'inputs.json'
        if path.exists():
            raise RuntimeError('Refusing to overwrite input ledger')
        dump(path, data_and_ledger()[-1])
    elif args.command == 'preflight':
        preflight(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
