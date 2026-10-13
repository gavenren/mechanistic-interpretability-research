#!/usr/bin/env python3
"""Replay saved register-study outputs without importing its runner or retraining."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
SEEDS = [17011, 17021, 17027, 17033]
ROUTES = ['base', 'natural_target', 'initial', 'immediate', 'transition',
          'same_state_immediate', 'same_state_transition']
STAGES = ['immediate', 'copy_write_first', 'copy_write_final',
          'write_copy_first', 'write_copy_final']
TOL = 1e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0
        self.maxima = {}

    def true(self, value, label):
        self.count += 1
        if not bool(value):
            raise AssertionError(label)

    def close(self, actual, expected, label, group='arithmetic', tol=TOL):
        a, b = np.asarray(actual), np.asarray(expected)
        self.true(a.shape == b.shape, label + ': shape')
        self.true(np.isfinite(a).all() and np.isfinite(b).all(), label + ': finite')
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.maxima[group] = max(self.maxima.get(group, 0.0), error)
        self.true(error <= tol, f'{label}: error {error} exceeds {tol}')


def one_step(state, op):
    a, b = divmod(int(state), 4)
    if op < 4:
        a = op
    elif op < 8:
        b = op - 4
    elif op == 8:
        b = a
    elif op == 9:
        a = b
    elif op == 10:
        a = (a + 1) % 4
    elif op == 11:
        b = (b + 1) % 4
    elif op == 12:
        a, b = b, a
    elif op != 13:
        raise ValueError(op)
    return 4 * a + b


def oracle(history):
    state, states = 0, []
    for op in history:
        state = one_step(state, op)
        states.append(state)
    return states


def check_inputs(data, check):
    ops = [f'WRITE_A_{i}' for i in range(4)] + [f'WRITE_B_{i}' for i in range(4)]
    ops += ['COPY_A_TO_B', 'COPY_B_TO_A', 'INC_A', 'INC_B', 'SWAP', 'NOP']
    check.true(data['operations'] == ops, 'operation order')
    used, factual = set(), []
    random = np.random.default_rng(71201)
    while len(factual) < 1024:
        history = tuple(random.integers(14, size=12).tolist())
        if history not in used:
            used.add(history)
            factual.append(list(history))
    check.true(factual == data['factual_histories'], 'factual input generation')
    check.true([oracle(h) for h in factual] == data['factual_states_after_each_operation'],
               'all factual labels')
    for name, seed, quota in [('train_receiver', 71203, 16), ('train_donor', 71204, 16),
                              ('test_receiver', 71205, 8), ('test_donor', 71206, 8)]:
        random = np.random.default_rng(seed)
        buckets = [[] for _ in range(16)]
        draws = 0
        while min(map(len, buckets)) < quota:
            draws += 1
            check.true(draws <= 1000000, 'history generation bound')
            history = tuple(random.integers(14, size=12).tolist())
            if history not in used:
                state = oracle(history)[-1]
                if len(buckets[state]) < quota:
                    used.add(history)
                    buckets[state].append(list(history))
        bank = data['banks'][name]
        expected = [h for bucket in buckets for h in bucket]
        check.true(bank['histories'] == expected and bank['draws'] == draws and bank['seed'] == seed,
                   name + ': exact generated histories')
        check.true(bank['final_states'] == [oracle(h)[-1] for h in expected], name + ': oracle labels')
    check.true(len(used) == 1792, 'disjoint history count')
    check.close(data['factual_schedule'], np.random.default_rng(71202).integers(0, 1024, (1500, 64)),
                'factual batch schedule', tol=0)
    random = np.random.default_rng(71207)
    expected = np.stack([random.integers(0, 256, (1000, 64)), random.integers(1, 4, (1000, 64)),
                         random.integers(0, 4, (1000, 64)), random.integers(0, 16, (1000, 64)),
                         random.integers(0, 14, (1000, 64))], axis=-1)
    check.close(data['editor_schedule'], expected, 'edit request schedule', tol=0)


def make_grid(data):
    rows = []
    for receiver, state in enumerate(data['banks']['test_receiver']['final_states']):
        a, b = divmod(state, 4)
        within = receiver % 8
        for new in range(4):
            if new == a:
                continue
            for donor_b in range(4):
                for write in range(4):
                    if write != a and write != new:
                        rows.append([receiver, state, new, donor_b, write,
                                     8 * (4 * new + donor_b) + within,
                                     8 * (4 * new + b) + (within + 1) % 8,
                                     8 * state + (within + 1) % 8])
    return np.array(rows, dtype=np.int64)


def targets(grid, factual=False):
    a = grid[:, 1] // 4 if factual else grid[:, 2]
    b, w = grid[:, 1] % 4, grid[:, 4]
    return np.stack([4 * a + b, 5 * a, 4 * w + a, 4 * w + b, 5 * w], axis=1)


class ManualGRU:
    """Explicit PyTorch gate equations; does not call nn.GRU or the study runner."""
    def __init__(self, state):
        self.state = {k: torch.from_numpy(v) for k, v in state.items()}
        self.eye = torch.eye(14, dtype=torch.float64)

    def advance(self, h, op):
        s = self.state
        x = self.eye[op]
        xi = x @ s['gru.weight_ih_l0'].T + s['gru.bias_ih_l0']
        hh = h @ s['gru.weight_hh_l0'].T + s['gru.bias_hh_l0']
        xr, xz, xn = xi.chunk(3, dim=-1)
        hr, hz, hn = hh.chunk(3, dim=-1)
        reset = torch.sigmoid(xr + hr)
        update = torch.sigmoid(xz + hz)
        new = torch.tanh(xn + reset * hn)
        return (1 - update) * new + update * h

    def encode(self, histories):
        history = torch.as_tensor(histories, dtype=torch.long)
        h = torch.zeros((len(history), 64), dtype=torch.float64)
        for op in history.T:
            h = self.advance(h, op)
        return h

    def read(self, h):
        return h @ self.state['readout.weight'].T + self.state['readout.bias']

    def continue_both(self, h, writes):
        copy = torch.full_like(writes, 8)
        first_c = self.advance(h, copy)
        first_w = self.advance(h, writes)
        states = [h, first_c, self.advance(first_c, writes), first_w, self.advance(first_w, copy)]
        return torch.stack([self.read(s) for s in states], dim=1)


def score(logits, truth):
    pred = logits.argmax(-1)
    correct = pred == truth
    chosen = np.take_along_axis(logits, truth[..., None], -1)[..., 0]
    mask = np.eye(16, dtype=bool)[truth]
    margins = chosen - np.where(mask, -np.inf, logits).max(-1)
    shifted = logits - logits.max(-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(-1, keepdims=True)
    chosen_p = np.take_along_axis(probabilities, truth[..., None], -1)[..., 0]
    contracts = correct[:, 0] & correct[:, 2] & correct[:, 4]
    return pred, correct, margins, chosen_p, contracts


def read_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as stream:
        yield from csv.DictReader(stream)


def check_training(results, check):
    counts, prior = Counter(), {}
    for row in read_csv(results / 'training.csv.gz'):
        key = int(row['seed']), row['stage']
        check.true(key[0] in SEEDS and key[1] in ('factual', 'immediate', 'transition'), 'training identity')
        step = int(row['update'])
        check.true(step == prior.get(key, 0) + 1, 'training update sequence')
        prior[key] = step
        counts[key] += 1
        loss = float(row['loss'])
        check.true(np.isfinite(loss) and loss >= 0, 'finite recorded training loss')
        if key[1] == 'factual':
            check.true(row['immediate_loss'] == row['transition_loss'] == '', 'factual loss fields')
        elif key[1] == 'immediate':
            check.close(loss, float(row['immediate_loss']), 'immediate recorded objective')
            check.true(row['transition_loss'] == '', 'immediate has no future loss')
        else:
            check.close(loss, (float(row['immediate_loss']) + float(row['transition_loss'])) / 2,
                        'transition recorded objective')
    for seed in SEEDS:
        for stage in ('factual', 'immediate', 'transition'):
            check.true(counts[seed, stage] == (1500 if stage == 'factual' else 1000), 'training row count')
    return sum(counts.values())


def check_controls(results, check):
    rows = list(read_csv(results / 'controls.csv'))
    expected = set()
    for seed in SEEDS:
        for first in range(0, 3072, 256):
            for route in ('initial', 'immediate', 'transition'):
                kinds = ['orthonormal', 'idempotent']
                if route != 'initial':
                    kinds += ['self_hidden', 'self_logits']
                expected.update((seed, route, first, k) for k in kinds)
            expected.add((seed, 'natural_target', first, 'full_donor_interface'))
    observed = [(int(r['seed']), r['route'], int(r['batch_first']), r['kind']) for r in rows]
    check.true(len(rows) == 528 and len(set(observed)) == 528 and set(observed) == expected,
               'complete numerical-control grid')
    errors = np.array([float(r['max_error']) for r in rows])
    check.true(np.isfinite(errors).all() and (errors >= 0).all() and (errors <= TOL).all(),
               'saved numerical-control bounds')
    return float(errors.max())


def run(args, check, receipt):
    result = args.results.resolve()
    meta = json.loads((result / 'run-metadata.json').read_text())
    check.true(meta['status'] == 'complete', 'completed run required')
    check.true(meta['versions']['numpy'] == np.__version__ == '2.3.5', 'NumPy version')
    check.true(meta['versions']['torch'] == torch.__version__ == '2.8.0', 'PyTorch version')
    check.true(platform.python_version_tuple()[:2] == ('3', '12'), 'Python version')
    check.true(meta['seeds'] == SEEDS and meta['dtype'] == 'float64' and meta['threads'] == 2,
               'fixed seeds, dtype, and threads')
    check.true(meta['elapsed_seconds'] <= 1200, 'fixed run time limit')
    repo = STUDY.parents[1]
    for name, key in [('analysis-plan.md', 'plan_sha256'), ('code/run.py', 'runner_sha256'),
                      ('inputs.json', 'inputs_sha256'), ('requirements.txt', None), ('sources.md', None)]:
        path = STUDY / name
        locked = subprocess.check_output(['git', 'show', meta['lock_commit'] + ':' + path.relative_to(repo).as_posix()], cwd=repo)
        check.true(locked == path.read_bytes(), 'locked file bytes: ' + name)
        if key:
            check.true(digest(path) == meta[key], 'recorded input identity: ' + name)
    manifest = json.loads((result / 'manifest.json').read_text())
    actual_names = {p.relative_to(result).as_posix() for p in result.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    check.true(set(manifest) == actual_names, 'complete result manifest')
    for name, wanted in manifest.items():
        check.true(digest(result / name) == wanted, 'result hash: ' + name)
    data = json.loads((STUDY / 'inputs.json').read_text())
    check_inputs(data, check)
    grid = make_grid(data)
    check.true(grid.shape == (3072, 8), 'test-grid shape')
    desired, factual = targets(grid), targets(grid, factual=True)
    summary = json.loads((result / 'summary.json').read_text())
    check.true([s['seed'] for s in summary['seeds']] == SEEDS, 'summary seed order')
    receipt['training_rows_checked'] = check_training(result, check)
    receipt['saved_control_maximum'] = check_controls(result, check)
    row_iter = iter(read_csv(result / 'contracts.csv.gz'))
    differences, ability_passes, contract_rows = [], [], 0
    shape = {'gru.weight_ih_l0': (192, 14), 'gru.weight_hh_l0': (192, 64),
             'gru.bias_ih_l0': (192,), 'gru.bias_hh_l0': (192,),
             'readout.weight': (16, 64), 'readout.bias': (16,)}
    for seed_index, seed in enumerate(SEEDS):
        for step in range(250, 1501, 250):
            with np.load(result / f'model-{seed}-step-{step}.npz') as arrays:
                check.true(set(arrays.files) == set(shape), 'model weight keys')
                for key, dims in shape.items():
                    check.true(arrays[key].shape == dims and arrays[key].dtype == np.float64
                               and np.isfinite(arrays[key]).all(), 'model weight shape and values')
                if step == 1500:
                    weights = dict(arrays)
        model = ManualGRU(weights)
        hidden_r = model.encode(data['banks']['test_receiver']['histories'])
        hidden_d = model.encode(data['banks']['test_donor']['histories'])
        h, d = hidden_r[grid[:, 0]], hidden_d[grid[:, 5]]
        writes = torch.from_numpy(grid[:, 4])
        basis = {}
        for route in ('initial', 'immediate', 'transition'):
            steps = [None] if route == 'initial' else range(250, 1001, 250)
            for step in steps:
                suffix = 'initial' if step is None else f'{route}-step-{step}'
                arrays = dict(np.load(result / f'editor-{seed}-{suffix}.npz'))
                check.true(set(arrays) == {'W', 'U'}, 'edit weight keys')
                w, u = torch.from_numpy(arrays['W']), torch.from_numpy(arrays['U'])
                check.true(w.shape == u.shape == (64, 4), 'edit weight shape')
                expected_u = torch.linalg.qr(w, mode='reduced')[0]
                check.close(u.numpy(), expected_u.numpy(), 'saved QR basis', 'basis')
                check.close((u.T @ u).numpy(), np.eye(4), 'basis orthonormality', 'basis')
                p = u @ u.T
                check.close((p @ p).numpy(), p.numpy(), 'projection idempotence', 'basis')
                if route == 'initial':
                    torch.manual_seed(81000 + seed)
                    check.close(w.numpy(), torch.randn(64, 4, dtype=torch.float64).numpy(),
                                'initial edit seed', tol=0)
                basis[route] = u
        saved_scores, saved_logits = {}, {}
        seed_summary = summary['seeds'][seed_index]
        check.true(json.loads((result / f'summary-{seed}.json').read_text()) == seed_summary,
                   'individual seed summary')
        for route in ROUTES:
            if route == 'base':
                edited = h
            elif route == 'natural_target':
                edited = hidden_d[grid[:, 6]]
            else:
                key = route.removeprefix('same_state_')
                u = basis[key]
                donor = hidden_d[grid[:, 7]] if route.startswith('same_state_') else d
                edited = h + (donor - h) @ (u @ u.T)
            replay = model.continue_both(edited, writes).numpy()
            norms = (edited - h).norm(dim=-1).numpy()
            truth = factual if route.startswith('same_state_') else desired
            arrays = dict(np.load(result / f'logits-{seed}-{route}.npz'))
            check.close(arrays['contracts'], grid, 'saved contract grid', tol=0)
            check.close(arrays['targets'], truth, 'saved target grid', tol=0)
            check.close(replay, arrays['logits'], f'full logit replay {seed} {route}', 'logits')
            check.close(norms, arrays['edit_norm'], 'edit-norm replay', 'edit_norm')
            logits = arrays['logits']
            pred, correct, margins, probabilities, success = score(logits, truth)
            check.true(np.array_equal(replay.argmax(-1), pred), 'replayed argmax identities')
            group = seed_summary['routes'][route]
            expected = {'contracts': 3072, 'contract_success': success.mean(),
                        'mean_edit_norm': arrays['edit_norm'].mean(),
                        'immediate_A_accuracy': (pred[:, 0] // 4 == truth[:, 0] // 4).mean(),
                        'immediate_B_accuracy': (pred[:, 0] % 4 == truth[:, 0] % 4).mean()}
            for j, stage in enumerate(STAGES):
                expected.update({stage + '_accuracy': correct[:, j].mean(),
                                 stage + '_margin': margins[:, j].mean(),
                                 stage + '_probability': probabilities[:, j].mean()})
            for key, value in expected.items():
                check.close(group[key], value, 'route summary: ' + key)
            same_b = grid[:, 3] == grid[:, 1] % 4
            for key, mask in [('same', same_b), ('different', ~same_b)]:
                strata = group['donor_B_strata'][key]
                check.true(strata['contracts'] == int(mask.sum()), 'donor-B stratum count')
                check.close(strata['contract_success'], success[mask].mean(), 'donor-B stratum score')
            for i in range(3072):
                row = next(row_iter)
                identity = [int(row[k]) for k in ['seed', 'contract', 'receiver', 'receiver_state', 'new_A', 'donor_B', 'write_A']]
                check.true(identity == [seed, i, *grid[i, :5]] and row['route'] == route,
                           'contract table identity')
                check.true(int(row['contract_success']) == success[i], 'contract conjunction')
                check.close(float(row['edit_norm']), arrays['edit_norm'][i], 'contract edit norm')
                for j, stage in enumerate(STAGES):
                    check.true(int(row[stage + '_target']) == truth[i, j], 'contract target')
                    check.true(int(row[stage + '_prediction']) == pred[i, j], 'contract prediction')
                    check.close(float(row[stage + '_margin']), margins[i, j], 'contract margin')
                    check.close(float(row[stage + '_probability']), probabilities[i, j], 'contract probability')
                contract_rows += 1
            saved_scores[route], saved_logits[route] = success.mean(), logits
            if route in ('immediate', 'transition'):
                self_edit = h + (h - h) @ (basis[route] @ basis[route].T)
                check.close(self_edit.numpy(), h.numpy(), 'self hidden', 'self', tol=0)
                check.close(model.continue_both(self_edit, writes).numpy(), model.continue_both(h, writes).numpy(),
                            'self current and future readouts', 'self', tol=0)
            if route == 'natural_target':
                histories = np.asarray(data['banks']['test_donor']['histories'])[grid[:, 6]]
                copy = np.full(3072, 8)
                cw = np.column_stack([histories, copy, grid[:, 4]])
                wc = np.column_stack([histories, grid[:, 4], copy])
                check.close(model.read(model.encode(cw)).numpy(), replay[:, 2], 'complete copy/write donor continuation', 'natural_interface')
                check.close(model.read(model.encode(wc)).numpy(), replay[:, 4], 'complete write/copy donor continuation', 'natural_interface')
        factual_ok = saved_logits['base'].argmax(-1) == factual
        natural_ok = saved_logits['natural_target'].argmax(-1) == desired
        ability = {'factual_current': factual_ok[:, 0].mean(), 'factual_first_steps': factual_ok[:, [1, 3]].mean(),
                   'factual_final_steps': factual_ok[:, [2, 4]].mean(), 'natural_target_all_steps': natural_ok.mean()}
        for key, value in ability.items():
            check.close(seed_summary['ability'][key], value, 'ability measure')
        passed = all(v >= 0.99 for v in ability.values())
        check.true(seed_summary['ability']['passes'] == passed, 'ability gate')
        ability_passes.append(passed)
        difference = saved_scores['transition'] - saved_scores['immediate']
        differences.append(difference)
        check.close(seed_summary['primary_difference'], difference, 'paired primary difference')
        print(json.dumps({'replayed_seed': seed, 'max_logit_error': check.maxima['logits']}), flush=True)
    check.true(next(row_iter, None) is None, 'no extra contract rows')
    differences = np.array(differences)
    random = np.random.default_rng(71208)
    resampled = differences[random.integers(0, 4, (10000, 4))].mean(axis=1)
    interval = np.quantile(resampled, [.025, .975])
    check.close(summary['primary_mean_difference'], differences.mean(), 'primary mean')
    check.close(summary['primary_seed_bootstrap_interval'], interval, 'fixed seed bootstrap')
    check.true(summary['positive_seeds'] == int((differences > 0).sum()), 'positive seed count')
    check.true(summary['directional_support_rule_passes'] == bool(differences.mean() > 0 and (differences > 0).sum() >= 3),
               'directional support rule')
    check.true(summary['all_models_pass_ability'] == all(ability_passes), 'joint ability gate')
    expected_counts = {'training_rows': 14000, 'contract_rows': 86016, 'control_rows': 528,
                       'factual_optimizer_updates': 6000, 'edit_optimizer_updates': 8000,
                       'factual_training_GRU_positions': 4608000, 'transition_training_extra_GRU_positions': 256000,
                       'immediate_arm_training_readouts': 256000, 'transition_arm_training_readouts': 512000,
                       'history_encoding_GRU_positions': 36864, 'test_and_control_GRU_positions': 884736,
                       'test_and_control_readouts': 1019904, 'test_contracts_per_route_per_model': 3072,
                       'saved_test_readouts': 430080}
    for key, expected in expected_counts.items():
        check.true(meta[key] == expected, 'metadata count: ' + key)
    receipt.update({'lock_commit': meta['lock_commit'], 'result_manifest_sha256': digest(result / 'manifest.json'),
                    'result_files_verified': len(manifest), 'model_checkpoints_checked': 24,
                    'editor_checkpoints_checked': 36, 'final_model_replays': 4,
                    'contract_rows_replayed': contract_rows, 'full_logit_values_replayed': contract_rows * 5 * 16,
                    'numerical_control_rows_checked': 528, 'training_trajectories_replayed': False,
                    'primary_mean_difference': float(differences.mean()),
                    'primary_seed_bootstrap_interval': interval.tolist(),
                    'scope': 'All saved test logits replayed with explicit GRU gate equations and matrix-projection edits. All input rules, targets, table scores, seed summaries, and primary calculations checked. Training records and saved checkpoints checked; training was not repeated.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to replace an audit receipt')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    check = Checks()
    started = time.monotonic()
    receipt = {'status': 'running', 'review_kind': 'independent AI review, not human peer review',
               'audit_sha256': digest(__file__), 'fixed_tolerance': TOL,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__}}
    try:
        with torch.inference_mode():
            run(args, check, receipt)
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        receipt.update({'check_count': check.count, 'maximum_errors': check.maxima,
                        'elapsed_seconds': time.monotonic() - started})
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
