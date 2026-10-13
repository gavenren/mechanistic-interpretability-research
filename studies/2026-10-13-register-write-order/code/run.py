#!/usr/bin/env python3
"""Fixed register write-order study. Scientific runs need an immutable public lock."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [17011, 17021, 17027, 17033]
OPS = [f'WRITE_A_{v}' for v in range(4)] + [f'WRITE_B_{v}' for v in range(4)] + [
    'COPY_A_TO_B', 'COPY_B_TO_A', 'INC_A', 'INC_B', 'SWAP', 'NOP']
PACKAGES = {'numpy': '2.3.5', 'torch': '2.8.0'}
TOL = 1e-10
CAP_SECONDS = 1200
ROUTES = ['base', 'natural_target', 'initial', 'immediate', 'transition',
          'same_state_immediate', 'same_state_transition']
STAGES = ['immediate', 'copy_write_first', 'copy_write_final',
          'write_copy_first', 'write_copy_final']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value, compact=False):
    Path(path).write_text(json.dumps(value, indent=None if compact else 2,
                                    sort_keys=True, allow_nan=False) + '\n')


def versions():
    return {'python': platform.python_version(), **{k: version(k) for k in PACKAGES}}


def configure():
    for package, expected in PACKAGES.items():
        if version(package) != expected:
            raise RuntimeError(f'{package} must equal {expected}')
    if platform.python_version_tuple()[:2] != ('3', '12'):
        raise RuntimeError('Python 3.12 is required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)


def transition(states, operations):
    states, operations = np.broadcast_arrays(np.asarray(states, dtype=np.int64),
                                             np.asarray(operations, dtype=np.int64))
    a, b = states // 4, states % 4
    aa, bb = a.copy(), b.copy()
    aa = np.where(operations < 4, operations, aa)
    bb = np.where((operations >= 4) & (operations < 8), operations - 4, bb)
    bb = np.where(operations == 8, a, bb)
    aa = np.where(operations == 9, b, aa)
    aa = np.where(operations == 10, (a + 1) % 4, aa)
    bb = np.where(operations == 11, (b + 1) % 4, bb)
    aa = np.where(operations == 12, b, aa)
    bb = np.where(operations == 12, a, bb)
    if np.any((operations < 0) | (operations > 13)):
        raise ValueError('Operation outside the fixed vocabulary')
    return 4 * aa + bb


def labels(histories):
    histories = np.asarray(histories, dtype=np.int64)
    state = np.zeros(len(histories), dtype=np.int64)
    rows = []
    for column in histories.T:
        state = transition(state, column)
        rows.append(state.copy())
    return np.stack(rows, axis=1)


def make_inputs():
    used = set()
    rng = np.random.default_rng(71201)
    factual = []
    while len(factual) < 1024:
        row = tuple(rng.integers(0, 14, size=12).tolist())
        if row not in used:
            used.add(row)
            factual.append(row)
    banks = {}
    for name, seed, count in [('train_receiver', 71203, 16), ('train_donor', 71204, 16),
                               ('test_receiver', 71205, 8), ('test_donor', 71206, 8)]:
        rng = np.random.default_rng(seed)
        buckets = [[] for _ in range(16)]
        for attempt in range(1_000_000):
            row = tuple(rng.integers(0, 14, size=12).tolist())
            if row in used:
                continue
            state = int(labels([row])[0, -1])
            if len(buckets[state]) < count:
                buckets[state].append(row)
                used.add(row)
            if all(len(bucket) == count for bucket in buckets):
                break
        else:
            raise RuntimeError('Could not fill exact history quotas')
        histories = [list(row) for bucket in buckets for row in bucket]
        banks[name] = {'seed': seed, 'draws': attempt + 1, 'histories': histories,
                       'final_states': labels(histories)[:, -1].tolist()}
    rng = np.random.default_rng(71207)
    schedule = np.stack([rng.integers(0, 256, (1000, 64)),
                         rng.integers(1, 4, (1000, 64)),
                         rng.integers(0, 4, (1000, 64)),
                         rng.integers(0, 16, (1000, 64)),
                         rng.integers(0, 14, (1000, 64))], axis=-1)
    factual_schedule = np.random.default_rng(71202).integers(0, 1024, (1500, 64))
    return {'operations': OPS, 'initial_state': [0, 0], 'history_length': 12,
            'factual_seed': 71201, 'factual_histories': [list(r) for r in factual],
            'factual_states_after_each_operation': labels(factual).tolist(),
            'factual_schedule_seed': 71202, 'factual_schedule': factual_schedule.tolist(),
            'banks': banks, 'editor_schedule_seed': 71207,
            'editor_schedule_fields': ['receiver', 'a_offset', 'donor_b', 'donor_index', 'operation'],
            'editor_schedule': schedule.tolist()}


def inputs_check(data):
    if data != make_inputs():
        raise RuntimeError('Inputs differ from fixed rules')
    histories = data['factual_histories'] + [r for bank in data['banks'].values()
                                          for r in bank['histories']]
    assert len(histories) == 1792 and len({tuple(r) for r in histories}) == 1792
    for bank in data['banks'].values():
        assert labels(bank['histories'])[:, -1].tolist() == bank['final_states']


def verify_lock(commit):
    repo = ROOT.parents[1]
    resolved = subprocess.run(['git', 'rev-parse', commit + '^{commit}'], cwd=repo,
                              check=True, capture_output=True).stdout.decode().strip()
    if resolved != commit:
        raise RuntimeError('A full immutable commit SHA is required')
    for name in ['analysis-plan.md', 'code/run.py', 'inputs.json', 'requirements.txt', 'sources.md']:
        path = ROOT / name
        rel = path.relative_to(repo).as_posix()
        locked = subprocess.run(['git', 'show', commit + ':' + rel], cwd=repo,
                                check=True, capture_output=True).stdout
        if locked != path.read_bytes():
            raise RuntimeError('File differs from the public lock: ' + rel)


class Table:
    def __init__(self, path):
        self.file = gzip.open(path, 'wt', newline='') if str(path).endswith('.gz') else Path(path).open('w', newline='')
        self.writer = None
        self.count = 0

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count += 1

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(14, 64, batch_first=True, dtype=torch.float64)
        self.readout = nn.Linear(64, 16, dtype=torch.float64)

    def encode(self, tokens, hidden=None):
        x = F.one_hot(tokens, 14).to(torch.float64)
        out, final = self.gru(x, None if hidden is None else hidden.unsqueeze(0))
        return out, final[0]

    def advance(self, hidden, operations):
        return self.encode(operations.unsqueeze(1), hidden)[1]

    def forward(self, tokens):
        return self.readout(self.encode(tokens)[0])


def save_weights(path, model):
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})


def basis(weight):
    return torch.linalg.qr(weight, mode='reduced')[0]


def edit(hidden, donor, u):
    return hidden + ((donor - hidden) @ u) @ u.T


def rollout(model, hidden, write):
    copy_op = torch.full_like(write, 8)
    cw_first = model.advance(hidden, copy_op)
    cw_final = model.advance(cw_first, write)
    wc_first = model.advance(hidden, write)
    wc_final = model.advance(wc_first, copy_op)
    return torch.stack([model.readout(h) for h in [hidden, cw_first, cw_final, wc_first, wc_final]], dim=1)


def target_grid(state, write):
    cw1 = transition(state, 8)
    wc1 = transition(state, write)
    return np.stack([state, cw1, transition(cw1, write), wc1, transition(wc1, 8)], axis=1)


def contract_grid(data):
    rows = []
    states = data['banks']['test_receiver']['final_states']
    for receiver, state in enumerate(states):
        a, b = divmod(state, 4)
        within = receiver % 8
        for anew in range(4):
            if anew == a:
                continue
            for bd in range(4):
                for w in range(4):
                    if w in (a, anew):
                        continue
                    rows.append([receiver, state, anew, bd, w,
                                 (4 * anew + bd) * 8 + within,
                                 (4 * anew + b) * 8 + (within + 1) % 8,
                                 state * 8 + (within + 1) % 8])
    grid = np.asarray(rows, dtype=np.int64)
    assert grid.shape == (3072, 8)
    return grid


def check_budget(start):
    if time.monotonic() - start > CAP_SECONDS:
        raise RuntimeError('The fixed 20-minute limit was reached')


def train_model(model, seed, data, out, trace, start):
    tokens = torch.tensor(data['factual_histories'], dtype=torch.long)
    truth = torch.tensor(labels(data['factual_histories']), dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    began = time.monotonic()
    for step, indices in enumerate(data['factual_schedule'], 1):
        check_budget(start)
        logits = model(tokens[indices])
        loss = F.cross_entropy(logits.reshape(-1, 16), truth[indices].reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite factual loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.add({'seed': seed, 'stage': 'factual', 'update': step, 'loss': loss.item(),
                   'immediate_loss': '', 'transition_loss': ''})
        if step % 250 == 0:
            save_weights(out / f'model-{seed}-step-{step}.npz', model)
            trace.flush()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return time.monotonic() - began


def train_editor(model, seed, arm, initial, hidden, data, out, trace, start):
    weight = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([weight], lr=0.01)
    states = np.asarray(data['banks']['train_receiver']['final_states'])
    began = time.monotonic()
    for step, batch in enumerate(np.asarray(data['editor_schedule']), 1):
        check_budget(start)
        receiver, offset, bd, di, op = batch.T
        state = states[receiver]
        anew = (state // 4 + offset) % 4
        donor = (4 * anew + bd) * 16 + di
        target = 4 * anew + state % 4
        h = edit(hidden['train_receiver'][receiver], hidden['train_donor'][donor], basis(weight))
        immediate_loss = F.cross_entropy(model.readout(h), torch.from_numpy(target))
        if arm == 'transition':
            future = model.advance(h, torch.from_numpy(op))
            future_loss = F.cross_entropy(model.readout(future), torch.from_numpy(transition(target, op)))
            loss = (immediate_loss + future_loss) / 2
        else:
            future_loss = None
            loss = immediate_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite edit loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.add({'seed': seed, 'stage': arm, 'update': step, 'loss': loss.item(),
                   'immediate_loss': immediate_loss.item(),
                   'transition_loss': '' if future_loss is None else future_loss.item()})
        if step % 250 == 0:
            np.savez_compressed(out / f'editor-{seed}-{arm}-step-{step}.npz',
                                W=weight.detach().numpy(), U=basis(weight).detach().numpy())
            trace.flush()
    return basis(weight).detach(), time.monotonic() - began


def score_arrays(logits, targets):
    predictions = logits.argmax(axis=-1)
    correct = predictions == targets
    chosen = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    other = logits.copy()
    np.put_along_axis(other, targets[..., None], -np.inf, axis=-1)
    margin = chosen - other.max(axis=-1)
    probability = torch.from_numpy(logits).softmax(-1).numpy()
    probability = np.take_along_axis(probability, targets[..., None], axis=-1)[..., 0]
    return predictions, correct, margin, probability


def evaluate(model, seed, hidden, bases, data, out, controls, row_table, start):
    grid = contract_grid(data)
    states, anew, write = grid[:, 1], grid[:, 2], grid[:, 4]
    desired = target_grid(4 * anew + states % 4, write)
    factual = target_grid(states, write)
    totals, saved, edit_norms = {}, {}, {}
    with torch.inference_mode():
        for route in ROUTES:
            chunks, norms = [], []
            for first in range(0, len(grid), 256):
                check_budget(start)
                g = grid[first:first + 256]
                h = hidden['test_receiver'][g[:, 0]]
                donor = hidden['test_donor'][g[:, 5]]
                w = torch.from_numpy(g[:, 4])
                if route == 'base':
                    changed = h
                elif route == 'natural_target':
                    changed = hidden['test_donor'][g[:, 6]]
                elif route.startswith('same_state_'):
                    changed = edit(h, hidden['test_donor'][g[:, 7]], bases[route[11:]])
                else:
                    changed = edit(h, donor, bases[route])
                result = rollout(model, changed, w)
                chunks.append(result.numpy())
                norms.append((changed - h).norm(dim=-1).numpy())
                if route in ('initial', 'immediate', 'transition'):
                    u = bases[route]
                    identity = torch.eye(4, dtype=torch.float64)
                    projection = u @ u.T
                    errors = {'orthonormal': (u.T @ u - identity).abs().max().item(),
                              'idempotent': (projection @ projection - projection).abs().max().item()}
                    if route != 'initial':
                        errors.update({'self_hidden': (edit(h, h, u) - h).abs().max().item(),
                                       'self_logits': (rollout(model, edit(h, h, u), w) - rollout(model, h, w)).abs().max().item()})
                    for kind, error in errors.items():
                        controls.add({'seed': seed, 'route': route, 'batch_first': first,
                                      'kind': kind, 'max_error': error})
                    controls.flush()
                    if max(errors.values()) > TOL:
                        raise RuntimeError('Editor numerical control failed')
                if route == 'natural_target':
                    histories = torch.tensor(np.asarray(data['banks']['test_donor']['histories'])[g[:, 6]])
                    cw = torch.cat([histories, torch.stack([torch.full_like(w, 8), w], dim=1)], dim=1)
                    wc = torch.cat([histories, torch.stack([w, torch.full_like(w, 8)], dim=1)], dim=1)
                    cw_logits, wc_logits = model(cw), model(wc)
                    expected = torch.stack([cw_logits[:, 11], cw_logits[:, 12], cw_logits[:, 13],
                                            wc_logits[:, 12], wc_logits[:, 13]], dim=1)
                    error = (result - expected).abs().max().item()
                    controls.add({'seed': seed, 'route': route, 'batch_first': first,
                                  'kind': 'full_donor_interface', 'max_error': error})
                    controls.flush()
                    if error > TOL:
                        raise RuntimeError('Full donor interface control failed')
            logits, norms = np.concatenate(chunks), np.concatenate(norms)
            target = factual if route.startswith('same_state_') else desired
            pred, correct, margin, probability = score_arrays(logits, target)
            contract = correct[:, [0, 2, 4]].all(axis=1)
            result = {'contracts': len(grid), 'contract_success': float(contract.mean()),
                      'mean_edit_norm': float(norms.mean()),
                      'immediate_A_accuracy': float((pred[:, 0] // 4 == target[:, 0] // 4).mean()),
                      'immediate_B_accuracy': float((pred[:, 0] % 4 == target[:, 0] % 4).mean())}
            for j, stage in enumerate(STAGES):
                result[stage + '_accuracy'] = float(correct[:, j].mean())
                result[stage + '_margin'] = float(margin[:, j].mean())
                result[stage + '_probability'] = float(probability[:, j].mean())
            result['donor_B_strata'] = {}
            for name, mask in [('same', grid[:, 3] == grid[:, 1] % 4),
                               ('different', grid[:, 3] != grid[:, 1] % 4)]:
                result['donor_B_strata'][name] = {'contracts': int(mask.sum()),
                                                'contract_success': float(contract[mask].mean())}
            totals[route] = result
            saved[route] = logits
            edit_norms[route] = norms
            for i in range(len(grid)):
                row = {'seed': seed, 'route': route, 'contract': i, 'receiver': int(grid[i, 0]),
                       'receiver_state': int(grid[i, 1]), 'new_A': int(grid[i, 2]),
                       'donor_B': int(grid[i, 3]), 'write_A': int(grid[i, 4]),
                       'edit_norm': float(norms[i]), 'contract_success': int(contract[i])}
                for j, stage in enumerate(STAGES):
                    row[stage + '_target'] = int(target[i, j])
                    row[stage + '_prediction'] = int(pred[i, j])
                    row[stage + '_margin'] = float(margin[i, j])
                    row[stage + '_probability'] = float(probability[i, j])
                row_table.add(row)
            row_table.flush()
            np.savez_compressed(out / f'logits-{seed}-{route}.npz', logits=logits, edit_norm=norms,
                                targets=target, contracts=grid)
        base_correct = saved['base'].argmax(-1) == factual
        natural_correct = saved['natural_target'].argmax(-1) == desired
        ability = {'factual_current': float(base_correct[:, 0].mean()),
                   'factual_first_steps': float(base_correct[:, [1, 3]].mean()),
                   'factual_final_steps': float(base_correct[:, [2, 4]].mean()),
                   'natural_target_all_steps': float(natural_correct.mean())}
        ability['passes'] = all(v >= 0.99 for v in ability.values())
    return {'seed': seed, 'ability': ability, 'routes': totals,
            'primary_difference': totals['transition']['contract_success'] - totals['immediate']['contract_success']}


def manifest(out):
    return {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}


def run(args):
    verify_lock(args.lock)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Refusing a nonempty output directory')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = json.loads((ROOT / 'inputs.json').read_text())
    inputs_check(data)
    metadata = {'status': 'running', 'lock_commit': args.lock, 'versions': versions(),
                'seeds': SEEDS, 'threads': 2, 'dtype': 'float64', 'inputs_sha256': sha(ROOT / 'inputs.json'),
                'plan_sha256': sha(ROOT / 'analysis-plan.md'), 'runner_sha256': sha(__file__)}
    dump(out / 'run-metadata.json', metadata)
    trace, controls, rows = Table(out / 'training.csv.gz'), Table(out / 'controls.csv'), Table(out / 'contracts.csv.gz')
    seed_results = []
    try:
        for seed in SEEDS:
            check_budget(started)
            torch.manual_seed(seed)
            model = Model()
            assert sum(p.numel() for p in model.parameters()) == 16400
            factual_seconds = train_model(model, seed, data, out, trace, started)
            with torch.inference_mode():
                hidden = {name: model.encode(torch.tensor(bank['histories']))[1].clone()
                          for name, bank in data['banks'].items()}
            # Ordinary tensors are needed for the edit gradient, while base weights stay fixed.
            hidden = {name: value.clone() for name, value in hidden.items()}
            torch.manual_seed(81000 + seed)
            initial = torch.randn(64, 4, dtype=torch.float64)
            bases = {'initial': basis(initial)}
            np.savez_compressed(out / f'editor-{seed}-initial.npz', W=initial.numpy(), U=bases['initial'].numpy())
            times = {'factual': factual_seconds}
            base_hash = {key: tensor.clone() for key, tensor in model.state_dict().items()}
            for arm in ('immediate', 'transition'):
                bases[arm], times[arm] = train_editor(model, seed, arm, initial, hidden, data, out, trace, started)
                if any(not torch.equal(value, model.state_dict()[key]) for key, value in base_hash.items()):
                    raise RuntimeError('Frozen model weights changed during edit training')
            evaluation_start = time.monotonic()
            result = evaluate(model, seed, hidden, bases, data, out, controls, rows, started)
            times['evaluation'] = time.monotonic() - evaluation_start
            result['seconds'] = times
            seed_results.append(result)
            dump(out / f'summary-{seed}.json', result)
            print(json.dumps({'completed_seed': seed, 'ability': result['ability'],
                              'primary_difference': result['primary_difference'], 'seconds': times}), flush=True)
        differences = np.array([r['primary_difference'] for r in seed_results])
        rng = np.random.default_rng(71208)
        means = differences[rng.integers(0, 4, (10000, 4))].mean(axis=1)
        summary = {'seeds': seed_results, 'primary_mean_difference': float(differences.mean()),
                   'primary_seed_bootstrap_interval': np.quantile(means, [.025, .975]).tolist(),
                   'positive_seeds': int((differences > 0).sum()),
                   'directional_support_rule_passes': bool(differences.mean() > 0 and (differences > 0).sum() >= 3),
                   'all_models_pass_ability': all(r['ability']['passes'] for r in seed_results)}
        dump(out / 'summary.json', summary)
        check_budget(started)
        verify_lock(args.lock)
        assert trace.count == 14000 and rows.count == 86016 and controls.count == 528
        metadata.update({'status': 'complete', 'elapsed_seconds': time.monotonic() - started,
                         'training_rows': trace.count, 'contract_rows': rows.count, 'control_rows': controls.count,
                         'factual_optimizer_updates': 6000, 'edit_optimizer_updates': 8000,
                         'factual_training_GRU_positions': 4608000,
                         'transition_training_extra_GRU_positions': 256000,
                         'immediate_arm_training_readouts': 256000,
                         'transition_arm_training_readouts': 512000,
                         'history_encoding_GRU_positions': 36864,
                         'test_and_control_GRU_positions': 884736,
                         'test_and_control_readouts': 1019904,
                         'test_contracts_per_route_per_model': 3072,
                         'saved_test_readouts': 430080,
                         'scientific_model_outputs_collected': True})
    except BaseException as error:
        metadata.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                         'elapsed_seconds': time.monotonic() - started, 'completed_seeds': len(seed_results)})
        raise
    finally:
        trace.close()
        controls.close()
        rows.close()
        dump(out / 'run-metadata.json', metadata)
        dump(out / 'manifest.json', manifest(out))


def preflight(args):
    data = json.loads((ROOT / 'inputs.json').read_text())
    inputs_check(data)
    # Independent scalar state-machine oracle.
    for a in range(4):
        for b in range(4):
            expected = [(v, b) for v in range(4)] + [(a, v) for v in range(4)] + [
                (a, a), (b, b), ((a + 1) % 4, b), (a, (b + 1) % 4), (b, a), (a, b)]
            assert transition(4 * a + b, np.arange(14)).tolist() == [4 * x + y for x, y in expected]
    grid = contract_grid(data)
    truth = target_grid(4 * grid[:, 2] + grid[:, 1] % 4, grid[:, 4])
    assert np.all(truth[:, 2] == 4 * grid[:, 4] + grid[:, 2])
    assert np.all(truth[:, 4] == 5 * grid[:, 4])
    # Dummy model and artificial histories only. No study history is passed to the model.
    torch.manual_seed(99811)
    model = Model()
    assert sum(p.numel() for p in model.parameters()) == 16400
    errors = []
    study_histories = {tuple(row) for row in data['factual_histories']}
    study_histories.update(tuple(row) for bank in data['banks'].values() for row in bank['histories'])
    for batch_size in (64, 256):
        indices = np.arange(batch_size)
        dummy = ((indices[:, None] // (14 ** (np.arange(12) % 3))) + np.arange(12)) % 14
        assert len({tuple(r) for r in dummy}) == batch_size
        assert not study_histories.intersection(tuple(row) for row in dummy)
        tokens = torch.tensor(dummy)
        full, h = model.encode(tokens)
        sequential = torch.zeros(batch_size, 64, dtype=torch.float64)
        for column in tokens.T:
            sequential = model.advance(sequential, column)
        u = basis(torch.randn(64, 4, dtype=torch.float64))
        write = torch.arange(batch_size) % 4
        donor = h.flip(0)
        suffix = torch.stack([torch.full_like(write, 8), write], dim=1)
        joined = torch.cat([tokens.flip(0), suffix], dim=1)
        interface = rollout(model, donor, write)
        expected = model(joined)[:, -2:]
        values = {'batch_size': batch_size,
                  'sequential': (h - sequential).abs().max().item(),
                  'self_hidden': (edit(h, h, u) - h).abs().max().item(),
                  'self_logits': (rollout(model, edit(h, h, u), write) - rollout(model, h, write)).abs().max().item(),
                  'full_donor_interface': (interface[:, 1:3] - expected).abs().max().item(),
                  'orthonormal': (u.T @ u - torch.eye(4, dtype=torch.float64)).abs().max().item()}
        errors.append(values)
        assert max(v for k, v in values.items() if k != 'batch_size') <= TOL
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight = nn.Parameter(torch.randn(64, 4, dtype=torch.float64))
    h, donor = h.detach(), donor.detach()
    changed = edit(h, donor, basis(weight))
    loss = F.cross_entropy(model.readout(model.advance(changed, write)), torch.arange(256) % 16)
    loss.backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())
    receipt = {'status': 'passed', 'scientific_model_outputs_collected': False,
               'model_training_performed': False, 'dummy_batch_sizes': [64, 256], 'errors': errors,
               'distinct_histories': 1792, 'test_contracts': len(grid), 'parameter_count': 16400,
               'versions': versions(), 'plan_sha256': sha(ROOT / 'analysis-plan.md'),
               'runner_sha256': sha(__file__), 'inputs_sha256': sha(ROOT / 'inputs.json'),
               'future_only_gradient_routes_to_editor_only': True,
               'dummy_histories_disjoint_from_study': True,
               'future_only_editor_gradient_norm': weight.grad.norm().item()}
    output = Path(args.out)
    if output.exists():
        raise RuntimeError('Refusing to overwrite a preflight receipt')
    dump(output, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'preflight', 'run'])
    parser.add_argument('--out')
    parser.add_argument('--lock')
    args = parser.parse_args()
    configure()
    if args.mode == 'prepare':
        path = ROOT / 'inputs.json'
        data = make_inputs()
        if path.exists():
            if json.loads(path.read_text()) != data:
                raise RuntimeError('Existing input artifact differs')
        else:
            dump(path, data, compact=True)
        print(json.dumps({'histories': 1792, 'contracts': 3072, 'inputs_sha256': sha(path)}))
    elif args.mode == 'preflight':
        if not args.out:
            parser.error('--out is required')
        preflight(args)
    else:
        if not args.out or not args.lock:
            parser.error('--out and --lock are required')
        run(args)


if __name__ == '__main__':
    main()
