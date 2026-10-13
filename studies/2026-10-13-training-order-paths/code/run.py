#!/usr/bin/env python3
"""Fixed training-order/path experiment; study runs require a public code lock."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import itertools
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1103, 2207, 3301, 4409, 5519, 6607, 7717, 8803]
SCHEDULES = ('AB', 'BA', 'ALT')
TAILS = (0, 200, 2000)
TOL = 1e-10
CAP_SECONDS = 1200
PACKAGES = {'numpy': '2.3.5', 'torch': '2.8.0'}
MASKS = [[0,6,14,15,17,20,23,25], [10,13,16,20,21,23,25,28],
         [3,6,8,11,17,19,20,26], [2,7,10,11,14,22,30,31],
         [3,8,14,20,22,25,27,30], [0,2,4,11,15,16,20,31],
         [1,2,3,4,5,19,27,28], [0,6,7,11,14,17,23,30]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + '\n')


def verify_lock(commit):
    repo = ROOT.parents[1]
    resolved = subprocess.run(['git', 'rev-parse', commit + '^{commit}'], cwd=repo,
                              check=True, capture_output=True).stdout.decode().strip()
    if resolved != commit:
        raise RuntimeError('The lock must be a full immutable commit SHA.')
    for path in (ROOT / 'analysis-plan.md', Path(__file__).resolve()):
        rel = path.relative_to(repo).as_posix()
        locked = subprocess.run(['git', 'show', commit + ':' + rel], cwd=repo,
                                check=True, capture_output=True).stdout
        if locked != path.read_bytes():
            raise RuntimeError('Local bytes differ from lock: ' + rel)


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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(5, 32, dtype=torch.float64)
        self.second = nn.Linear(32, 32, dtype=torch.float64)
        self.last = nn.Linear(32, 1, dtype=torch.float64)
        self.gelu = nn.GELU(approximate='none')

    def after_first(self, a):
        return self.last(self.gelu(self.second(self.gelu(a)))).squeeze(-1)

    def forward(self, x):
        return self.after_first(self.first(x))


def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                           dampening=0, weight_decay=0, nesterov=False)


def data():
    x = torch.tensor(list(itertools.product((-1., 1.), repeat=5)), dtype=torch.float64)
    y = x[:, 0] * x[:, 1]
    batches = {'A': torch.where(x[:, 2] == y)[0], 'B': torch.where(x[:, 2] != y)[0],
               'T': torch.arange(32)}
    splits = np.where((x[:, 3] == x[:, 4]).numpy(), 'calibration', 'test')
    donor = x.clone()
    donor[:, 2] *= -1
    donor_ids = [next(i for i, candidate in enumerate(x) if torch.equal(candidate, row)) for row in donor]
    for ids in batches.values():
        assert int((y[ids] == 1).sum()) == len(ids) // 2
    assert all(len(batches[b]) == 16 for b in ('A', 'B'))
    assert sum(splits == 'calibration') == sum(splits == 'test') == 16
    return x, y, donor, donor_ids, batches, splits


def schedules():
    return {'AB': ['A'] * 500 + ['B'] * 500,
            'BA': ['B'] * 500 + ['A'] * 500, 'ALT': ['A', 'B'] * 500}


def maxerr(a, b):
    return float((a - b).abs().max())


def norm(arrays):
    return float(torch.sqrt(sum((v * v).sum() for v in arrays)))


class Experiment:
    def __init__(self, output, lock):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'checkpoints').mkdir()
        self.lock = lock
        self.started = time.monotonic()
        self.updates = 0
        self.model = self.opt = None
        self.context = {}
        self.tables = {name: Table(self.output / path) for name, path in {
            'train': 'training.csv.gz', 'intact': 'intact.csv',
            'interventions': 'interventions.csv.gz', 'sets': 'set-means.csv',
            'selection': 'selection.csv', 'controls': 'controls.csv',
            'models': 'model-metrics.csv'}.items()}
        self.checkpoints = []
        self.model_rows = []
        self.selected = {}
        self.max_control_error = 0.

    def budget(self):
        if time.monotonic() - self.started > CAP_SECONDS:
            raise TimeoutError('The fixed 20-minute budget was reached.')

    def control(self, name, error=0., tolerance=TOL, **tags):
        error = float(error)
        self.max_control_error = max(error, self.max_control_error)
        passed = bool(np.isfinite(error) and error <= tolerance)
        self.tables['controls'].add(dict(**tags, name=name, error=error,
                                         tolerance=tolerance, passed=passed))
        self.tables['controls'].flush()
        if not passed:
            raise AssertionError(f'Control failed: {name}, error={error}, tolerance={tolerance}')

    def save(self, label, model, opt=None, **tags):
        values = {'weight.' + k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
        if opt is not None:
            for key, parameter in model.named_parameters():
                buffer = opt.state[parameter].get('momentum_buffer')
                if buffer is not None:
                    values['momentum.' + key] = buffer.detach().numpy().copy()
        path = self.output / 'checkpoints' / (label + '.npz')
        np.savez_compressed(path, **values)
        self.checkpoints.append(dict(path=path.relative_to(self.output).as_posix(),
                                     sha256=sha(path), **tags))
        dump(self.output / 'checkpoints.json', self.checkpoints)

    def train_step(self, x, y, ids, **tags):
        if self.updates % 10 == 0:
            self.budget()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(x[ids])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, (y[ids] + 1) / 2)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss.')
        loss.backward()
        self.opt.step()
        self.updates += 1
        self.tables['train'].add(dict(**tags, pre_update_loss=float(loss.detach())))
        if self.updates % 10 == 0:
            self.tables['train'].flush()

    @torch.no_grad()
    def evaluate(self, x, y, donor, donor_ids, splits, **tags):
        self.budget()
        model = self.model.eval()
        a, donor_a = model.first(x), model.first(donor)
        logits, donor_logits = model(x), model(donor)
        q = torch.sigmoid(y * logits)
        orientation = -(x[:, 2] * y)
        calibration = torch.tensor(splits == 'calibration')
        common = dict(seed=tags['seed'], schedule=tags['schedule'], condition=tags['condition'], tail=tags['tail'])
        self.control('self_logit', maxerr(model.after_first(a.clone()), logits), **common)
        self.control('full_input_logit', maxerr(model.after_first(donor_a), donor_logits), **common)
        analytic_delta = -2 * x[:, 2, None] * model.first.weight[:, 2]
        self.control('z_edge_delta', maxerr(donor_a - a, analytic_delta), **common)
        singleton_logits, scores = [], []
        for unit in range(32):
            patched_a = a.clone()
            patched_a[:, unit] = donor_a[:, unit]
            l = model.after_first(patched_a)
            singleton_logits.append(l)
            scores.append(float((torch.sigmoid(y * l) - q)[calibration].abs().mean()))
        ranked = sorted(range(32), key=lambda i: (-scores[i], i))
        chosen = sorted(ranked[:8])
        for unit in range(32):
            self.tables['selection'].add(dict(**common, unit=unit, calibration_score=scores[unit],
                                               selected=unit in chosen, rank=ranked.index(unit) + 1))
        sets = [('unit', str(i), [i]) for i in range(32)]
        sets += [('selected', 'selected', chosen), ('complement', 'complement', [i for i in range(32) if i not in chosen]),
                 ('full', 'full', list(range(32)))]
        sets += [('random', str(i), mask) for i, mask in enumerate(MASKS)]
        for kind, set_id, mask in sets:
            patched_a = a.clone()
            patched_a[:, mask] = donor_a[:, mask]
            edge_a = a.clone()
            edge_a[:, mask] += analytic_delta[:, mask]
            self.control('set_edge_vector_' + kind + '_' + set_id, maxerr(patched_a, edge_a), **common)
            l = singleton_logits[int(set_id)] if kind == 'unit' else model.after_first(patched_a)
            restored = patched_a.clone()
            restored[:, mask] = a[:, mask]
            self.control('restore_vector_' + kind + '_' + set_id, maxerr(restored, a), tolerance=0., **common)
            self.control('restore_logit_' + kind + '_' + set_id, maxerr(model.after_first(restored), logits), **common)
            probability_change = torch.sigmoid(y * l) - q
            margin_change = y * (l - logits)
            oriented = orientation * probability_change
            oriented_margin = orientation * margin_change
            for i in range(32):
                self.tables['interventions'].add(dict(**common, receiver=i, kind=kind, set_id=set_id,
                    patched_logit=float(l[i]), probability_change=float(probability_change[i]),
                    oriented_probability=float(oriented[i]), margin_change=float(margin_change[i]),
                    oriented_margin=float(oriented_margin[i])))
            for split in ('calibration', 'test'):
                select = torch.tensor(splits == split)
                row = dict(**common, split=split, kind=kind, set_id=set_id,
                           mean_oriented_probability=float(oriented[select].mean()),
                           mean_absolute_probability_change=float(probability_change[select].abs().mean()),
                           mean_oriented_margin=float(oriented_margin[select].mean()))
                self.tables['sets'].add(row)
                if kind == 'selected' and split == 'test':
                    self.selected[(common['seed'], common['schedule'], common['condition'], common['tail'])] = row
        changed_x1 = x.clone()
        changed_x1[:, 0] *= -1
        changed_y = changed_x1[:, 0] * changed_x1[:, 1]
        self.control('positive_control_oracle', maxerr(changed_y, -y), tolerance=0., **common)
        flip_logits = model(changed_x1)
        for i in range(32):
            self.tables['intact'].add(dict(**common, receiver=i, split=str(splits[i]), donor=donor_ids[i],
                y=int(y[i]), z=int(x[i, 2]), logit=float(logits[i]), correct_probability=float(q[i]),
                correct=bool((logits[i] > 0) == (y[i] > 0)), margin=float(y[i] * logits[i]),
                donor_logit=float(donor_logits[i]), x1_flip_logit=float(flip_logits[i]),
                x1_flip_correct=bool((flip_logits[i] > 0) == (changed_y[i] > 0))))
        buffers = [state['momentum_buffer'] for state in self.opt.state.values() if 'momentum_buffer' in state]
        metric = dict(**common, accuracy=float(((logits > 0) == (y > 0)).double().mean()),
                      cross_entropy=float(nn.functional.binary_cross_entropy_with_logits(logits, (y + 1) / 2)),
                      mean_correct_margin=float((y * logits).mean()), weight_norm=norm(model.parameters()),
                      momentum_norm=norm(buffers), x1_flip_accuracy=float(((flip_logits > 0) == (changed_y > 0)).double().mean()))
        self.model_rows.append(metric)
        self.tables['models'].add(metric)
        for table in self.tables.values():
            table.flush()
        model.train()

    def run(self):
        verify_lock(self.lock)
        x, y, donor, donor_ids, batches, splits = data()
        sequences = schedules()
        batch_lists = {key: value.tolist() for key, value in batches.items()}
        counts = {}
        for name, sequence in sequences.items():
            exposure = np.zeros(32, dtype=np.int64)
            for key in sequence:
                exposure[batch_lists[key]] += 1
            assert np.all(exposure == 500)
            counts[name] = exposure.tolist()
        dump(self.output / 'inputs.json', [dict(receiver=i, input=x[i].tolist(), y=int(y[i]), donor=donor_ids[i], split=str(splits[i])) for i in range(32)])
        dump(self.output / 'batch-sequences.json', dict(prefix=sequences, tail=['T'] * 2000, batches=batch_lists, prefix_exposures=counts))
        dump(self.output / 'fixed-random-masks.json', MASKS)
        for seed in SEEDS:
            torch.manual_seed(seed)
            initial = Net()
            assert sum(p.numel() for p in initial.parameters()) == 1281
            self.save(f'{seed}-initial', initial, seed=seed, stage='initial')
            for schedule in SCHEDULES:
                self.model = copy.deepcopy(initial)
                self.opt = optimizer(self.model)
                self.context = dict(seed=seed, schedule=schedule, condition='prefix', tail=-1)
                self.control('initial_copy', max(maxerr(v, initial.state_dict()[k]) for k, v in self.model.state_dict().items()), tolerance=0., **self.context)
                for step, key in enumerate(sequences[schedule], 1):
                    self.context['step'] = step
                    self.train_step(x, y, batches[key], seed=seed, schedule=schedule, condition='prefix', stage='prefix', step=step, batch=key)
                    if step == 500:
                        self.save(f'{seed}-{schedule}-prefix500', self.model, self.opt, seed=seed, schedule=schedule, stage='prefix', step=500)
                prefix_weights = copy.deepcopy(self.model.state_dict())
                prefix_optimizer = copy.deepcopy(self.opt.state_dict())
                for condition in ('keep', 'reset'):
                    self.model = copy.deepcopy(initial)
                    self.model.load_state_dict(prefix_weights)
                    self.opt = optimizer(self.model)
                    self.opt.load_state_dict(copy.deepcopy(prefix_optimizer))
                    self.context = dict(seed=seed, schedule=schedule, condition=condition, tail=0)
                    self.control('fork_weights', max(maxerr(v, prefix_weights[k]) for k, v in self.model.state_dict().items()), tolerance=0., **self.context)
                    old_buffers = [s['momentum_buffer'] for s in prefix_optimizer['state'].values()]
                    copied_buffers = [s['momentum_buffer'] for s in self.opt.state.values()]
                    self.control('fork_momentum_copy', max(maxerr(v, w) for v, w in zip(copied_buffers, old_buffers)), tolerance=0., **self.context)
                    if condition == 'reset':
                        for state in self.opt.state.values():
                            state['momentum_buffer'].zero_()
                        self.control('reset_momentum_zero', max(float(s['momentum_buffer'].abs().max()) for s in self.opt.state.values()), tolerance=0., **self.context)
                    self.control('post_reset_weights', max(maxerr(v, prefix_weights[k]) for k, v in self.model.state_dict().items()), tolerance=0., **self.context)
                    for tail in range(2001):
                        self.context['tail'] = tail
                        if tail > 0:
                            self.train_step(x, y, batches['T'], seed=seed, schedule=schedule, condition=condition, stage='tail', step=tail, batch='T')
                        if tail in TAILS:
                            self.save(f'{seed}-{schedule}-{condition}-tail{tail}', self.model, self.opt, **self.context)
                            self.evaluate(x, y, donor, donor_ids, splits, **self.context)
                print(f'Completed seed {seed}, schedule {schedule}; updates={self.updates}', flush=True)
        assert self.updates == 120000
        assert len(self.checkpoints) == 176 and len(self.model_rows) == 144
        assert self.tables['intact'].count == 4608
        assert self.tables['interventions'].count == 198144
        assert self.tables['sets'].count == 12384
        assert self.tables['selection'].count == 4608
        verify_lock(self.lock)
        self.budget()
        self.summarize()
        self.budget()

    def summarize(self):
        pairs = []
        for seed in SEEDS:
            get = lambda schedule, condition: self.selected[(seed, schedule, condition, 2000)]['mean_oriented_probability']
            reset = get('BA', 'reset') - get('AB', 'reset')
            keep = get('BA', 'keep') - get('AB', 'keep')
            pairs.append(dict(seed=seed, reset_BA_minus_AB=reset, keep_BA_minus_AB=keep,
                              reset_minus_keep_contrast=reset - keep,
                              reset_ALT=get('ALT', 'reset'), keep_ALT=get('ALT', 'keep')))
        values = np.array([p['reset_BA_minus_AB'] for p in pairs])
        draws = np.random.default_rng(161803).integers(0, 8, size=(10000, 8))
        interval = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], method='linear')
        summary = dict(primary=dict(name='reset selected-eight test effect: BA minus AB at tail 2000',
            mean=float(values.mean()), median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max()),
            descriptive_paired_seed_bootstrap_95=interval.tolist(), numerical_negligibility_threshold=1e-8,
            numerically_negligible=bool(abs(values.mean()) < 1e-8)), per_seed=pairs,
            max_control_error=self.max_control_error, completed_updates=self.updates,
            model_checkpoints=len(self.checkpoints), evaluated_checkpoints=len(self.model_rows),
            table_rows={name: table.count for name, table in self.tables.items()})
        dump(self.output / 'summary.json', summary)

    def finish(self, status, error=None):
        for table in self.tables.values():
            table.close()
        receipt = dict(status=status, error=error, lock_commit=self.lock, seconds=time.monotonic() - self.started,
            completed_updates=self.updates, last_context=self.context, seeds=SEEDS, cpu_threads=torch.get_num_threads(),
            dtype='float64 training and evaluation', platform=platform.platform(), python=platform.python_version(),
            packages={name: version(name) for name in ('torch', 'numpy')},
            plan_sha256=sha(ROOT / 'analysis-plan.md'), runner_sha256=sha(__file__),
            max_control_error=self.max_control_error, table_rows={name: table.count for name, table in self.tables.items()})
        dump(self.output / 'run-receipt.json', receipt)
        files = {p.relative_to(self.output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
                 for p in sorted(self.output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        dump(self.output / 'manifest.json', files)


@torch.no_grad()
def preflight(output):
    """No training, no study seed, and no study-model output evaluation."""
    path = Path(output)
    if path.exists():
        raise FileExistsError('Preflight output exists.')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(987654)
    model = Net()
    x = torch.tensor([[0.25, -0.7, 0.4, 0.8, -0.2], [-0.8, 0.2, -0.6, 0.1, 0.9]], dtype=torch.float64)
    donor = x.clone()
    donor[:, 2] *= -1
    a, da = model.first(x), model.first(donor)
    patched = a.clone()
    patched[:, MASKS[0]] = da[:, MASKS[0]]
    restored = patched.clone()
    restored[:, MASKS[0]] = a[:, MASKS[0]]
    errors = dict(self=maxerr(model.after_first(a), model(x)), full=maxerr(model.after_first(da), model(donor)),
                  restore_vector=maxerr(restored, a), restore_logits=maxerr(model.after_first(restored), model(x)),
                  edge_formula=maxerr(da - a, -2 * x[:, 2, None] * model.first.weight[:, 2]))
    assert max(errors.values()) <= TOL
    study_x, study_y, _, _, batches, splits = data()
    exposures = {}
    for name, sequence in schedules().items():
        count = np.zeros(32, dtype=int)
        for key in sequence:
            count[batches[key].numpy()] += 1
        assert np.all(count == 500)
        exposures[name] = count.tolist()
    dump(path, dict(status='passed', scope='dummy algebra only; study data counts only; no training or scientific evaluation',
                    errors=errors, tolerance=TOL, data_count=len(study_x), labels_positive=int((study_y > 0).sum()),
                    split_counts={s: int(sum(splits == s)) for s in ('calibration', 'test')}, prefix_exposures=exposures,
                    params=sum(p.numel() for p in model.parameters()), runner_sha256=sha(__file__), plan_sha256=sha(ROOT / 'analysis-plan.md')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--lock-commit')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if {name: version(name) for name in PACKAGES} != PACKAGES:
        raise RuntimeError('Package versions differ from fixed requirements.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.preflight:
        preflight(args.output)
        return
    if not args.lock_commit:
        parser.error('Scientific runs require --lock-commit.')
    verify_lock(args.lock_commit)
    exp = Experiment(args.output, args.lock_commit)
    try:
        exp.run()
    except BaseException as error:
        if exp.model is not None:
            exp.save('incomplete-last-state', exp.model, exp.opt, **exp.context)
        exp.finish('incomplete', type(error).__name__ + ': ' + str(error))
        raise
    exp.finish('complete')


if __name__ == '__main__':
    main()
