#!/usr/bin/env python3
"""Independent saved-weight audit of the fixed post-failure vision follow-up.

This file does not import either scientific runner and does not repeat training.
The edit uses a ReLU output hook. The full saved output grid and every recorded
model control are replayed. A fresh receipt preserves any failed check.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STUDY = Path(__file__).resolve().parents[1]
LOCK = '782dc2c8d73a3fa9e67a03c8cbd394a5548d9bf9'
ORIGINAL_LOCK = '5d7a88d9e79d974f7e320359039e11e813d20b15'
SHIFTS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
FOUR = [(-4, 0), (4, 0), (0, -4), (0, 4)]
REUSED = [(0, 'direct'), (0, 'filtered'), (1, 'direct'),
          (1, 'filtered'), (2, 'direct')]
DATA_HASHES = {
    'optdigits.tra': 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
    'optdigits.tes': '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'}
LOGIT_TOL = 1e-10
SCORE_TOL = 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def locked(path, commit):
    repo = STUDY.parents[1]
    blob = subprocess.run(['git', 'show', commit + ':' + path.relative_to(repo).as_posix()],
                          cwd=repo, check=True, capture_output=True).stdout
    assert blob == path.read_bytes(), str(path)


def compare(got, saved, path='root', tolerance=SCORE_TOL):
    if isinstance(saved, dict):
        assert set(got) == set(saved), (path, set(got) ^ set(saved))
        for key in saved:
            compare(got[key], saved[key], path + '.' + key, tolerance)
    elif isinstance(saved, list):
        assert len(got) == len(saved), path
        for i, (a, b) in enumerate(zip(got, saved)):
            compare(a, b, path + f'[{i}]', tolerance)
    elif isinstance(saved, float):
        assert np.isfinite(got) and abs(got-saved) <= tolerance, (path, got, saved)
    else:
        assert got == saved, (path, got, saved)


class ReplayNet(nn.Module):
    """Same mathematical network, with the intervention in a separate hook."""
    def __init__(self, filtered):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, padding_mode='circular')
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(32, 10)
        self.register_buffer('kernel', torch.tensor([[[[1., 2., 1.],
                                                       [2., 4., 2.],
                                                       [1., 2., 1.]]]]) / 16)
        self.filtered = filtered
        self.channel = None
        self.local_errors = None
        self.relu1.register_forward_hook(self.edit)

    def edit(self, module, args, activation):
        if self.channel is None:
            return activation
        changed = activation.clone()
        if self.channel >= 0:
            changed[:, self.channel, :, :] = 0
            mask = torch.arange(8) != self.channel
            self.local_errors = (float(changed[:, self.channel].abs().max()),
                                 float((changed[:, mask]-activation[:, mask]).abs().max()))
        return changed

    def reduce(self, value):
        if self.filtered:
            value = F.conv2d(F.pad(value, (1, 1, 1, 1), mode='circular'),
                             self.kernel.repeat(value.shape[1], 1, 1, 1),
                             groups=value.shape[1])
        return value[:, :, 0::2, 0::2]

    def forward(self, x):
        x = self.reduce(self.relu1(self.conv1(x)))
        x = self.reduce(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        return self.fc(x.mean((-2, -1)))


def parameter_hash(model):
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value.update(name.encode())
        value.update(parameter.detach().numpy().tobytes())
    return value.hexdigest()


def shifted(x, dy, dx):
    # Explicit source indices, independent of the runner's torch.roll call.
    rows = (torch.arange(x.shape[-2])-dy) % x.shape[-2]
    columns = (torch.arange(x.shape[-1])-dx) % x.shape[-1]
    return x[:, :, rows[:, None], columns[None, :]]


def score(raw, labels):
    flat = raw.reshape(-1, 10)
    exponents = np.exp(flat-flat.max(1)[:, None])
    p = (exponents / exponents.sum(1)[:, None]).reshape(raw.shape)
    effect = np.stack([p[:, 0]-p[:, c+1] for c in range(8)], axis=1)
    errors = np.stack([np.abs(effect[s]-effect[0]).sum(-1)/2 for s in range(1, 5)])
    base = np.stack([np.abs(p[s, 0]-p[0, 0]).sum(-1)/2 for s in range(1, 5)])
    ablated = np.stack([np.abs(p[s, 1:]-p[0, 1:]).sum(-1)/2 for s in range(1, 5)])
    magnitudes = np.abs(effect).sum(-1)/2
    tops = raw[:, 0].argmax(-1)
    correct = tops == labels
    all_correct = np.logical_and.reduce(correct, axis=0)
    summary = dict(D=float(errors.mean()), baseline_shift_tv=float(base.mean()),
        ablated_shift_tv=float(ablated.mean()), triangle_bound=float((base[:, None]+ablated).mean()),
        effect_size_original=float(magnitudes[0].mean()), effect_size_all_shifts=float(magnitudes.mean()),
        accuracy_original=float(correct[0].mean()), accuracy_shifted=float(correct[1:].mean()),
        top1_consistency=float((tops[1:]==tops[0]).mean()), correct_all_shifts_n=int(all_correct.sum()),
        correct_all_shifts_D=float(errors[:, :, all_correct].mean()) if all_correct.any() else None,
        by_channel=[dict(channel=c, D=float(errors[:, c].mean()),
            effect_size_original=float(magnitudes[0, c].mean()),
            ablated_shift_tv=float(ablated[:, c].mean())) for c in range(8)],
        by_shift=[dict(dy=SHIFTS[s][0], dx=SHIFTS[s][1], D=float(errors[s-1].mean()),
            accuracy=float(correct[s].mean()), baseline_shift_tv=float(base[s-1].mean()),
            ablated_shift_tv=float(ablated[s-1].mean())) for s in range(1, 5)],
        by_digit=[dict(digit=d, n=int((labels==d).sum()), D=float(errors[:, :, labels==d].mean()),
            accuracy_original=float(correct[0, labels==d].mean())) for d in range(10)])
    return summary, (float(abs(p.sum(-1)-1).max()),
                     max(0., float((errors-base[:, None]-ablated).max())))


def audit(results, receipt):
    started = time.monotonic()
    receipt.update(status='running', script_sha256=digest(__file__), results=os.path.relpath(results, STUDY.parents[1]),
        model_logit_tolerance=LOGIT_TOL, score_tolerance=SCORE_TOL,
        training_repeated=False, runner_imported=False, lock_commit=LOCK)
    manifest = read_json(results/'manifest.json')
    assert manifest['status'] == 'complete'
    receipt['manifest_sha256'] = digest(results/'manifest.json')
    assert set(manifest['files']) == {p.relative_to(results).as_posix() for p in results.rglob('*')
                                      if p.is_file() and p.name != 'manifest.json'}
    for name, expected in manifest['files'].items():
        assert digest(results/name) == expected, name
    metadata = read_json(results/'metadata.json')
    assert metadata['lock_commit'] == LOCK
    assert metadata['training_dtype'] == 'float32' and metadata['evaluation_dtype'] == 'float64'
    for name in ['followup-plan.md', 'analysis-plan.md', 'code/run_float64.py', 'code/run.py', 'requirements.txt']:
        locked(STUDY/name, LOCK)
    assert metadata['plan_sha256'] == digest(STUDY/'followup-plan.md')
    assert metadata['runner_sha256'] == digest(STUDY/'code/run_float64.py')
    for name, expected in DATA_HASHES.items():
        assert digest(STUDY/'data'/name) == expected
        locked(STUDY/'data'/name, LOCK)
    raw = np.loadtxt(STUDY/'data/optdigits.tes', delimiter=',', dtype=np.int64)
    assert raw.shape == (1797, 65)
    labels = raw[:, -1]
    np.testing.assert_array_equal(labels, np.load(results/'test-labels.npy'))
    pixels = np.pad(raw[:, :64].reshape(-1, 1, 8, 8)/16., ((0, 0), (0, 0), (4, 4), (4, 4)))
    inputs = torch.from_numpy(pixels)
    original = STUDY/'results'
    old_manifest = read_json(original/'manifest.json')
    assert old_manifest['status'] == 'failed'
    for name, expected in old_manifest['files'].items():
        assert digest(original/name) == expected
    for path in [original/'manifest.json', original/'training.csv']:
        locked(path, ORIGINAL_LOCK)
    assert (original/'training.csv').read_bytes() == (results/'reused-training.csv').read_bytes()
    reuse = read_json(results/'reuse-verification.json')
    assert reuse['original_results_commit'] == ORIGINAL_LOCK
    assert reuse['original_manifest_sha256'] == digest(original/'manifest.json')
    assert reuse['original_training_sha256'] == digest(original/'training.csv')
    assert reuse['model_calls'] == 0
    for pair in REUSED:
        name = f'{pair[0]}-{pair[1]}.npz'
        locked(original/'models'/name, ORIGINAL_LOCK)
        assert (results/'models'/name).read_bytes() == (original/'models'/name).read_bytes()
        item = next(r for r in reuse['models'] if (r['seed'], r['architecture']) == pair)
        assert item['checkpoint_sha256'] == digest(original/'models'/name)
    for seed in range(8):
        rng = np.random.default_rng(10000+seed)
        orders, moves = [], []
        for epoch in range(10):
            orders.append(rng.permutation(3823))
            moves.append(rng.integers(-2, 3, (3823, 2), dtype=np.int8))
        saved = np.load(results/'schedules'/f'seed-{seed}.npz')
        np.testing.assert_array_equal(saved['orders'], orders)
        np.testing.assert_array_equal(saved['shifts'], moves)
        if seed < 3:
            path = original/'schedules'/f'seed-{seed}.npz'
            locked(path, ORIGINAL_LOCK)
            assert path.read_bytes() == (results/'schedules'/path.name).read_bytes()
            assert reuse['shared_schedule_sha256'][path.name] == digest(path)
    all_pairs = [(s, a) for s in range(8) for a in ('direct', 'filtered')]
    for filename, pairs in [('reused-training.csv', REUSED), ('training.csv', [p for p in all_pairs if p not in REUSED])]:
        rows = table(results/filename)
        assert len(rows) == 300*len(pairs)
        for index, row in enumerate(rows):
            pair = pairs[index//300]
            within = index % 300
            assert (int(row['seed']), row['architecture']) == pair
            assert (int(row['epoch']), int(row['batch'])) == (within//30, within % 30)
            assert int(row['global_update']) == index+1
            assert int(row['n']) == (111 if within % 30 == 29 else 128)
            assert np.isfinite(float(row['loss']))
        if filename == 'reused-training.csv':
            for i, pair in enumerate(pairs):
                encoded = json.dumps(rows[300*i:300*(i+1)], sort_keys=True, separators=(',', ':')).encode()
                item = next(r for r in reuse['models'] if (r['seed'], r['architecture']) == pair)
                assert item['training_rows'] == 300
                assert item['training_rows_sha256'] == hashlib.sha256(encoded).hexdigest()
    controls = table(results/'controls.csv')
    assert len(controls) == 45632
    control_map = {}
    for row in controls:
        key = (int(row['seed']), row['architecture'], row['shift'], int(row['batch']), int(row['channel']), row['name'])
        assert key not in control_map
        error, limit = float(row['error']), float(row['tolerance'])
        assert np.isfinite(error) and error <= limit and row['passed'] == 'True'
        expected = 0. if row['name'] in ('no_op_logits', 'ablated_channel_zero', 'untouched_channels') else (1e-12 if row['name'] in ('probability_row_sum', 'triangle_inequality_excess') else 1e-10)
        assert limit == expected
        control_map[key] = error
    visited = set()
    control_error = 0.
    def check(pair, move, batch, channel, name, value):
        nonlocal control_error
        key = (*pair, move, batch, channel, name)
        assert key not in visited
        visited.add(key)
        expected = control_map[key]
        control_error = max(control_error, abs(expected-value))
        tolerance = 0. if name in ('no_op_logits', 'ablated_channel_zero', 'untouched_channels') else (SCORE_TOL if name in ('probability_row_sum', 'triangle_inequality_excess') else LOGIT_TOL)
        assert abs(expected-value) <= tolerance, (key, value, expected)
    summaries = read_json(results/'model-summaries.json')
    assert len(summaries) == 16
    reconstructed = []
    calls = examples = saved_count = 0
    maximum = 0.
    receipt['models_replayed'] = []
    for pair, saved_summary in zip(all_pairs, summaries):
        seed, arch = pair
        torch.manual_seed(seed)
        model = ReplayNet(arch == 'filtered')
        initial_hash = parameter_hash(model)
        weights = dict(np.load(results/'models'/f'{seed}-{arch}.npz'))
        assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in weights.values())
        model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
        model.double().eval()
        saved = np.load(results/'logits'/f'{seed}-{arch}.npz')['logits']
        assert saved.shape == (5, 9, 1797, 10) and saved.dtype == np.float64 and np.isfinite(saved).all()
        def forward(x, channel):
            nonlocal calls, examples
            assert time.monotonic()-started <= 1200, 'Audit time cap exceeded'
            calls += 1
            examples += len(x)
            model.channel = channel
            return model(x)
        for si, (dy, dx) in enumerate(SHIFTS):
            move = f'{dy},{dx}'
            for start in range(0, 1797, 128):
                b = start//128
                source = inputs[start:start+128]
                x = shifted(source, dy, dx)
                base = forward(x, None)
                error = float(np.max(np.abs(base.numpy()-saved[si, 0, start:start+len(x)])))
                maximum = max(maximum, error)
                assert error <= LOGIT_TOL
                saved_count += base.numel()
                noop = forward(x, -1)
                check(pair, move, b, -1, 'no_op_logits', float(abs(noop-base).max()))
                model.channel = None
                features = model.relu1(model.conv1(x))
                expected = shifted(model.relu1(model.conv1(source)), dy, dx)
                check(pair, move, b, -1, 'first_features_shift', float(abs(features-expected).max()))
                for channel in range(8):
                    logits = forward(x, channel)
                    for name, value in zip(('ablated_channel_zero', 'untouched_channels'), model.local_errors):
                        check(pair, move, b, channel, name, value)
                    error = float(np.max(np.abs(logits.numpy()-saved[si, channel+1, start:start+len(x)])))
                    maximum = max(maximum, error)
                    assert error <= LOGIT_TOL
                    saved_count += logits.numel()
        for dy, dx in FOUR:
            move = f'{dy},{dx}'
            for start in range(0, 1797, 128):
                x = shifted(inputs[start:start+128], dy, dx)
                for condition in range(9):
                    channel = condition-1
                    logits = forward(x, None if channel == -1 else channel)
                    if channel >= 0:
                        for name, value in zip(('ablated_channel_zero', 'untouched_channels'), model.local_errors):
                            check(pair, move, start//128, channel, name, value)
                    error = float(np.max(np.abs(logits.numpy()-saved[0, condition, start:start+len(x)])))
                    check(pair, move, start//128, channel, 'four_pixel_logits', error)
        metrics, score_controls = score(saved, labels)
        for name, value in zip(('probability_row_sum', 'triangle_inequality_excess'), score_controls):
            check(pair, 'all', -1, -1, name, value)
        got = dict(seed=seed, architecture=arch, reused_training=pair in REUSED,
                   initial_parameter_sha256=initial_hash, **metrics)
        compare(got, saved_summary, f'model[{seed},{arch}]')
        reconstructed.append(got)
        receipt['models_replayed'].append(dict(seed=seed, architecture=arch))
        print(f'Audit complete: seed {seed}, {arch}', flush=True)
    assert visited == set(control_map)
    assert calls == 20640 and examples == 2472672 and saved_count == 12938400
    differences = np.array([reconstructed[2*s]['D']-reconstructed[2*s+1]['D'] for s in range(8)])
    indices = np.random.default_rng(2026101417).integers(0, 8, (10000, 8))
    interval = np.percentile(differences[indices].mean(1), [2.5, 97.5], method='linear')
    summary = read_json(results/'summary.json')
    compare(differences.tolist(), summary['paired_seed_differences'], 'paired differences')
    compare(float(differences.mean()), summary['mean'], 'mean')
    compare(interval.tolist(), summary['percentile_95_interval'], 'interval')
    for key, expected in dict(control_rows=45632, calls=23940, examples=2893202,
                              new_training_updates=3300, reused_training_updates=1500,
                              bootstrap_seed=2026101417, bootstrap_repeats=10000).items():
        assert summary[key] == expected
    recorded_maxima = {name: max(float(row['error']) for row in controls if row['name']==name)
                        for name in {row['name'] for row in controls}}
    compare(recorded_maxima, summary['control_maxima'], 'control maxima', 0.)
    assert digest(results/'manifest.json') == receipt['manifest_sha256']
    for name, expected in manifest['files'].items():
        assert digest(results/name) == expected
    receipt.update(status='passed', full_model_calls=calls, forwarded_examples=examples,
        saved_logit_values_checked=saved_count, maximum_saved_logit_error=maximum,
        controls_replayed=len(visited), maximum_control_record_difference=control_error,
        new_training_records_checked=3300, reused_training_records_checked=1500,
        reused_checkpoints_verified=5, schedules_checked=8, model_summaries_checked=16,
        all_scores_and_seed_bootstrap_match=True, elapsed_seconds=time.monotonic()-started,
        scope='All saved logits and all control conditions replayed with a separate ReLU-hook model. All score, schedule, training-record, and reuse checks passed. Training was not repeated.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), 'Use a fresh receipt path.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    receipt = {}
    try:
        with torch.no_grad():
            audit(args.results.resolve(), receipt)
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
