"""Audit saved Experiment 2C records without model forwards or training.

This program imports no study code and no Torch module. It checks saved data,
arithmetic, choice rules, gates, and file hashes. Run only after science stops.
"""
from __future__ import annotations
import time
PROCESS_STARTED = time.monotonic()
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import tempfile
import zipfile

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
# NumPy is imported in main, only after the lock and live checks are active.

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
SEEDS = tuple(range(91001, 91009))
MASKS = tuple(itertools.combinations(range(12), 3))
STEPS = (1000, 2000, 3000, 4000)
TOL = 1e-10
SHAPES = {'W1': (16, 12), 'b1': (12,), 'W2': (12, 24), 'b2': (24,), 'W3': (24, 4), 'b3': (4,)}
REPORT_METRICS = ('active_changed_accuracy', 'inactive_changed_accuracy', 'all_changed_accuracy',
                  'active_changed_tv', 'inactive_changed_tv', 'same_value_answer_change')


class Audit:
    def __init__(self, seconds):
        # Include imports and argument parsing. Keep two seconds for the final
        # audit receipt within the 49-second run allocation. One further second
        # of the parent's 50-second allocation covers pre-run syntax checks.
        self.started = PROCESS_STARTED; self.deadline = self.started + seconds - 2
        self.total_deadline = self.started + seconds
        self.checks = 0; self.raw_files = 0; self.max_error = 0.; self.hashed = 0
        self.busy = False; self.last_disk = -math.inf

    def guard(self):
        if time.monotonic() >= self.deadline: raise TimeoutError('Saved-result audit time limit reached')
        try:
            permit = json.loads(Path(os.environ['MI_USAGE_PERMIT']).read_text())
            expiry = permit['expires_at']; now = time.time()
            valid = (permit['allowed'] is True and type(expiry) in (int, float)
                     and math.isfinite(expiry) and now < expiry <= now + 120)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            raise RuntimeError('Usage permit missing or invalid') from None
        if not valid: raise RuntimeError('Usage permit denied or expired')
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin': memory *= 1024
        if memory > 4 * 1024**3: raise RuntimeError('Peak memory exceeds 4 GiB')
        if time.monotonic() - self.last_disk >= 5:
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free disk')
            self.last_disk = time.monotonic()

    def alarm(self, *_):
        if self.busy: return
        self.busy = True
        try: self.guard()
        finally: self.busy = False

    def require(self, condition, message):
        self.checks += 1
        if not condition: raise AssertionError(message)

    def near(self, a, b, message):
        a, b = np.asarray(a), np.asarray(b)
        self.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message + ': shape/finite')
        error = float(np.max(np.abs(a - b))) if a.size else 0.
        self.max_error = max(self.max_error, error)
        self.require(error <= TOL, message)

    def digest(self, path):
        self.guard(); h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                self.guard(); h.update(block)
        self.hashed += 1
        return h.hexdigest()

    def load(self, path):
        self.guard()
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        for key, value in data.items():
            if np.issubdtype(value.dtype, np.number):
                self.require(np.isfinite(value).all(), 'Nonfinite saved array: ' + key)
        return data

    def check_manifest(self, folder, name='manifest.json', inner=False, allowed_extra=()):
        fixed = json.loads((folder / name).read_text())
        if inner: fixed = fixed['files']
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()
                  and p.name != name and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
        self.require(actual - set(allowed_extra) == set(fixed), 'Manifest file membership differs: ' + str(folder.relative_to(STUDY)))
        for relative, expected in fixed.items():
            path = Path(relative)
            self.require(not path.is_absolute() and '..' not in path.parts, 'Unsafe manifest path')
            self.require(self.digest(folder / relative) == expected, 'Manifest hash differs: ' + relative)
        return fixed


def read_json(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open(newline='') as stream: return list(csv.DictReader(stream))


def parse_metric(row, key):
    value = row[key]
    if key in ('eligible', 'absolute_gate_pass'):
        if type(value) is bool: return value
        if value not in ('True', 'False'): raise AssertionError('Invalid saved boolean')
        return value == 'True'
    return int(value) if key.endswith('_n') else float(value)


def gate(metric):
    return bool(metric['active_changed_accuracy'] >= .95 and metric['all_changed_accuracy'] >= .95
                and metric['active_changed_tv'] <= .05 and metric['inactive_changed_tv'] <= .05
                and metric['same_value_answer_change'] <= .01)


def eligible(metric):
    return bool(metric['inactive_changed_tv'] <= .05 and metric['same_value_answer_change'] <= .01)


def reported_metrics(metric):
    return {name: metric[name] for name in REPORT_METRICS}


def descriptive_means(rows):
    return {name: math.fsum(row[name] for row in rows) / len(rows) if rows else None
            for name in REPORT_METRICS}


def audit_data(audit, path):
    data = audit.load(path)
    states = np.array(list(itertools.product(range(4), range(4), range(2), range(6))), dtype=np.int64)
    x = np.concatenate([np.eye(size, dtype=np.float64)[states[:, i]] for i, size in enumerate((4, 4, 2, 6))], axis=1)
    labels = np.where(states[:, 2] == 0, states[:, 0], states[:, 1])
    r = np.repeat(np.arange(192), 192); d = np.tile(np.arange(192), 192)
    cf = ((states[d, 0] * 4 + states[r, 1]) * 2 + states[r, 2]) * 6 + states[r, 3]
    lookup = {}
    for s, same in itertools.product((0, 1), (False, True)):
        blocks = [(ri, value) for ri in range(192) for value in range(4)
                  if states[ri, 2] == s and (states[ri, 0] == value) == same]
        blocks = sorted(blocks, key=lambda pair: hashlib.sha256(f'selector-specificity-v1:{pair[0]}:{pair[1]}'.encode()).digest())
        for position, pair in enumerate(blocks):
            lookup[pair] = 0 if position * 2 < len(blocks) else (1 if position * 4 < len(blocks) * 3 else 2)
    split = np.array([lookup[int(ri), int(states[di, 0])] for ri, di in zip(r, d)])
    expected = dict(states=states, x=x, labels=labels, r=r, d=d, cf=cf, split=split,
                    active=states[r, 2] == 0, same_value=states[r, 0] == states[d, 0])
    audit.require(set(data) == set(expected), 'Data array names differ')
    for key, value in expected.items(): audit.require(np.array_equal(data[key], value), 'Data ledger differs: ' + key)
    audit.require([int(np.sum(split == part)) for part in range(3)] == [18432, 9216, 9216], 'Wrong split counts')
    return data


def probabilities(audit, logits, saved):
    audit.require(logits.shape == saved.shape and logits.ndim == 2 and logits.shape[1] == 4, 'Wrong output shape')
    scaled = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    denominator = np.array([math.fsum(row) for row in scaled])
    audit.near(saved, scaled / denominator[:, None], 'Saved logits and probabilities disagree')
    audit.require(np.all(saved >= 0) and np.all(saved <= 1), 'Probability outside zero to one')


def metric_values(prob, ids, data, intact):
    active = data['active'][ids].astype(bool); same = data['same_value'][ids].astype(bool)
    r, cf = data['r'][ids], data['cf'][ids]
    delta = np.abs(prob - intact[cf]); receiver_delta = np.abs(prob - intact[r])
    # Explicit four-coordinate grouping and scalar mean reductions differ from
    # the scientific implementation's row sums and group means.
    tv = ((delta[:, 0] + delta[:, 1]) + (delta[:, 2] + delta[:, 3])) * .5
    rtv = ((receiver_delta[:, 0] + receiver_delta[:, 1]) + (receiver_delta[:, 2] + receiver_delta[:, 3])) * .5
    answer = prob.argmax(1); correct = answer == data['labels'][cf]; change = answer != intact[r].argmax(1)
    result = {}
    for name, group in (('active_changed', active & ~same), ('inactive_changed', ~active & ~same),
                        ('all_changed', ~same), ('same_value', same)):
        n = int(group.sum())
        if n == 0: raise AssertionError('An expected metric group is empty')
        result.update({name + '_n': n, name + '_accuracy': int(np.count_nonzero(correct[group])) / n,
                       name + '_tv': math.fsum(tv[group]) / n, name + '_receiver_tv': math.fsum(rtv[group]) / n,
                       name + '_answer_change': int(np.count_nonzero(change[group])) / n})
    result['eligible'] = eligible(result); result['absolute_gate_pass'] = gate(result)
    return result


def compare_metrics(audit, computed, recorded):
    for key, expected in computed.items():
        observed = parse_metric(recorded, key)
        if type(expected) in (bool, int): audit.require(observed == expected, 'Metric or gate differs: ' + key)
        else: audit.near(observed, expected, 'Metric differs: ' + key)
    for name in ('maximum_replay_error', 'maximum_metric_replay_error'):
        if name in recorded: audit.require(0 <= float(recorded[name]) <= TOL, 'Recorded replay exceeds tolerance')


def raw_metrics(audit, path, expected_ids, data, intact, recorded):
    saved = audit.load(path); audit.raw_files += 1
    audit.require(set(saved) == {'pair_ids', 'logits', 'probabilities'}, 'Wrong raw-result keys')
    audit.require(np.array_equal(saved['pair_ids'], expected_ids), 'Raw result uses wrong pair set')
    audit.require(saved['logits'].dtype == np.float64 and saved['probabilities'].dtype == np.float64, 'Raw precision differs')
    probabilities(audit, saved['logits'], saved['probabilities'])
    computed = metric_values(saved['probabilities'], expected_ids, data, intact)
    compare_metrics(audit, computed, recorded)
    return computed, saved['probabilities']


def audit_models(audit, data):
    folder = STUDY / 'training/models'; summary = read_json(folder / 'summary.json')
    audit.require(summary['passed'] is True and summary['status'] == 'complete', 'Model calculation panel is incomplete')
    audit.require(summary['seeds'] == list(SEEDS) and len(summary['models']) == 8, 'Eight planned models are not present')
    audit.require([row['seed'] for row in summary['models']] == list(SEEDS), 'Model seeds or order differ')
    intact, results = {}, []
    for seed, recorded in zip(SEEDS, summary['models']):
        audit.guard(); directory = folder / f'seed-{seed}'
        weights = audit.load(directory / 'final.npz')
        audit.require(set(weights) == set(SHAPES), 'Model weight keys differ')
        for key, shape in SHAPES.items():
            audit.require(weights[key].shape == shape and weights[key].dtype == np.float64, 'Model architecture differs')
        saved = audit.load(directory / 'evaluation.npz')
        audit.require(np.array_equal(saved['labels'], data['labels']), 'Ordinary labels differ')
        probabilities(audit, saved['logits'], saved['probabilities'])
        accuracy = int(np.count_nonzero(saved['probabilities'].argmax(1) == data['labels'])) / 192
        local = read_json(directory / 'summary.json')
        audit.require(recorded == local, 'Model summaries differ')
        audit.near(local['ordinary_accuracy'], accuracy, 'Ordinary accuracy differs')
        audit.require(local['updates'] == 4000 and local['status'] == 'complete', 'Base training is incomplete')
        audit.require(local['ability_gate_pass'] == (accuracy >= .99), 'Ability decision differs')
        audit.require(local['max_scalar_replay_error'] <= TOL, 'Base replay did not pass')
        trace = table(directory / 'training.csv')
        audit.require([int(row['step']) for row in trace] == list(range(1, 4001)), 'Base update trace is incomplete')
        audit.require(all(math.isfinite(float(row['loss_before_update'])) for row in trace), 'Nonfinite base trace')
        optimizer = read_json(directory / 'optimizer-final.json')
        audit.require(optimizer['completed_updates'] == 4000 and optimizer['automatic_resume_allowed'] is False, 'Wrong final optimizer boundary')
        intact[seed] = saved['probabilities']; results.append(dict(seed=seed, accuracy=accuracy, passed=accuracy >= .99))
    audit.require(summary['ability_gate_pass'] == all(row['passed'] for row in results), 'Full-panel ability gate differs')
    return intact, results, summary['ability_gate_pass']


def audit_calibration(audit, data, intact):
    folder = STUDY / 'calibration/edits'; choices = read_json(folder / 'choices.json')
    training = read_json(folder / 'training.json')
    audit.require([row['seed'] for row in training] == list(SEEDS), 'Rotation training seed panel differs')
    for row in training:
        audit.require(row['updates'] == 4000 and row['initialization_seed'] == row['seed'] + 1000
                      and row['minibatch_seed'] == row['seed'] + 2000, 'Rotation training schedule or seeds differ')
    audit.require(set(choices['models']) == {str(seed) for seed in SEEDS}, 'Choice model panel differs')
    audit.require(choices['candidate_masks'] == [list(mask) for mask in MASKS], 'Candidate mask set differs')
    audit.require(choices['final_test_effects_collected'] is False, 'Calibration claims final test collection')
    models_path = STUDY / 'training/models'
    for path, value in choices['source_hashes'].items(): audit.require(audit.digest(models_path / path) == value, 'Model source hash changed')
    selection_ids = np.flatnonzero(data['split'] == 1)
    learning_ids = np.flatnonzero(data['split'] == 0)
    missing = {'method2': [], 'method3': []}
    for seed in SEEDS:
        audit.guard(); directory = folder / f'seed-{seed}'
        masks = table(directory / 'mask-scores.csv'); rotations = table(directory / 'rotation-scores.csv')
        audit.require([int(row['candidate']) for row in masks] == list(range(220)), 'Mask candidate list differs')
        audit.require([tuple(map(int, row['mask'].split())) for row in masks] == list(MASKS), 'Mask coordinates differ')
        audit.require([int(row['candidate']) for row in rotations] == list(STEPS), 'Rotation candidate steps differ')
        for row in masks + rotations:
            metric = {key: parse_metric(row, key) for key in row if key not in ('candidate', 'mask')}
            audit.require(metric['eligible'] == eligible(metric), 'Candidate eligibility differs')
            audit.require(metric['absolute_gate_pass'] == gate(metric), 'Candidate gate differs')
            for name, count in (('active_changed', 3456), ('inactive_changed', 3456), ('all_changed', 6912), ('same_value', 2304)):
                audit.require(metric[name + '_n'] == count, 'Candidate group count differs')
            audit.require(all(math.isfinite(value) for value in metric.values() if type(value) is float), 'Nonfinite candidate metric')
        eligible_masks = [row for row in masks if parse_metric(row, 'eligible')]
        eligible_rotations = [row for row in rotations if parse_metric(row, 'eligible')]
        rank = lambda row: (float(row['active_changed_tv']), int(row['candidate']))
        selected = {'method1': min(masks, key=rank),
                    'method2': min(eligible_masks, key=rank) if eligible_masks else None,
                    'method3': min(eligible_rotations, key=rank) if eligible_rotations else None}
        record = choices['models'][str(seed)]
        audit.require(record == read_json(directory / 'choices.json'), 'Root and local choices differ')
        audit.require(record['weight_sha256'] == audit.digest(models_path / f'seed-{seed}/final.npz'), 'Chosen model weight hash differs')
        for method, expected in selected.items():
            choice = record['methods'][method]
            audit.require(choice['available'] == (expected is not None), 'Missing-eligible status differs')
            if expected is None:
                missing[method].append(seed)
                audit.require(not (directory / f'{method}-selection.npz').exists(), 'Unavailable choice has a raw selection result')
                continue
            audit.require(int(choice['selection_metrics']['candidate']) == int(expected['candidate']), 'Minimum-score selection differs')
            compare_metrics(audit, {key: parse_metric(expected, key) for key in expected if key not in ('candidate', 'mask', 'maximum_replay_error', 'maximum_metric_replay_error')}, choice['selection_metrics'])
            if method == 'method3':
                audit.require(choice['step'] == int(expected['candidate']), 'Wrong selected rotation step')
                audit.require(choice['checkpoint'] == f'seed-{seed}/rotation-step-{choice["step"]}.npz', 'Wrong selected checkpoint path')
                audit.require(choice['checkpoint_sha256'] == audit.digest(folder / choice['checkpoint']), 'Selected checkpoint hash differs')
            else: audit.require(choice['mask'] == list(MASKS[int(expected['candidate'])]), 'Selected mask differs')
            raw_metrics(audit, directory / f'{method}-selection.npz', selection_ids, data, intact[seed], choice['selection_metrics'])
        draws = audit.load(directory / 'learning-draws.npz')['pair_ids']
        audit.require(draws.shape == (4000, 192), 'Rotation draw shape differs')
        audit.require(np.all(np.isin(draws, learning_ids)), 'A rotation draw is outside learning data')
        audit.require(np.all(data['active'][draws[:, :64]] & ~data['same_value'][draws[:, :64]]), 'Wrong first learning group')
        audit.require(np.all(~data['active'][draws[:, 64:128]] & ~data['same_value'][draws[:, 64:128]]), 'Wrong second learning group')
        audit.require(np.all(data['same_value'][draws[:, 128:]]), 'Wrong third learning group')
        for step in STEPS:
            saved = audit.load(directory / f'rotation-step-{step}.npz')
            audit.require(saved['a'].shape == (12, 12) and saved['q'].shape == (12, 12), 'Rotation shape differs')
            audit.require(int(saved['adam_step']) == step and saved['losses'].shape == (step, 4), 'Rotation update record differs')
            audit.near(saved['q'].T @ saved['q'], np.eye(12), 'Rotation is not orthogonal')
            audit.near(saved['losses'][:, 0], saved['losses'][:, 1:].sum(1), 'Rotation loss is not the three-group sum')
    summary = read_json(folder / 'summary.json')
    audit.require(summary['passed'] is True and summary['status'] == 'complete', 'Edit calibration is incomplete')
    audit.require(summary['eligible_method2_models'] == 8 - len(missing['method2'])
                  and summary['eligible_method3_models'] == 8 - len(missing['method3']), 'Eligible-model counts differ')
    return choices, missing


def audit_final(audit, data, intact, choices):
    folder = STUDY / 'results/edits'; expected_ids = np.flatnonzero(data['split'] == 2)
    summary = read_json(folder / 'summary.json'); per_model = read_json(folder / 'per-model-results.json')
    audit.require(summary['passed'] is True and summary['status'] == 'complete' and summary['models'] == 8,
                  'Final calculation panel is incomplete')
    audit.require(summary['final_test_pairs_per_model'] == 9216, 'Final pair count differs')
    audit.require([row['seed'] for row in per_model] == list(SEEDS), 'Final model panel differs')
    rows = table(folder / 'test-group-metrics.csv')
    indexed = {(int(row['seed']), row['method']): row for row in rows}
    audit.require(len(indexed) == len(rows), 'Duplicate final metric rows')
    ledger = audit.load(folder / 'test-ledger.npz')
    audit.require(np.array_equal(ledger['pair_ids'], expected_ids), 'Final test IDs differ')
    for key in ('r', 'd', 'cf', 'active', 'same_value'):
        audit.require(np.array_equal(ledger[key], data[key][expected_ids]), 'Final ledger field differs: ' + key)
    audit.require(np.array_equal(ledger['required_label'], data['labels'][data['cf'][expected_ids]]), 'Final required labels differ')
    random_ids = np.random.Generator(np.random.PCG64(94001)).permutation(220)[:20].tolist()
    audit.require(choices['random_control_mask_ids'] == random_ids, 'Fixed random control IDs differ')
    expected_table_keys = set(); recomputed = []
    controls = {name: [] for name in ('none', 'full', 'coherent', 'random')}
    for record in per_model:
        seed = record['seed']; audit.guard(); directory = folder / f'seed-{seed}'
        audit.require(record == read_json(directory / 'summary.json'), 'Final per-model summaries differ')
        outcomes = {}
        for method in ('method1', 'method2', 'method3'):
            choice = choices['models'][str(seed)]['methods'][method]
            result = record['outcomes'][method]
            audit.require(result['available'] == choice['available'], 'Final availability changed')
            if not choice['available']:
                audit.require(result['absolute_gate_pass'] is False, 'Unavailable method counted as passing')
                audit.require(not (directory / f'{method}-test.npz').exists(), 'Unavailable method has a final raw result')
                outcomes[method] = dict(available=False, absolute_gate_pass=False)
                continue
            key = (seed, method); expected_table_keys.add(key)
            metric, _ = raw_metrics(audit, directory / f'{method}-test.npz', expected_ids, data, intact[seed], indexed[key])
            compare_metrics(audit, metric, result)
            outcomes[method] = dict(available=True, **metric)
        for kind in ('none', 'full', 'coherent'):
            key = (seed, 'control_' + kind); expected_table_keys.add(key)
            metric, prob = raw_metrics(audit, directory / f'control-{kind}.npz', expected_ids, data, intact[seed], indexed[key])
            controls[kind].append(dict(seed=seed, **metric))
            source = data[{'none': 'r', 'full': 'd', 'coherent': 'cf'}[kind]][expected_ids]
            audit.near(prob, intact[seed][source], 'Intact control output differs')
        for index, candidate in enumerate(random_ids):
            key = (seed, f'random_{index:02d}_mask_{candidate}'); expected_table_keys.add(key)
            metric, _ = raw_metrics(audit, directory / f'control-random-{index:02d}.npz', expected_ids, data, intact[seed], indexed[key])
            controls['random'].append(dict(seed=seed, candidate=candidate, **metric))
        gain = (outcomes['method3']['active_changed_accuracy'] - outcomes['method1']['active_changed_accuracy']) if outcomes['method3']['available'] else None
        secondary = (outcomes['method3']['all_changed_accuracy'] - outcomes['method1']['all_changed_accuracy']) if outcomes['method3']['available'] else None
        if gain is None: audit.require(record['active_accuracy_gain'] is None and record['all_changed_accuracy_gain'] is None, 'Missing gain was filled')
        else:
            audit.near(record['active_accuracy_gain'], gain, 'Primary paired gain differs')
            audit.near(record['all_changed_accuracy_gain'], secondary, 'Secondary paired gain differs')
        recomputed.append(dict(seed=seed, outcomes=outcomes, gain=gain, secondary=secondary))
    audit.require(set(indexed) == expected_table_keys, 'Final table has extra or missing methods')
    gates = {method: all(row['outcomes'][method]['available'] and row['outcomes'][method]['absolute_gate_pass'] for row in recomputed)
             for method in ('method1', 'method2', 'method3')}
    complete = all(row['gain'] is not None for row in recomputed)
    mean_gain = math.fsum(row['gain'] for row in recomputed) / 8 if complete else None
    secondary = math.fsum(row['secondary'] for row in recomputed) / 8 if complete else None
    positive = sum(row['gain'] is not None and row['gain'] > 0 for row in recomputed)
    improvement = bool(complete and gates['method3'] and mean_gain >= .05 and positive >= 6)
    audit.require(summary['variable_explanation_gate_by_method'] == gates, 'Whole-method practical gates differ')
    audit.require(summary['learned_method_improvement_gate_pass'] == improvement, 'Improvement gate differs')
    audit.require(summary['positive_model_differences'] == positive, 'Positive paired count differs')
    for name, value in (('primary_active_accuracy_mean_gain', mean_gain), ('all_changed_secondary_mean_gain', secondary)):
        if value is None: audit.require(summary[name] is None, 'Incomplete comparison was assigned a mean')
        else: audit.near(summary[name], value, 'Mean paired gain differs')
    audit.require(summary['normal_significance_test_used'] is False, 'An unplanned significance test is reported')
    audit.require(summary['numerical_controls_pass'] is True, 'Numerical controls did not pass')
    audit.require(summary['missing_eligible_candidates_are_failures_of_method_claim'] is True,
                  'Missing choices do not count as method-claim failures')
    method_tables = []
    method_means = {}
    for method in ('method1', 'method2', 'method3'):
        available = [row for row in recomputed if row['outcomes'][method]['available']]
        missing = [row['seed'] for row in recomputed if not row['outcomes'][method]['available']]
        method_means[method] = dict(
            scope='Descriptive means over available models only; not the eight-model claim',
            available_models=len(available), missing_seeds=missing,
            absolute_gate_pass_models=sum(row['outcomes'][method]['absolute_gate_pass'] for row in available),
            means=descriptive_means([row['outcomes'][method] for row in available]))
        for row in recomputed:
            result = row['outcomes'][method]
            metrics = reported_metrics(result) if result['available'] else {name: None for name in REPORT_METRICS}
            method_tables.append(dict(seed=row['seed'], method=method, available=result['available'],
                                      absolute_gate_pass=result['absolute_gate_pass'], **metrics))
    paired_available = [row for row in recomputed if row['gain'] is not None]
    conditional = dict(
        scope='Descriptive paired gains over available learned methods only; does not replace the primary eight-model result',
        available_models=len(paired_available), seeds=[row['seed'] for row in paired_available],
        missing_seeds=[row['seed'] for row in recomputed if row['gain'] is None],
        active_accuracy_mean_gain=math.fsum(row['gain'] for row in paired_available) / len(paired_available) if paired_available else None,
        all_changed_accuracy_mean_gain=math.fsum(row['secondary'] for row in paired_available) / len(paired_available) if paired_available else None)
    control_means = {name: dict(scope='Equal-weight descriptive mean over fixed model-control combinations',
                               models=8, combinations=len(values), means=descriptive_means(values))
                     for name, values in controls.items()}
    return dict(method_gates=gates, improvement_gate=improvement, primary_mean_gain=mean_gain,
                secondary_mean_gain=secondary, positive_differences=positive,
                per_model_method_metrics=method_tables, available_model_descriptive_means=method_means,
                available_model_paired_descriptive_gains=conditional, fixed_control_descriptive_means=control_means)


def audit_all(audit):
    for phase in ('preparation', 'training'):
        audit.check_manifest(STUDY / phase)
    audit.require(read_json(STUDY / 'preparation/status.json')['status'] == 'passed', 'Preparation did not pass')
    audit.require(read_json(STUDY / 'training/status.json')['status'] == 'complete', 'Training is incomplete')
    data = audit_data(audit, STUDY / 'training/models/data.npz')
    intact, ability, ability_gate = audit_models(audit, data)
    if not ability_gate:
        return dict(scope='Complete ordinary-model panel; edit stages blocked by ability', ability=ability,
                    ability_gate=False, edit_results_checked=False)
    fixed = audit.check_manifest(STUDY / 'calibration', allowed_extra=('frozen.zip',))
    audit.check_manifest(STUDY / 'calibration/edits', name='calibration-manifest.json', inner=True)
    audit.check_manifest(STUDY / 'results')
    for phase in ('calibration', 'results'):
        audit.require(read_json(STUDY / phase / 'status.json')['status'] == 'complete', phase + ' is incomplete')
    with zipfile.ZipFile(STUDY / 'calibration/frozen.zip') as archive:
        names = archive.namelist(); expected = {name for name in fixed if name.startswith('edits/')}
        audit.require(len(names) == len(set(names)) and set(names) == expected, 'Public calibration archive members differ')
        for name in names:
            audit.guard(); audit.require(hashlib.sha256(archive.read(name)).hexdigest() == fixed[name], 'Calibration archive bytes differ')
    choices, missing = audit_calibration(audit, data, intact)
    final = audit_final(audit, data, intact, choices)
    return dict(scope='Complete saved-result consistency audit', ability=ability, ability_gate=True,
                edit_results_checked=True, missing_eligible_candidates=missing, **final)


def atomic_receipt(path, result):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def error_record(exc):
    # Permit failures have fixed messages from guard; they contain no private
    # permit path, account details, or remaining-usage values.
    return dict(passed=False, error_type=type(exc).__name__, reason=str(exc).replace(str(ROOT), '[repository]'))


def main():
    global np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=49.)
    parser.add_argument('--output', type=Path, default=STUDY / 'analysis/audit.json')
    args = parser.parse_args()
    if not 2 < args.seconds <= 49: raise ValueError('Audit limit must exceed two seconds and be at most 49 seconds')
    if args.output.exists(): raise FileExistsError('Preserve the previous audit result')
    audit = Audit(args.seconds)
    lock = None; previous_alarm = None; timer_active = False
    try:
        lock_path = ROOT / '.local-state/research-compute.lock'
        lock_path.parent.mkdir(exist_ok=True)
        lock = lock_path.open('a')
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another research process holds the compute lock') from None
        # Reserve this new receipt. Only this process can replace its own file.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x'): pass
        previous_alarm = signal.signal(signal.SIGALRM, audit.alarm)
        signal.setitimer(signal.ITIMER_REAL, 1, 1); timer_active = True
        try:
            audit.guard()
            import numpy as np
            audit.guard()
            result = dict(passed=True, **audit_all(audit))
            audit.guard()
        except BaseException as exc:
            result = error_record(exc)
        closing_started = time.monotonic()
        audit.deadline = min(audit.total_deadline, closing_started + 2.)
        work_seconds = closing_started - audit.started
        result.update(no_model_forwards=True, no_training=True, study_code_imported=False,
                      saved_raw_files_checked=audit.raw_files, checks=audit.checks, hashed_files=audit.hashed,
                      maximum_arithmetic_difference=audit.max_error, work_seconds=work_seconds,
                      charged_seconds=work_seconds + 2., total_time_limit_seconds=args.seconds,
                      closing_receipt_allowance_seconds=2., closing_write_overrun=False,
                      controls=dict(memory_limit_gib=4, minimum_free_disk_gib=20, blas_threads=2,
                                    permit_maximum_lifetime_seconds=120, alarm_interval_seconds=1,
                                    shared_compute_lock=True),
                      timing_rule='Imports and checks are measured. Two seconds are charged for closing writes. The run fails if closing writes exceed that allowance or total elapsed time exceeds the limit.',
                      limits=['This checks saved output arithmetic, selection logic, and file integrity. It does not repeat training or model forwards.',
                              'Unselected candidate effects are not recomputed. Their recorded tables support eligibility and minimum-choice checks.',
                              'Rotation orthogonality and loss accounting are checked; gradient updates and matrix exponentials are not repeated.',
                              'The one-second signal is checked between native calls. A native-call overrun is recorded as a failure when control returns.'])
        try:
            audit.guard()
            atomic_receipt(args.output, result)
            result['seconds_through_first_receipt'] = time.monotonic() - audit.started
            audit.guard()
            atomic_receipt(args.output, result)
            # This check includes both receipt writes, including their fsyncs.
            audit.guard()
        except BaseException as exc:
            # Stop computation. A small failure receipt is permitted after a
            # resource stop; a failure can never be changed back to a pass.
            signal.setitimer(signal.ITIMER_REAL, 0); timer_active = False
            now = time.monotonic()
            result.update(error_record(exc))
            result.update(seconds_before_failure_receipt=now - audit.started,
                          closing_write_overrun=(now > audit.total_deadline or now - closing_started > 2.))
            atomic_receipt(args.output, result)
    finally:
        if timer_active: signal.setitimer(signal.ITIMER_REAL, 0)
        if previous_alarm is not None: signal.signal(signal.SIGALRM, previous_alarm)
        if lock is not None: lock.close()
    # Include lock release in the actual overrun decision. The final failure
    # record cannot self-time its own last byte; the outer process timer remains
    # the exact wall-time record, and any observed excess has exit status one.
    elapsed = time.monotonic() - audit.started
    if elapsed > args.seconds or time.monotonic() - closing_started > 2.:
        result.update(passed=False, error_type='TimeoutError', reason='Audit closing writes exceeded the time allocation',
                      closing_write_overrun=True, seconds_before_failure_receipt=elapsed)
        atomic_receipt(args.output, result)
    print(json.dumps(dict(passed=result['passed'], observed_seconds=elapsed,
                          charged_seconds=result['charged_seconds'], closing_write_overrun=result['closing_write_overrun'])), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
