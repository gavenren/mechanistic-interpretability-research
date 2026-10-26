"""Check saved Experiment 2B evidence. Do not run a model or new trials."""
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
from statistics import NormalDist
import subprocess
import sys
import time
import zipfile

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
PUBLIC_CALIBRATION = '5d05c4bdefc2f67986c89cae86bf1b25cb95f923'
NAMES = ('signed_class_zero', 'mean_absolute_class_zero', 'target_margin_gain',
         'target_probability_gain', 'answer_accuracy')
START = None
np = None
PRIOR_AUDIT_SECONDS = 0.


def need(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def peak():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def guard(*_):
    need(time.monotonic() - START + PRIOR_AUDIT_SECONDS < 59, 'Audit time limit')
    need(peak() <= 4 * 1024**3, 'Audit memory limit')
    try:
        permit = read(ROOT / '.local-state/edit-procedure-permit.json')
        expiry = permit['expires_at']; now = time.time()
        allowed = (permit['allowed'] is True and type(expiry) in (int, float)
                   and math.isfinite(expiry) and now < expiry <= now + 120)
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        allowed = False
    need(allowed, 'Usage permit is missing, invalid, or expired')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            guard(); h.update(block)
    return h.hexdigest()


def close(a, b, tolerance=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid compared array')
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    need(error <= tolerance, 'Saved value differs from its recalculation')
    return error


def arrays(path):
    guard()
    with np.load(path, allow_pickle=False) as saved:
        result = {name: saved[name].copy() for name in saved.files}
    need(all(np.isfinite(value).all() for value in result.values()), 'Nonfinite saved array')
    return result


def check_manifest(folder, name='manifest.json', extras=()):
    record = read(folder / name)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()
              and p.name != name and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
    need(actual == set(record) | set(extras), 'Manifest file list differs: ' + folder.name)
    for relative, expected in record.items():
        path = (folder / relative).resolve()
        need(path.is_relative_to(folder.resolve()), 'Invalid manifest path')
        need(digest(path) == expected, 'Manifest hash differs: ' + relative)
    return len(record)


def fixed_pairs(states):
    split = {}
    for active_s, same in itertools.product(range(2), repeat=2):
        blocks = [(r, a) for r in range(192) for a in range(4)
                  if states[r, 2] == active_s and int(states[r, 0] == a) == same]
        blocks.sort(key=lambda x: hashlib.sha256(f'selector-procedure-v1:{x[0]}:{x[1]}'.encode()).digest())
        split.update({pair: int(i >= len(blocks) // 2) for i, pair in enumerate(blocks)})
    rows = []
    for r, (a, b, s, n) in enumerate(states):
        for requested, context in itertools.product(range(4), range(2)):
            db, ds, dn = (b + context) % 4, s ^ context, (n + context) % 6
            donor = ((requested * 4 + db) * 2 + ds) * 6 + dn
            target_2a = ((requested * 4 + b) * 2 + s) * 6 + n
            rows.append((r, donor, target_2a, split[r, requested], int(a == requested), int(s == 0),
                         int(b == db), int(s == ds), int(n == dn), context))
    rows.sort(key=lambda row: 192 * row[0] + row[1])
    rows = np.asarray(rows, np.int64)
    return rows, rows[:, 0] * 192 + rows[:, 1]


def measures(logits, probability, labels):
    at = np.arange(len(labels))
    gain = probability[:, at, labels] - probability[0, at, labels]
    margin = logits[:, at, labels] - logits[:, at, (labels + 1) % 4]
    signed = probability[:, :, 0] - probability[0, :, 0]
    correct = probability.argmax(2) == labels
    tv = np.abs(probability - probability[255]).sum(2) * .5
    scores = np.stack((signed.mean(1), np.abs(signed).mean(1), (margin - margin[0]).mean(1),
                       gain.mean(1), correct.mean(1)), axis=1)
    accuracy = (probability.argmax(2) == probability[255].argmax(1)).mean(1)
    mean_tv = tv.mean(1)
    return scores, accuracy, mean_tv, gain, correct, tv


def check_search(saved, passed, orders):
    valid = np.flatnonzero(passed).tolist()
    minimum = min(mask.bit_count() for mask in valid)
    minima = [mask for mask in valid if mask.bit_count() == minimum]
    need(saved['passing_masks'] == valid and saved['minimum_masks'] == minima
         and saved['minimum_size'] == minimum, 'Exhaustive subset result differs')
    need(len(saved['greedy']) == 32, 'Greedy path count differs')
    excess = []
    for row, order in zip(saved['greedy'], orders):
        need(row['order'] == list(order) and len(row['steps']) == 8, 'Greedy order differs')
        kept = set(range(8))
        for step, bit in zip(row['steps'], order):
            before = sum(2**x for x in kept); proposed = kept - {bit}
            mask = sum(2**x for x in proposed); accepted = bool(passed[mask])
            need(step == dict(before=before, removed_bit=bit, proposal=mask, accepted=accepted), 'Greedy step differs')
            if accepted: kept = proposed
        mask = sum(2**x for x in kept)
        further = [x for x in sorted(kept) if passed[mask - 2**x]]
        need(row['retained_mask'] == mask and row['retained_size'] == len(kept)
             and row['further_single_deletions'] == further and row['excess_size'] == len(kept) - minimum,
             'Greedy endpoint differs')
        excess.append(len(kept) - minimum)
    need(saved['all_greedy_orders_find_minimum'] == all(x == 0 for x in excess), 'Greedy claim differs')
    return minimum, minima, max(excess)


def subsets_check():
    with (ROOT / 'studies/2026-10-13-inactive-variable-audit/data/inputs.csv').open() as stream:
        source = list(csv.DictReader(stream))
    states = np.array([[int(row[x]) for x in ('a', 'b', 's', 'n')] for row in source])
    labels = np.array([int(row['label']) for row in source])
    rows, ids = fixed_pairs(states)
    first, last = tuple(range(8)), tuple(reversed(range(8)))
    rest = [p for p in itertools.permutations(range(8)) if p not in (first, last)]
    rest.sort(key=lambda p: hashlib.sha256(('search-procedure-v1:order:' + ','.join(map(str, p))).encode()).digest())
    orders = [first, last] + rest[:30]
    bank_order = sorted(range(1, 255), key=lambda m: hashlib.sha256(f'search-procedure-v1:subset:{m}'.encode()).digest())
    frozen = read(STUDY / 'calibration/subsets/calibration.json')
    need([x['seed'] for x in frozen['models']] == list(range(8)), 'Frozen model order differs')
    output = []; banks = {}; replay_max = 0.; total = 0; score_rows = []
    for seed in range(8):
        record = frozen['models'][seed]; selected = None
        for split, phase, prefix in ((0, 'calibration', 'development'), (1, 'results', 'test')):
            guard(); expected_rows = rows[rows[:, 3] == split]; expected_ids = ids[rows[:, 3] == split]
            folder = STUDY / phase / 'subsets' / f'{prefix}-seed-{seed}'
            logits, probability = [], []
            for start in range(0, 256, 16):
                a = arrays(folder / 'subsets' / f'block-{start:03d}.npz')
                need(np.array_equal(a['rows'], expected_rows) and np.array_equal(a['row_ids'], expected_ids), 'Subset row split differs')
                need(np.array_equal(a['subset_ids'], np.arange(start, start + 16)), 'Subset mask order differs')
                need(a['logits'].shape == a['probabilities'].shape == (16, 768, 4), 'Subset output shape differs')
                normalized = np.exp(a['logits'] - a['logits'].max(2, keepdims=True))
                close(a['probabilities'], normalized / normalized.sum(2, keepdims=True))
                check = read(folder / 'subsets' / f'block-{start:03d}-audit.json')
                need(check['passed'] is True and check['probability_rows'] == 16 * 768
                     and check['scalar_anchors'] == 4 and check['tolerance'] == 1e-10, 'Subset replay record differs')
                error = max(check['maximum_replay_error'], check['maximum_scalar_anchor_error'])
                need(0 <= error <= 1e-10, 'Subset replay error exceeds tolerance')
                replay_max = max(replay_max, error); total += 16 * 768
                logits.append(a['logits']); probability.append(a['probabilities'])
            logits, probability = np.concatenate(logits), np.concatenate(probability)
            target = labels[expected_rows[:, 0]]
            need(np.mean(probability[255].argmax(1) == target) >= .99, 'Full-panel accuracy failed')
            scores, accuracy, mean_tv, gain, correct, tv = measures(logits, probability, target)
            passed = (accuracy >= .99) & (mean_tv <= .01)
            saved = read(folder / 'summary.json')
            close(scores, saved['scores']); close(accuracy, saved['gate_accuracy']); close(mean_tv, saved['gate_mean_tv'])
            need(np.array_equal(passed, saved['passed_masks']), 'Subset pass decisions differ')
            minimum, minima, excess = check_search(saved['search'], passed, orders)
            need(saved['score_names'] == list(NAMES) and len(saved['groups']) == 8, 'Subset summary schema differs')
            for group in saved['groups']:
                take = ((expected_rows[:, 4] == group['same_a']) & (expected_rows[:, 5] == group['active'])
                        & (expected_rows[:, 9] == group['context']))
                need(int(take.sum()) == group['n'], 'Subset group count differs')
                close(tv[:, take].mean(1), group['tv']); close(correct[:, take].mean(1), group['accuracy'])
                close(measures(logits[:, take], probability[:, take], target[take])[0], group['scores'])
            if split == 0:
                choices = [m for m in range(256) if m.bit_count() == 4]
                selected = {name: min(choices, key=lambda m: (-scores[m, i], m)) for i, name in enumerate(NAMES)}
                need(record['score_choices'] == selected, 'Frozen score choice differs from development scores')
                single = arrays(folder / 'single-deletions.npz')
                need(np.array_equal(single['rows'], expected_rows), 'Panel-selection row split differs')
                sensitivity = np.abs(single['probabilities'] - probability[255]).sum(2).mean(1) / 2
                panel = sorted(range(12), key=lambda n: (-float(sensitivity[n]), n))[:8]
                need(panel == record['panel'], 'Frozen panel differs from development sensitivity')
                bank_path = STUDY / 'calibration/subsets' / record['bank']; bank = arrays(bank_path)
                need(digest(bank_path) == record['bank_sha256'], 'Frozen bank hash differs')
                need(np.array_equal(bank['subset_ids'], bank_order) and np.array_equal(bank['rows'], expected_rows)
                     and np.array_equal(bank['row_ids'], expected_ids), 'Development bank IDs differ')
                effects = gain[bank_order].T; close(effects, bank['effects']); close(effects.mean(0), bank['mean'])
                centered = effects - effects.mean(0); centered -= centered.mean(0)
                centered[:, np.all(effects == effects[0], axis=0)] = 0
                close(centered, bank['centered_effects']); close((centered**2).mean(0), bank['variance'])
                banks[seed] = bank['centered_effects']
            else:
                for name, mask in selected.items():
                    chosen = saved['development_selected_candidates'][name]
                    need(chosen['mask'] == mask and chosen['passed'] == bool(passed[mask]), 'Final selected candidate differs')
                    close(chosen['scores'], scores[mask]); close(chosen['accuracy'], accuracy[mask]); close(chosen['mean_tv'], mean_tv[mask])
                    score_rows.append(dict(seed=seed, score=name, mask=mask, accuracy=float(accuracy[mask]),
                                           mean_tv=float(mean_tv[mask]), passed=bool(passed[mask])))
                output.append(dict(seed=seed, minimum_size=minimum, minimum_masks=minima,
                                   passing_masks=np.flatnonzero(passed).tolist(), maximum_greedy_excess=excess))
    need(total == 3145728, 'Combined subset output count differs')
    return dict(models=output, selected_candidates=score_rows, output_rows=total,
                recorded_maximum_replay_error=replay_max), banks, bank_order


def selection_check(banks, order):
    folder = STUDY / 'results/selection'
    controls = read(folder / 'selection-controls.json'); need(len(controls) == 24, 'Selection control count differs')
    with (folder / 'selection-rates.csv').open() as stream: table = list(csv.DictReader(stream))
    need(len(table) == 72, 'Selection table count differs')
    lookup = {(int(row['model']), int(row['candidates']), row['method']): row for row in table}
    need(len(lookup) == 72, 'Duplicate selection table row')
    normal = NormalDist(); threshold = normal.inv_cdf(.95); interval_z = normal.inv_cdf(.975)
    conditions = []; rate_rows = []; replay_max = 0.
    for model in range(8):
        for count in (8, 32, 128):
            guard(); a = arrays(folder / f'selection-seed-{model}-candidates-{count}.npz')
            bank = banks[model][:, :count]; variance = (bank * bank).mean(0)
            close(a['known_variance'], variance)
            need(a['all_selection_means'].shape == (2000, count), 'Candidate mean count differs')
            need(np.array_equal(a['candidate_subset_ids'], order[:count]), 'Candidate order differs')
            seeds = (2026102600 + 10000 * model + count, 2026107600 + 10000 * model + count)
            for name, seed in zip(('selection_indices', 'validation_indices'), seeds):
                recorded = a[name]
                need(recorded.shape == (2000, 64) and recorded.dtype == np.uint16, 'Recorded draw shape or type differs')
                regenerated = np.random.Generator(np.random.PCG64(seed)).integers(0, 768, size=(2000, 64), dtype=np.int64)
                need(np.array_equal(recorded, regenerated), 'Recorded draw seed differs')
            standardized = np.zeros((2000, count)); positive = variance > 0
            standardized[:, positive] = a['all_selection_means'][:, positive] / np.sqrt(variance[positive] / 64)
            winner = standardized.argmax(1)
            need(np.array_equal(winner, a['selected_column']) and np.array_equal(np.asarray(order)[winner], a['selected_subset_id']), 'Recorded winner differs')
            means = [bank[a[key], winner[:, None]].sum(1) / 64 for key in ('selection_indices', 'validation_indices')]
            active = variance[winner] > 0; zs = []
            for mean, phase in zip(means, ('selection', 'validation')):
                close(mean, a[f'selected_{phase}_mean'])
                z = np.zeros(2000); z[active] = mean[active] / np.sqrt(variance[winner[active]] / 64)
                close(z, a[f'selected_{phase}_z']); zs.append(z)
            check = controls[len(conditions)]
            need(check['model'] == model and check['candidates'] == count and check['passed'] is True
                 and (check['selection_seed'], check['validation_seed']) == seeds, 'Selection control order differs')
            need(0 <= check['saved_trial_replay_error'] <= 1e-10, 'Selection replay error exceeds tolerance')
            replay_max = max(replay_max, check['saved_trial_replay_error'])
            for method, flags in (('reused', (zs[0] > threshold) & active),
                                  ('independent_validation', (zs[1] > threshold) & active),
                                  ('bonferroni', (zs[0] > normal.inv_cdf(1 - .05 / count)) & active)):
                key = 'reject_validation' if method == 'independent_validation' else 'reject_' + method
                need(np.array_equal(flags, a[key]), 'Saved rejection decision differs')
                rejected = int(flags.sum()); rate = rejected / 2000; z2 = interval_z**2
                upper = (rate + z2 / 4000 + interval_z * math.sqrt(rate * (1-rate) / 2000 + z2 / (4*2000**2))) / (1 + z2 / 2000)
                row = lookup[model, count, method]
                need(int(row['trials']) == 2000 and int(row['rejected']) == rejected
                     and (row['meets_7_percent_bound'] == 'True') == (upper <= .07), 'Selection count or bound decision differs')
                close(float(row['rejection_rate']), rate); close(float(row['wilson_upper_95']), upper)
                item = dict(model=model, candidates=count, method=method, rejected=rejected,
                            trials=2000, rate=rate, wilson_upper=upper, passes_bound=upper <= .07)
                rate_rows.append(item)
                if method == 'independent_validation': conditions.append(item)
    summary = read(folder / 'selection-summary.json'); passed = sum(x['passes_bound'] for x in conditions)
    need(summary['primary_calibration_gate_pass'] == (passed == 24)
         and summary['later_positive_claims_may_use_validation_rule'] == (passed == 24)
         and summary['primary_conditions_passed'] == passed and summary['total_trials'] == 48000,
         'Primary scientific gate differs')
    close(summary['largest_primary_wilson_upper'], max(x['wilson_upper'] for x in conditions))
    return dict(primary_gate_passed=passed == 24, primary_conditions_passed=passed,
                failed_conditions=[x for x in conditions if not x['passes_bound']], rates=rate_rows,
                recorded_maximum_replay_error=replay_max, total_trials=48000,
                audit_scope='All recorded draws, winners, selected means, scores, decisions and intervals checked. Full candidate-mean replay was not repeated; the fixed runner recorded it.')


def main():
    global START, np, PRIOR_AUDIT_SECONDS
    START = time.monotonic(); receipt = dict(status='failed', scientific_work=False, charged_seconds=60)
    output = STUDY / 'analysis/audit.json'
    previous = STUDY / 'analysis/audit-attempt-1.json'
    if previous.exists():
        earlier = read(previous)
        PRIOR_AUDIT_SECONDS = earlier['measured_seconds_before_receipt'] + earlier['closing_receipt_allowance_seconds']
        receipt.update(prior_failed_audit='audit-attempt-1.json', prior_audit_seconds_with_allowance=PRIOR_AUDIT_SECONDS,
                       charge_note='The 60-second charge is shared across both audit attempts. Do not add the failed receipt charge again.')
    need(not output.exists(), 'Keep the existing audit receipt')
    with (ROOT / '.local-state/research-compute.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            guard(); signal.signal(signal.SIGALRM, guard); signal.setitimer(signal.ITIMER_REAL, 1, 1)
            import numpy as numpy
            np = numpy
            status = read(STUDY / 'results/status.json'); need(status['status'] == 'complete', 'Scientific run is not complete')
            need(status['plan_lock'] == PUBLIC_CALIBRATION and status['forwarded_tokens'] == 0, 'Wrong scientific lock or token charge')
            receipt['manifests'] = dict(results=check_manifest(STUDY / 'results'),
                preparation=check_manifest(STUDY / 'preparation'),
                calibration=check_manifest(STUDY / 'calibration', extras=('banks.zip',)),
                calibration_subsets=check_manifest(STUDY / 'calibration/subsets', 'calibration-manifest.json'))
            for name in ('status.json', 'manifest.json', 'subsets/calibration.json', 'banks.zip'):
                path = STUDY / 'calibration' / name
                public = subprocess.check_output(['git', 'show', PUBLIC_CALIBRATION + ':' + path.relative_to(ROOT).as_posix()], cwd=ROOT)
                need(public == path.read_bytes(), 'Public calibration bytes differ')
            raw = read(STUDY / 'calibration/manifest.json')
            with zipfile.ZipFile(STUDY / 'calibration/banks.zip') as archive:
                names = {f'subsets/bank-seed-{seed}.npz' for seed in range(8)}
                need(len(archive.namelist()) == 8 and set(archive.namelist()) == names, 'Public bank members differ')
                for name in names: need(hashlib.sha256(archive.read(name)).hexdigest() == raw[name], 'Public bank hash differs')
            source = read(STUDY / 'input-manifest.json')
            for name, expected in source['files'].items(): need(digest(ROOT / name) == expected, 'Input hash differs')
            receipt['source_files_checked'] = len(source['files'])
            receipt['subsets'], banks, order = subsets_check()
            receipt['selection'] = selection_check(banks, order)
            guard(); receipt['status'] = 'passed'
        except BaseException as exc:
            receipt.update(status='failed', error_type=type(exc).__name__, reason=str(exc).replace(str(ROOT), '[repository]'))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            receipt.update(measured_seconds_before_receipt=time.monotonic() - START, closing_receipt_allowance_seconds=1,
                           peak_memory_bytes=peak(), model_forward_calls=0, new_simulation_trials=0)
            receipt['combined_audit_seconds_with_closing_allowances'] = PRIOR_AUDIT_SECONDS + receipt['measured_seconds_before_receipt'] + 1
            output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            if time.monotonic() - START + PRIOR_AUDIT_SECONDS >= 60:
                receipt.update(status='failed', reason='Audit exceeded its full 60-second allowance')
                output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'measured_seconds_before_receipt', 'peak_memory_bytes')}))
    return receipt['status'] == 'passed'


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
