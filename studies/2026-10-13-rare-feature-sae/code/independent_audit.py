#!/usr/bin/env python3
"""Read-only audit from saved weights; does not import the study runner or train."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
CHECKS = 0
NUMBERS = 0
MAX_ERROR = 0.0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def near(actual, expected, label, tolerance=2e-11):
    global NUMBERS, MAX_ERROR
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    check(actual.shape == expected.shape, label + ': shape')
    check(bool(np.all(np.isfinite(actual))) and bool(np.all(np.isfinite(expected))), label + ': finite')
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    NUMBERS += actual.size
    MAX_ERROR = max(MAX_ERROR, error)
    check(error <= tolerance, f'{label}: difference {error}')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array_digest(array):
    return digest(np.ascontiguousarray(array).tobytes())


def read_csv(name):
    with (HERE / 'results' / name).open(newline='') as source:
        return list(csv.DictReader(source))


def key(row):
    return int(row['seed']), float(row['p']), float(row['amplitude']), row['objective']


def probability(h, direction, amplitude):
    logits = 8 * (np.matmul(h, direction) / amplitude - 0.5)
    return 1 / (1 + np.exp(np.clip(-logits, -700, 700)))


def rebuild_data(seed):
    dictionary_rng = np.random.default_rng(1000 + seed)
    orthogonal, upper = np.linalg.qr(dictionary_rng.normal(size=(12, 12)))
    orthogonal *= np.where(np.diag(upper) < 0, -1., 1.)
    direction = orthogonal[:, 0]
    coefficients = dictionary_rng.normal(size=(20, 11))
    coefficients /= np.sqrt(np.sum(coefficients ** 2, axis=1))[:, None]
    dictionary = coefficients @ orthogonal[:, 1:].T
    rng = np.random.default_rng(2000 + seed)
    banks = {}
    for split, count in [('train', 10000), ('validation', 1024), ('test', 2048)]:
        present = rng.random((count, 20)) < .1
        magnitudes = rng.uniform(.5, 1.5, (count, 20))
        banks[split] = (present * magnitudes) @ dictionary
    order = rng.permutation(10000)
    return direction, dictionary, banks, order


def encode(h, weights):
    pre = (h - weights['mean']) @ weights['Wenc'] + weights['benc']
    positive = np.maximum(pre, 0)
    indices = np.broadcast_to(np.arange(24), positive.shape)
    order = np.lexsort((indices, -positive), axis=1)[:, :4]
    code = np.zeros_like(positive)
    np.put_along_axis(code, order, np.take_along_axis(positive, order, axis=1), axis=1)
    reconstruction = code @ weights['Wdec'] + weights['bdec'] + weights['mean']
    return code, reconstruction


def evaluate(background, weights, direction, amplitude):
    off = background
    on = background + amplitude * direction
    zoff, roff = encode(off, weights)
    zon, ron = encode(on, weights)
    delta = (zon - zoff)[:, :, None] * weights['Wdec'][None, :, :]
    # Explicit vectors, independent of the runner's scalar readout shortcut.
    q01 = probability(off[:, None, :] + delta, direction, amplitude)
    q10 = probability(on[:, None, :] - delta, direction, amplitude)
    r01 = probability(roff[:, None, :] + delta, direction, amplitude)
    r10 = probability(ron[:, None, :] - delta, direction, amplitude)
    q0, q1 = probability(off, direction, amplitude), probability(on, direction, amplitude)
    count0 = np.sum(zoff > 0, axis=1)[:, None] - (zoff > 0) + (zon > 0)
    count1 = np.sum(zon > 0, axis=1)[:, None] - (zon > 0) + (zoff > 0)
    native_error = (np.abs(q01 - q1[:, None]) + np.abs(q10 - q0[:, None])) / 2
    recon_error = (np.abs(r01 - q1[:, None]) + np.abs(r10 - q0[:, None])) / 2
    full = np.sum(delta, axis=1)
    baseline_arrays = {
        'full_native_error': (np.abs(probability(off + full, direction, amplitude) - q1)
                              + np.abs(probability(on - full, direction, amplitude) - q0)) / 2,
        'full_reconstruction_error': (np.abs(probability(ron, direction, amplitude) - q1)
                                      + np.abs(probability(roff, direction, amplitude) - q0)) / 2,
        'empty_native_error': np.abs(q1 - q0),
        'empty_reconstruction_error': (np.abs(probability(roff, direction, amplitude) - q1)
                                       + np.abs(probability(ron, direction, amplitude) - q0)) / 2,
    }
    scores = {
        'native_error': native_error.mean(axis=0),
        'reconstruction_error': recon_error.mean(axis=0),
        'native_off_to_on_error': np.abs(q01 - q1[:, None]).mean(axis=0),
        'native_on_to_off_error': np.abs(q10 - q0[:, None]).mean(axis=0),
        'decoder_cosine': (weights['Wdec'] @ direction) / np.linalg.norm(weights['Wdec'], axis=1),
        'activation_rate_off': np.mean(zoff > 0, axis=0),
        'activation_rate_on': np.mean(zon > 0, axis=0),
        'mean_activation_difference': np.mean(zon - zoff, axis=0),
        'patched_activity_mean': (count0.mean(axis=0) + count1.mean(axis=0)) / 2,
        'patched_activity_max': np.maximum(count0.max(axis=0), count1.max(axis=0)),
    }
    outputs = dict(native_off_to_on=q01, native_on_to_off=q10,
                   recon_off_to_on=r01, recon_on_to_off=r10, true_off=q0, true_on=q1,
                   native_error=native_error, reconstruction_error=recon_error,
                   patched_activity_off_to_on=count0, patched_activity_on_to_off=count1,
                   **baseline_arrays)
    return scores, baseline_arrays, outputs


def main():
    lock = json.loads((HERE / 'results' / 'lock.json').read_text())
    for name, expected_hash in lock['sha256'].items():
        content = (REPO / name).read_bytes()
        check(digest(content) == expected_hash, 'lock hash: ' + name)
        committed = subprocess.check_output(['git', 'show', lock['commit'] + ':' + name], cwd=REPO)
        check(content == committed, 'lock commit: ' + name)
    manifest = json.loads((HERE / 'results' / 'manifest.json').read_text())
    for name, expected_hash in manifest.items():
        check(digest((HERE / name).read_bytes()) == expected_hash, 'manifest: ' + name)

    selected = read_csv('selected-latents.csv')
    candidates = read_csv('all-latent-scores.csv')
    population = read_csv('population-metrics.csv')
    nuisance_rows = read_csv('same-label-controls.csv')
    check(len(nuisance_rows) == 144, 'all label-specific nuisance rows')
    check(len(selected) == 72 and len(candidates) == 72 * 2 * 24, 'complete fixed run grid')
    check(len({key(row) for row in selected}) == 72, 'unique run keys')
    records = json.loads((HERE / 'data' / 'generator-records.json').read_text())
    data = {}
    zero_counts = []
    for seed in range(6):
        direction, dictionary, banks, order = rebuild_data(seed)
        data[seed] = direction, banks, order
        record = next(row for row in records if row['seed'] == seed)
        saved = np.load(HERE / 'data' / f'dictionary-seed-{seed}.npz')
        near(direction, saved['rare_direction'], 'rare direction')
        near(dictionary, saved['background_dictionary'], 'background dictionary')
        near(dictionary @ direction, np.zeros(20), 'orthogonal task and background')
        # Exact generator hashes are also checked, beyond the numeric comparison.
        check(array_digest(direction) == record['rare_direction_sha256'], 'direction hash')
        check(array_digest(dictionary) == record['dictionary_sha256'], 'dictionary hash')
        check(array_digest(order) == record['permutation_sha256'], 'rare case order hash')
        sets = {}
        for split, bank in banks.items():
            check(array_digest(bank) == record['arrays'][split]['sha256'], split + ' array hash')
            zero_count = int(np.all(bank == 0, axis=1).sum())
            check(zero_count == record['zero_background_counts'][split], split + ' zero count')
            zero_counts.append(zero_count)
            sets[split] = {array_digest(row) for row in bank}
        for pair, count in record['exact_cross_split_shared_unique_values'].items():
            left, right = pair.split('__')
            check(len(sets[left] & sets[right]) == count, 'cross-split exact overlap')

    recalculated = {}
    for row in selected:
        seed, p, amplitude, objective = key(row)
        direction, banks, order = data[seed]
        name = f'seed-{seed}_p-{p:g}_a-{amplitude:g}_{objective}'
        weights = dict(np.load(HERE / 'models' / (name + '.npz')))
        near(np.linalg.norm(weights['Wdec'], axis=1), np.ones(24), name + ' decoder norms')
        labels = np.zeros(10000, dtype=int)
        labels[order[:round(p * 10000)]] = 1
        train_h = banks['train'] + amplitude * labels[:, None] * direction
        near(weights['mean'], train_h.mean(axis=0), name + ' training mean')
        near(float(row['training_rare_count']), float(labels.sum()), name + ' rare support')
        train_z, _ = encode(train_h, weights)
        near(float(row['dead_training_latents']), float((~np.any(train_z > 0, axis=0)).sum()), 'dead latents')
        near(float(row['training_activity_mean']), float(np.mean(np.sum(train_z > 0, axis=1))), 'training activity')
        result_by_split = {}
        for split in ('validation', 'test'):
            scores, baselines, outputs = evaluate(banks[split], weights, direction, amplitude)
            result_by_split[split] = scores, baselines, outputs
            indexed = {int(item['latent']): item for item in candidates if key(item) == key(row) and item['split'] == split}
            check(set(indexed) == set(range(24)), 'all candidate latent identities')
            for latent in range(24):
                for metric, values in scores.items():
                    near(float(indexed[latent][metric]), float(values[latent]), name + ' ' + split + ' ' + metric)
        validation = result_by_split['validation'][0]['native_error']
        chosen = int(np.flatnonzero(validation <= validation.min() + 1e-12)[0])
        check(chosen == int(row['selected_latent']), name + ' validation-only selection')
        near(float(row['validation_native_error']), float(validation[chosen]), name + ' validation winner')
        scores, baselines, outputs = result_by_split['test']
        for metric, values in scores.items():
            near(float(row['test_' + metric]), float(values[chosen]), name + ' selected ' + metric)
        for metric, values in baselines.items():
            near(float(row['test_' + metric]), float(values.mean()), name + ' ' + metric)
        saved_outputs = dict(np.load(HERE / 'results' / 'selected-test-outputs' / (name + '.npz')))
        for metric, values in outputs.items():
            near(saved_outputs[metric], values[:, chosen] if values.ndim == 2 else values, name + ' raw ' + metric)

        nuisance = []
        pops = []
        for label in (0, 1):
            h = banks['test'] + label * amplitude * direction
            z, rec = encode(h, weights)
            donor_indices = (np.arange(len(h)) + 1) % len(h)
            patched_code = z.copy()
            patched_code[:, chosen] = z[donor_indices, chosen]
            difference = (patched_code - z) @ weights['Wdec']
            true = probability(h, direction, amplitude)
            original_rec = probability(rec, direction, amplitude)
            native = probability(h + difference, direction, amplitude)
            recon = probability(rec + difference, direction, amplitude)
            near(saved_outputs[f'same_label_{label}_native'], native, name + ' nuisance native')
            near(saved_outputs[f'same_label_{label}_reconstruction'], recon, name + ' nuisance reconstruction')
            nuisance.append((np.abs(native - true).mean(), np.abs(recon - original_rec).mean(),
                             np.abs(recon - true).mean(), np.sum(patched_code > 0, axis=1).max()))
            stored_nuisance = [item for item in nuisance_rows if key(item) == key(row) and int(item['label']) == label]
            check(len(stored_nuisance) == 1, 'unique nuisance row')
            check(int(stored_nuisance[0]['selected_latent']) == chosen, 'nuisance selected latent')
            nuisance_expected = dict(native_change=nuisance[-1][0], reconstruction_change=nuisance[-1][1],
                                     reconstruction_true_error=nuisance[-1][2], patched_activity_max=nuisance[-1][3],
                                     patched_activity_mean=np.sum(patched_code > 0, axis=1).mean())
            for metric, value in nuisance_expected.items():
                near(float(stored_nuisance[0][metric]), float(value), name + ' label-specific nuisance ' + metric)
            logits = 8 * (rec @ direction / amplitude - .5)
            kl = true * np.log(true) + (1 - true) * np.log1p(-true) + np.logaddexp(0., logits) - true * logits
            pop = dict(reconstruction_mse=np.mean(np.sum((rec - h) ** 2, axis=1)),
                       downstream_kl=kl.mean(), probability_error=np.abs(original_rec - true).mean(),
                       classification_accuracy=np.mean((original_rec >= .5) == label),
                       activity_mean=np.mean(np.sum(z > 0, axis=1)))
            stored_pop = [item for item in population if key(item) == key(row) and int(item['label']) == label]
            check(len(stored_pop) == 1, 'unique population row')
            for metric, value in pop.items():
                near(float(stored_pop[0][metric]), float(value), name + ' population ' + metric)
            pops.append(pop)
        for metric in pops[0]:
            natural = (1 - p) * pops[0][metric] + p * pops[1][metric]
            near(float(row['natural_' + metric]), float(natural), name + ' natural ' + metric)
        for index, metric in enumerate(('same_label_native_change', 'same_label_reconstruction_change', 'same_label_reconstruction_true_error')):
            near(float(row[metric]), float(np.mean([x[index] for x in nuisance])), name + ' ' + metric)
        near(float(row['same_label_patched_activity_max']), float(max(x[3] for x in nuisance)), 'nuisance activity max')
        recalculated[key(row)] = float(scores['native_error'][chosen])

    differences = np.array([recalculated[(seed, .01, .25, 'downstream')] - recalculated[(seed, .01, .25, 'balanced')] for seed in range(6)])
    summary = json.loads((HERE / 'results' / 'summary.json').read_text())
    primary = summary['primary']
    near(primary['values'], differences, 'six primary seed differences')
    for metric, value in [('mean', differences.mean()), ('median', np.median(differences)), ('minimum', differences.min()), ('maximum', differences.max())]:
        near(primary[metric], float(value), 'primary ' + metric)
    indices = np.random.default_rng(20261013).integers(0, 6, size=(10000, 6))
    interval = np.quantile(differences[indices].mean(axis=1), [.025, .975])
    near(primary['ci95'], interval, 'paired percentile interval')
    for item in read_csv('paired-contrasts.csv'):
        seed, p, amplitude = int(item['seed']), float(item['p']), float(item['amplitude'])
        near(float(item['ordinary_minus_balanced']), recalculated[(seed, p, amplitude, 'downstream')] - recalculated[(seed, p, amplitude, 'balanced')], 'factorial paired contrast')
    for item in summary['factorial']:
        group = [row for row in selected if float(row['p']) == item['p'] and float(row['amplitude']) == item['amplitude'] and row['objective'] == item['objective']]
        check(len(group) == 6 and item['n'] == 6, 'factorial six-seed group')
        for metric in item:
            if metric not in ('p', 'amplitude', 'objective', 'n'):
                near(item[metric], float(np.mean([float(row[metric]) for row in group])), 'factorial mean ' + metric)
    receipt = dict(passed=True, model_runs=72, candidate_rows=3456, label_specific_nuisance_rows=len(nuisance_rows), checks=CHECKS,
                   numeric_values=NUMBERS, maximum_absolute_difference=MAX_ERROR,
                   manifest_files=len(manifest), zero_background_counts=zero_counts,
                   primary_seed_differences=differences.tolist(), primary_mean=float(differences.mean()),
                   primary_ci95=interval.tolist(), lock_commit=lock['commit'],
                   scope='Independent data reconstruction, all candidate scores, all selected raw outputs, model-derived population and nuisance metrics, and aggregate primary results. No training rerun.')
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
