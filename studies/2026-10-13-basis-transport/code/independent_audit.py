#!/usr/bin/env python3
"""Independent read-only calculation audit. No training and no runner imports."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
CHECKS = 0
NUMBERS = 0
MAX_ERROR = 0.0
MASKS = list(itertools.combinations(range(8), 4))


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def near(actual, expected, label, tolerance=1e-10):
    global NUMBERS, MAX_ERROR
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    check(actual.shape == expected.shape, label + ': shape')
    check(bool(np.all(np.isfinite(actual))) and bool(np.all(np.isfinite(expected))), label + ': finite')
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    NUMBERS += actual.size
    MAX_ERROR = max(MAX_ERROR, error)
    check(error <= tolerance, f'{label}: difference {error}')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(relative):
    with (HERE / relative).open(newline='') as source:
        return list(csv.DictReader(source))


def predict(states, weights, first_weight):
    shape = states.shape[:-1]
    flat = states.reshape(-1, 8)
    logits = np.tanh(flat @ first_weight + weights['b1']) @ weights['W2'] + weights['b2']
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return (exp / exp.sum(axis=-1, keepdims=True)).reshape(shape + (25,))


def metrics(prediction, target, labels, keep, matched):
    distances = np.abs(prediction - target).sum(axis=-1) / 2
    result = {'n': int(keep.sum()), 'tv': distances[keep].mean(),
              'semantic_accuracy': np.mean(prediction[keep].argmax(axis=-1) == labels[keep])}
    for name, condition in [('b_match', matched), ('b_mismatch', ~matched)]:
        group = keep & condition
        result[name + '_n'] = int(group.sum())
        result[name + '_tv'] = distances[group].mean()
    return result


def main():
    lock = json.loads((HERE / 'results' / 'lock.json').read_text())
    for filename, digest in lock['sha256'].items():
        content = (REPO / filename).read_bytes()
        check(sha(content) == digest, 'lock hash')
        check(content == subprocess.check_output(['git', 'show', lock['commit'] + ':' + filename], cwd=REPO), 'lock content')
    manifest = json.loads((HERE / 'results' / 'manifest.json').read_text())
    for filename, digest in manifest.items():
        check(sha((HERE / filename).read_bytes()) == digest, 'manifest ' + filename)

    inputs = csv_rows('data/inputs.csv')
    states = np.array([(a, b) for a in range(5) for b in range(5)])
    labels = 5 * ((states[:, 0] + states[:, 1]) % 5) + states[:, 1]
    near([[int(row[k]) for k in ('id', 'a', 'b', 'label')] for row in inputs],
         np.column_stack((np.arange(25), states, labels)), 'task states and labels')
    pair_rows = csv_rows('data/pairs.csv')
    pair_tuples = [(r, d) for r in range(25) for d in range(25) if r // 5 != d // 5]
    check(len(pair_rows) == 500, 'pair count')
    check([(int(row['receiver']), int(row['donor'])) for row in pair_rows] == pair_tuples, 'complete ordered pair set')
    receiver, donor = np.array(pair_tuples).T
    counterfactual = (donor // 5) * 5 + receiver % 5
    near([int(row['counterfactual']) for row in pair_rows], counterfactual, 'counterfactual ids')
    hashes = [sha(f'basis-audit-v1|{r}|{d}'.encode()) for r, d in pair_tuples]
    check([row['hash'] for row in pair_rows] == hashes, 'pair hashes')
    order = sorted(range(500), key=lambda i: (hashes[i], receiver[i] * 25 + donor[i]))
    test = np.zeros(500, dtype=bool)
    test[order[250:]] = True
    check(np.array_equal(test, np.array([row['split'] == 'test' for row in pair_rows])), 'fixed pair split')
    matched = receiver % 5 == donor % 5
    check(np.array_equal(matched, np.array([row['b_match'] == 'True' for row in pair_rows])), 'matched donor ids')

    basis_rows = csv_rows('data/bases.csv')
    archive = np.load(HERE / 'data' / 'transforms.npz')
    transforms = archive['matrices']
    check(transforms.shape == (26, 8, 8), 'transform count and shape')
    check(archive['names'].tolist() == [row['name'] for row in basis_rows], 'basis labels')
    expected = [np.eye(8)]
    rng = np.random.default_rng(99)
    permutation = rng.permutation(8)
    signs = rng.choice([-1., 1.], size=8)
    signed = np.eye(8)[:, np.argsort(permutation)] * signs[:, None]
    expected.append(signed)
    rotations = []
    for seed in range(100, 108):
        rotation, triangular = np.linalg.qr(np.random.default_rng(seed).normal(size=(8, 8)))
        rotation *= np.where(np.diag(triangular) < 0, -1., 1.)
        rotations.append(rotation)
    for condition in (1., 4., 16.):
        for rotation in rotations:
            expected.append(rotation * np.geomspace(condition ** -.5, condition ** .5, 8))
    near(transforms, np.array(expected), 'fixed transform generation')

    selected_rows = csv_rows('results/selected-masks.csv')
    operator_rows = csv_rows('results/operator-metrics.csv')
    control_rows = csv_rows('results/controls.csv')
    intact_rows = csv_rows('results/intact-predictions.csv')
    outcomes = csv_rows('results/seed-outcomes.csv')
    check(len(selected_rows) == 8 * 26, 'selected table count')
    check(len(operator_rows) == 8 * 26 * 4 * 2, 'operator table count')
    selected_index = {(int(row['seed']), int(row['basis_id'])): row for row in selected_rows}
    operator_index = {(int(row['seed']), int(row['basis_id']), row['method'], row['split']): row for row in operator_rows}
    control_index = {(int(row['seed']), int(row['basis_id'])): row for row in control_rows}
    mask_array = np.array([[index in mask for index in range(8)] for mask in MASKS])
    recomputed_selected = {}
    recomputed_operator = {}
    intact_accuracy = []
    max_transport = 0.
    max_diagonal_invariance = 0.
    max_permutation = 0.
    all_candidate_rows = 0
    native_projector = np.diag([1.] * 4 + [0.] * 4)

    for seed in range(8):
        weights = dict(np.load(HERE / 'models' / f'seed-{seed}.npz'))
        native = np.concatenate((weights['Ea'][states[:, 0]] + weights['ba'],
                                 weights['Eb'][states[:, 1]] + weights['bb']), axis=1)
        original = predict(native, weights, weights['W1'])
        target = original[counterfactual]
        semantic = labels[counterfactual]
        accuracy = np.mean(original.argmax(axis=1) == labels)
        intact_accuracy.append(float(accuracy))
        outcome = next(row for row in outcomes if int(row['seed']) == seed)
        near(float(outcome['accuracy']), float(accuracy), 'intact accuracy')
        near(float(outcome['final_loss']), float(-np.log(original[np.arange(25), labels]).mean()), 'intact loss')
        check((outcome['eligible'] == 'True') == (accuracy >= .99), 'eligibility')
        stored_intact = sorted([row for row in intact_rows if int(row['seed']) == seed], key=lambda row: int(row['input_id']))
        near([[float(row[f'p{j}']) for j in range(25)] for row in stored_intact], original, 'intact saved probabilities')
        native_patch = native[receiver].copy()
        native_patch[:, :4] = native[donor, :4]
        near(native_patch, native[counterfactual], 'native architecture intervention')
        candidates = csv_rows(f'results/candidate-scores-seed-{seed}.csv')
        all_candidate_rows += len(candidates)
        check(len(candidates) == 26 * 70 * 2, 'all candidate rows per seed')
        candidate_index = {(int(row['basis_id']), int(row['mask_id']), row['split']): row for row in candidates}
        check(len(candidate_index) == len(candidates), 'unique candidate identities')
        prediction_cache = {}

        for basis_id, transform in enumerate(transforms):
            inverse = np.linalg.inv(transform)
            first = inverse @ weights['W1']
            # Folded dense encoder is independent of the runner's z @ T path.
            encoder = np.zeros((10, 8))
            encoder[:5, :4], encoder[5:, 4:] = weights['Ea'], weights['Eb']
            offset = np.concatenate((weights['ba'], weights['bb']))
            x = np.concatenate((np.eye(5)[states[:, 0]], np.eye(5)[states[:, 1]]), axis=1)
            changed = x @ (encoder @ transform) + offset @ transform
            near(predict(changed, weights, first), original, 'intact coordinate invariance')
            transported = inverse @ native_projector @ transform
            image_basis, _ = np.linalg.qr(transform[:4, :].T)
            euclidean = image_basis @ image_basis.T
            near(transported @ transported, transported, 'transport idempotence')
            near(euclidean @ euclidean, euclidean, 'image idempotence')
            near(euclidean, euclidean.T, 'image symmetry')
            near(transported @ euclidean, transported, 'same row image')
            near(float(np.linalg.cond(transform)), float(basis_rows[basis_id]['condition']), 'condition number')
            base = changed[receiver]
            difference = changed[donor] - base
            transported_pred = predict(base + difference @ transported, weights, first)
            error = float(np.max(np.abs(transported_pred - target)))
            max_transport = max(max_transport, error)
            near(transported_pred, target, 'transported counterfactual probabilities')
            # Element replacement, rather than a diagonal matrix multiply.
            patched = np.where(mask_array[:, None, :], changed[donor][None, :, :], base[None, :, :])
            predictions = predict(patched, weights, first)
            prediction_cache[basis_id] = predictions
            discovery_errors = []
            for mask_id, mask in enumerate(MASKS):
                for split, keep in [('discovery', ~test), ('test', test)]:
                    stored = candidate_index[(basis_id, mask_id, split)]
                    check(stored['mask'] == ' '.join(map(str, mask)), 'candidate mask tuple')
                    current = metrics(predictions[mask_id], target, semantic, keep, matched)
                    for metric, value in current.items():
                        near(float(stored[metric]), float(value), 'candidate ' + metric)
                    if split == 'discovery':
                        discovery_errors.append(current['tv'])
            discovery_errors = np.array(discovery_errors)
            chosen = int(np.flatnonzero(discovery_errors <= discovery_errors.min() + 1e-12)[0])
            selected = selected_index[(seed, basis_id)]
            check(int(selected['mask_id']) == chosen, 'discovery-only mask selection')
            near(float(selected['discovery_minimum_tv']), float(discovery_errors.min()), 'discovery minimum')
            near(float(selected['discovery_tv']), float(discovery_errors[chosen]), 'chosen discovery error')
            current = metrics(predictions[chosen], target, semantic, test, matched)
            recomputed_selected[(seed, basis_id)] = current
            for metric, value in current.items():
                near(float(selected['test_' + metric]), float(value), 'selected test ' + metric)
            for method, operator in [('transported', transported), ('orthogonal_image', euclidean),
                                     ('empty', np.zeros((8, 8))), ('full', np.eye(8))]:
                prediction = predict(base + difference @ operator, weights, first)
                if method == 'orthogonal_image':
                    near(prediction[matched], target[matched], 'matched-b image projector control')
                for split, keep in [('discovery', ~test), ('test', test)]:
                    current = metrics(prediction, target, semantic, keep, matched)
                    recomputed_operator[(seed, basis_id, method, split)] = current
                    stored = operator_index[(seed, basis_id, method, split)]
                    for metric, value in current.items():
                        near(float(stored[metric]), float(value), 'operator ' + metric)
            controls = control_index[(seed, basis_id)]
            near(float(controls['operator_difference_fro']), float(np.linalg.norm(euclidean - transported)), 'operator distance')
            near(float(controls['transport_symmetry_residual']), float(np.max(np.abs(transported - transported.T))), 'oblique asymmetry')
            if basis_id == 1:
                mapping = np.argmax(np.abs(transform), axis=1)
                for mask_id, mask in enumerate(MASKS):
                    native_mask = tuple(np.flatnonzero(np.isin(mapping, mask)))
                    native_id = MASKS.index(native_mask)
                    difference_max = float(np.max(np.abs(predictions[mask_id] - prediction_cache[0][native_id])))
                    max_permutation = max(max_permutation, difference_max)
                    near(predictions[mask_id], prediction_cache[0][native_id], 'complete permutation mask family')
            if basis_id >= 10:
                equivalent_rotation = 2 + (basis_id - 2) % 8
                diagonal_error = float(np.max(np.abs(predictions - prediction_cache[equivalent_rotation])))
                max_diagonal_invariance = max(max_diagonal_invariance, diagonal_error)
                near(predictions, prediction_cache[equivalent_rotation], 'all-mask diagonal scaling invariance')

    summary = json.loads((HERE / 'results' / 'summary.json').read_text())
    for row in summary['per_seed']:
        seed, kind = row['seed'], row['kind']
        basis_ids = [int(item['basis_id']) for item in basis_rows if item['kind'] == kind]
        group = [recomputed_selected[(seed, index)] for index in basis_ids]
        tvs = np.array([item['tv'] for item in group])
        expected = dict(coordinate_tv_mean=tvs.mean(), coordinate_tv_min=tvs.min(), coordinate_tv_max=tvs.max(),
                        coordinate_accuracy_mean=np.mean([item['semantic_accuracy'] for item in group]),
                        coordinate_tv_change_from_identity=tvs.mean() - recomputed_selected[(seed, 0)]['tv'])
        for method in ('transported', 'orthogonal_image'):
            group = [recomputed_operator[(seed, index, method, 'test')] for index in basis_ids]
            expected[method + '_tv_mean'] = np.mean([item['tv'] for item in group])
            expected[method + '_accuracy_mean'] = np.mean([item['semantic_accuracy'] for item in group])
        for metric, value in expected.items():
            near(row[metric], float(value), 'per-seed ' + metric)
    for kind, metrics_summary in summary['across_seed'].items():
        for metric, stored in metrics_summary.items():
            values = np.array([row[metric] for row in summary['per_seed'] if row['kind'] == kind])
            near(stored['mean'], float(values.mean()), 'across-seed mean')
            near(stored['minimum'], float(values.min()), 'across-seed minimum')
            near(stored['maximum'], float(values.max()), 'across-seed maximum')
            check(stored['n'] == 8, 'independent seed count')
    receipt = dict(passed=True, trained_models=8, candidate_rows=all_candidate_rows,
                   selected_masks=len(selected_rows), operator_rows=len(operator_rows),
                   checks=CHECKS, numeric_values=NUMBERS, maximum_absolute_difference=MAX_ERROR,
                   max_transport_probability_difference=max_transport,
                   max_permutation_probability_difference=max_permutation,
                   max_diagonal_scaling_probability_difference=max_diagonal_invariance,
                   intact_accuracy=intact_accuracy, manifest_files=len(manifest), lock_commit=lock['commit'],
                   scope='All candidate masks and operators recomputed from saved weights with explicit element replacement; exact pair split, fixed transforms, complete controls, and primary summaries checked. Training was not repeated.')
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
