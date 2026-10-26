"""Fixed finite subset search. Preparation never opens trained model weights."""
from __future__ import annotations
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
PREVIOUS = STUDY.parent / '2026-10-26-edit-procedure'
ORIGINAL = STUDY.parent / '2026-10-13-inactive-variable-audit'
SOURCE_COMMIT = '352f1551dde4be89c0e603402a54b7adc534b604'
SELECTOR = PREVIOUS / 'code/selector.py'
INPUT_FILES = [SELECTOR] + [ORIGINAL / n for n in (
    'code/run.py', 'data/inputs.csv', 'data/pairs.csv', 'results/selected-masks.csv',
    'results/manifest.json')] + [ORIGINAL / f'models/seed-{seed}.npz' for seed in range(8)]
TOL = 1e-10
BLOCK = 16
SCORE_NAMES = ('signed_class_zero', 'mean_absolute_class_zero', 'target_margin_gain',
               'target_probability_gain', 'answer_accuracy')
ROW_FIELDS = ('receiver', 'donor', 'required_input_2a', 'split', 'same_a', 'active',
              'b_match', 's_match', 'n_match', 'context')
_SELECTOR_MODULE = None


def selector():
    global _SELECTOR_MODULE
    if _SELECTOR_MODULE is None:
        spec = importlib.util.spec_from_file_location('locked_selector_source', SELECTOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SELECTOR_MODULE = module
    return _SELECTOR_MODULE


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def source_check():
    relative = SELECTOR.relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', 'show', SOURCE_COMMIT + ':' + relative], cwd=ROOT) != SELECTOR.read_bytes():
        raise RuntimeError('The published Stage 2A selector source differs')
    sources = selector().source_check()
    sources[relative] = digest(SELECTOR)
    return sources


def output(path):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()): raise FileExistsError('Subset output directory is not empty')
    return path


def jsave(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def nsave(path, **values):
    with Path(path).open('xb') as stream: np.savez_compressed(stream, **values)


def close(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Subset shape or finite-value check failed')
    error = float(np.max(np.abs(a - b)))
    if error > TOL: raise RuntimeError('Subset arithmetic replay failed')
    return error


def pairs():
    old = selector().ledgers()['donors']
    states, _, _ = selector().inputs()
    r, d = states[old[:, 0]], states[old[:, 1]]
    matched = np.all(r[:, 1:] == d[:, 1:], axis=1)
    changed = ((d[:, 1] == (r[:, 1] + 1) % 4) & (d[:, 2] == (r[:, 2] ^ 1))
               & (d[:, 3] == (r[:, 3] + 1) % 6))
    take = matched | changed
    ids = np.flatnonzero(take).astype(np.int64)
    rows = np.column_stack((old[take], changed[take].astype(np.int64)))
    if len(rows) != 1536 or any(np.sum(rows[:, 3] == split) != 768 for split in (0, 1)):
        raise RuntimeError('Unexpected fixed pair counts')
    return rows, ids


def visit_orders():
    first, last = tuple(range(8)), tuple(reversed(range(8)))
    rest = [p for p in itertools.permutations(range(8)) if p not in (first, last)]
    rest.sort(key=lambda p: hashlib.sha256(('search-procedure-v1:order:' + ','.join(map(str, p))).encode()).digest())
    return [first, last] + rest[:30]


def bank_order():
    return sorted(range(1, 255), key=lambda m: hashlib.sha256(f'search-procedure-v1:subset:{m}'.encode()).digest())


def readout(h, p, reference=False):
    affine = selector().affine_reference if reference else lambda x, w, b: x @ w + b
    logits = affine(np.tanh(affine(h, p['W2'], p['b2'])), p['W3'], p['b3'])
    return logits, selector().softmax(logits)


def scalar_readout(h, p):
    hidden = [math.tanh(math.fsum(float(h[i]) * float(p['W2'][i, j]) for i in range(12))
                        + float(p['b2'][j])) for j in range(24)]
    logits = np.array([math.fsum(hidden[i] * float(p['W3'][i, j]) for i in range(24))
                       + float(p['b3'][j]) for j in range(4)])
    ex = [math.exp(float(value - max(logits))) for value in logits]
    return logits, np.array([value / math.fsum(ex) for value in ex])


def load_model(seed):
    with np.load(ORIGINAL / f'models/seed-{seed}.npz', allow_pickle=False) as saved:
        p = {key: saved[key].copy() for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')}
    if any(value.dtype != np.float64 or not np.isfinite(value).all() for value in p.values()):
        raise RuntimeError('Expected finite float64 weights')
    _, x, labels = selector().inputs()
    h, intact = selector().forward(x, p)
    hr, intact_reference = selector().forward(x, p, True)
    close(h, hr); close(intact, intact_reference)
    if np.mean(intact.argmax(1) == labels) < .99:
        raise RuntimeError('A selector failed ordinary task ability')
    return p, h, hr, intact, labels


def subset_states(h, rows, panel, ids, reference=False):
    r, d = rows[:, 0], rows[:, 1]
    values = np.broadcast_to(h[r], (len(ids), len(rows), 12)).copy()
    if reference:
        for index, subset in enumerate(ids):
            for bit, coordinate in enumerate(panel):
                if not int(subset) & (1 << bit): values[index, :, coordinate] = h[d, coordinate]
    else:
        for bit, coordinate in enumerate(panel):
            keep = (np.asarray(ids) & (1 << bit)) != 0
            values[:, :, coordinate] = np.where(keep[:, None], h[r, coordinate], h[d, coordinate])
    return values


def evaluate(out, rows, row_ids, panel, p, h, hr, guard, ids=tuple(range(256))):
    out = output(out)
    logits_all, probabilities_all, audits = [], [], []
    for start in range(0, len(ids), BLOCK):
        guard()
        block_ids = np.asarray(ids[start:start + BLOCK], dtype=np.int64)
        mixed = subset_states(h, rows, panel, block_ids)
        logits, probabilities = readout(mixed.reshape(-1, 12), p)
        logits = logits.reshape(len(block_ids), len(rows), 4)
        probabilities = probabilities.reshape(logits.shape)
        path = out / f'block-{start:03d}.npz'
        nsave(path, subset_ids=block_ids, row_ids=row_ids, rows=rows, logits=logits, probabilities=probabilities)
        guard()
        with np.load(path, allow_pickle=False) as saved:
            reference_state = subset_states(hr, saved['rows'], panel, saved['subset_ids'], True)
            reference_logits, reference_prob = readout(reference_state.reshape(-1, 12), p, True)
            error = max(close(saved['logits'], reference_logits.reshape(logits.shape)),
                        close(saved['probabilities'], reference_prob.reshape(logits.shape)))
            scalar_error = 0.
            # Four fixed entries per saved block also use scalar math.fsum and
            # math.exp. This is an extra check on the columnwise replay.
            for model_index, row_index in ((0, 0), (0, len(rows) - 1), (len(block_ids) - 1, 0), (len(block_ids) - 1, len(rows) - 1)):
                sl, sp = scalar_readout(reference_state[model_index, row_index], p)
                scalar_error = max(scalar_error, close(saved['logits'][model_index, row_index], sl),
                                   close(saved['probabilities'][model_index, row_index], sp))
        audit = dict(passed=True, probability_rows=len(block_ids) * len(rows), maximum_replay_error=error,
                     maximum_scalar_anchor_error=scalar_error, scalar_anchors=4, tolerance=TOL)
        jsave(out / f'block-{start:03d}-audit.json', audit)
        audits.append(audit); logits_all.append(logits); probabilities_all.append(probabilities)
    return np.concatenate(logits_all), np.concatenate(probabilities_all), audits


def choose_panel(out, rows, p, h, hr, intact, guard):
    guard()
    r, d = rows[:, 0], rows[:, 1]
    all_logits, all_prob, ref_logits_all, ref_prob = [], [], [], []
    for neuron in range(12):
        guard()
        mixed = h[r].copy(); mixed[:, neuron] = h[d, neuron]
        logits, prob = readout(mixed, p)
        reference = hr[r].copy(); reference[:, neuron] = hr[d, neuron]
        ref_logits, rp = readout(reference, p, True)
        close(logits, ref_logits); close(prob, rp)
        all_logits.append(logits); all_prob.append(prob); ref_logits_all.append(ref_logits); ref_prob.append(rp)
    path = out / 'single-deletions.npz'
    nsave(path, rows=rows, logits=np.array(all_logits), probabilities=np.array(all_prob))
    with np.load(path, allow_pickle=False) as saved:
        close(saved['logits'], np.array(ref_logits_all))
        close(saved['probabilities'], np.array(ref_prob))
        effects = np.abs(saved['probabilities'] - intact[r][None, :, :]).sum(2).mean(1) / 2
    # The ranking orders the panel. Mask bits refer to this published order.
    panel = sorted(range(12), key=lambda neuron: (-float(effects[neuron]), neuron))[:8]
    jsave(out / 'panel.json', dict(panel=panel, single_deletion_mean_tv=effects.tolist(),
                                  selection_split='development', ties='Exact score ties use neuron number'))
    return panel


def metrics(logits, probabilities, rows, intact, labels):
    r = rows[:, 0]; target = labels[r]; foil = (target + 1) % 4
    at = np.arange(len(rows))
    probability_gain = probabilities[:, at, target] - probabilities[0, at, target]
    margin = logits[:, at, target] - logits[:, at, foil]
    margin_gain = margin - margin[0]
    signed = probabilities[:, :, 0] - probabilities[0, :, 0]
    accuracy = probabilities.argmax(2) == target
    tv = np.abs(probabilities - intact[r][None, :, :]).sum(2) / 2
    scores = np.column_stack((signed.mean(1), np.abs(signed).mean(1), margin_gain.mean(1),
                              probability_gain.mean(1), accuracy.mean(1)))
    gate_accuracy = np.mean(probabilities.argmax(2) == intact[r].argmax(1), axis=1)
    gate_tv = tv.mean(1)
    passed = (gate_accuracy >= .99) & (gate_tv <= .01)
    return scores, passed, gate_accuracy, gate_tv, probability_gain, tv, accuracy


def freeze_choices(scores):
    candidates = [mask for mask in range(256) if mask.bit_count() == 4]
    return {name: min(candidates, key=lambda mask: (-float(scores[mask, index]), mask))
            for index, name in enumerate(SCORE_NAMES)}


def search(passed):
    valid = [mask for mask in range(256) if bool(passed[mask])]
    if 255 not in valid: raise RuntimeError('The full panel failed its no-change gate')
    minimum = min(mask.bit_count() for mask in valid)
    minima = [mask for mask in valid if mask.bit_count() == minimum]
    paths = []
    for order in visit_orders():
        current = 255; steps = []
        for bit in order:
            proposed = current & ~(1 << bit)
            accepted = bool(passed[proposed])
            steps.append(dict(before=current, removed_bit=bit, proposal=proposed, accepted=accepted))
            if accepted: current = proposed
        further = [bit for bit in range(8) if current & (1 << bit) and passed[current & ~(1 << bit)]]
        # Replay the visited state sequence using sets, independently of bit edits.
        kept = set(range(8))
        for step in steps:
            proposed_set = kept - {step['removed_bit']}
            proposal = sum(2**bit for bit in proposed_set)
            if proposal != step['proposal']: raise RuntimeError('Greedy path replay failed')
            if passed[proposal]: kept = proposed_set
        if sum(2**bit for bit in kept) != current: raise RuntimeError('Greedy endpoint replay failed')
        paths.append(dict(order=list(order), retained_mask=current, retained_size=current.bit_count(),
                          excess_size=current.bit_count() - minimum,
                          further_single_deletions=further, steps=steps))
    return dict(minimum_size=minimum, minimum_masks=minima, passing_masks=valid, greedy=paths,
                all_greedy_orders_find_minimum=all(row['excess_size'] == 0 for row in paths))


def summarize(out, rows, logits, probabilities, intact, labels, selected=None):
    scores, passed, accuracy, tvmean, gains, tv, correct = metrics(logits, probabilities, rows, intact, labels)
    # Compare aggregate gate decisions with a scalar reduction over saved values.
    scalar_accuracy = np.array([math.fsum(float(x) for x in (row.argmax(1) == intact[rows[:, 0]].argmax(1))) / len(rows)
                                for row in probabilities])
    scalar_tv = np.array([math.fsum(float(x) for x in row) / len(rows) for row in tv])
    close(accuracy, scalar_accuracy); close(tvmean, scalar_tv)
    if not np.array_equal(passed, (scalar_accuracy >= .99) & (scalar_tv <= .01)):
        raise RuntimeError('A gate depends on reduction order')
    groups = []
    for same, active, context in itertools.product(range(2), repeat=3):
        take = (rows[:, 4] == same) & (rows[:, 5] == active) & (rows[:, 9] == context)
        if not np.any(take): raise RuntimeError('A required context group is empty')
        groups.append(dict(same_a=same, active=active, context=context, n=int(take.sum()),
                           tv=tv[:, take].mean(1).tolist(), accuracy=correct[:, take].mean(1).tolist(),
                           scores=metrics(logits[:, take], probabilities[:, take], rows[take], intact, labels)[0].tolist()))
    result = dict(gate_accuracy=accuracy.tolist(), gate_mean_tv=tvmean.tolist(), passed_masks=passed.tolist(),
                  score_names=list(SCORE_NAMES), scores=scores.tolist(), groups=groups, search=search(passed))
    if selected is not None:
        result['development_selected_candidates'] = {name: dict(mask=int(mask), scores=scores[mask].tolist(),
                                                               passed=bool(passed[mask]), accuracy=float(accuracy[mask]),
                                                               mean_tv=float(tvmean[mask])) for name, mask in selected.items()}
    jsave(out / 'summary.json', result)
    return result, scores, gains


def manifest(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != 'calibration-manifest.json'}


def prepare(out, guard):
    started = time.monotonic(); out = output(out)
    guard(); sources = source_check()
    all_rows, all_ids = pairs(); take = all_rows[:, 3] == 0
    rows, row_ids = all_rows[take], all_ids[take]
    rng = np.random.default_rng(2026102602)
    p = {f'W{i}': rng.normal(0, .1, (ni, no)) for i, (ni, no) in enumerate(((16, 12), (12, 24), (24, 4)), 1)}
    p.update({f'b{i}': np.zeros(no) for i, no in enumerate((12, 24, 4), 1)})
    x = rng.normal(size=(192, 16))
    h, intact = selector().forward(x, p); hr, _ = selector().forward(x, p, True)
    close(h, hr)
    begun = time.monotonic()
    panel = choose_panel(out, rows, p, h, hr, intact, guard)
    panel_seconds = time.monotonic() - begun
    begun = time.monotonic()
    evaluate(out / 'dummy-subsets', rows, row_ids, panel, p, h, hr, guard, ids=tuple(range(15)) + (255,))
    subset_seconds = time.monotonic() - begun
    # Includes both development and final model phases, saved replay, and a
    # fixed two-minute allowance for summaries, banks, locks and receipts.
    projected = 4 * (panel_seconds * 8 + subset_seconds * (256 / 16) * 8 * 2 + 120)
    result = dict(passed=True, scientific_work=False, projected_seconds=projected, safety_factor=4,
                  panel_benchmark_seconds=panel_seconds, subset_benchmark_seconds=subset_seconds,
                  total_subset_output_rows=3145728, single_deletion_output_rows=73728,
                  fixed_other_work_allowance_seconds=120, sources=sources, seconds=time.monotonic() - started)
    guard(); jsave(out / 'summary.json', result)
    return result


def calibrate(out, guard):
    started = time.monotonic(); out = output(out)
    guard(); sources = source_check()
    all_rows, all_ids = pairs(); take = all_rows[:, 3] == 0
    rows, row_ids = all_rows[take], all_ids[take]
    nsave(out / 'development-pairs.npz', rows=rows, row_ids=row_ids)
    records = []
    for seed in range(8):
        guard(); p, h, hr, intact, labels = load_model(seed)
        folder = output(out / f'development-seed-{seed}')
        panel = choose_panel(folder, rows, p, h, hr, intact, guard)
        logits, prob, audits = evaluate(folder / 'subsets', rows, row_ids, panel, p, h, hr, guard)
        close(prob[255], intact[rows[:, 0]])
        summary, scores, gains = summarize(folder, rows, logits, prob, intact, labels)
        choices = freeze_choices(scores)
        order = np.array(bank_order(), dtype=np.int64)
        effects = gains[order].T.copy(); mean = effects.mean(0)
        centered = effects - mean
        # Remove the floating-point mean residual once. A constant column has
        # exactly zero null effect and variance, even if its mean rounded.
        centered -= centered.mean(0)
        constant = np.all(effects == effects[0], axis=0)
        centered[:, constant] = 0.
        variance = np.mean(centered * centered, axis=0)
        guard(); nsave(out / f'bank-seed-{seed}.npz', effects=effects, centered_effects=centered,
                       mean=mean, variance=variance, subset_ids=order, row_ids=row_ids, rows=rows)
        record = dict(seed=seed, panel=panel, score_choices=choices,
                      bank=f'bank-seed-{seed}.npz', summary=f'development-seed-{seed}/summary.json',
                      bank_sha256=digest(out / f'bank-seed-{seed}.npz'),
                      maximum_replay_error=max(max(a['maximum_replay_error'], a['maximum_scalar_anchor_error']) for a in audits))
        jsave(out / f'frozen-seed-{seed}.json', record); records.append(record)
    guard(); source_check()
    result = dict(passed=True, status='complete', phase='development only', models=records,
                  source_commit=SOURCE_COMMIT, sources=sources, row_fields=list(ROW_FIELDS),
                  score_names=list(SCORE_NAMES), rows_per_seed=768, subset_ids=list(range(256)),
                  seconds=time.monotonic() - started)
    jsave(out / 'calibration.json', result)
    jsave(out / 'calibration-manifest.json', manifest(out))
    return result


def verify_calibration(path):
    path = Path(path)
    fixed = json.loads((path / 'calibration-manifest.json').read_text())
    if fixed != manifest(path): raise RuntimeError('Calibration files differ from their manifest')
    record = json.loads((path / 'calibration.json').read_text())
    if record['status'] != 'complete' or not record['passed'] or [m['seed'] for m in record['models']] != list(range(8)):
        raise RuntimeError('Calibration is incomplete')
    for model in record['models']:
        if len(set(model['panel'])) != 8 or any(n not in range(12) for n in model['panel']):
            raise RuntimeError('Invalid frozen panel')
        scores = np.array(json.loads((path / model['summary']).read_text())['scores'])
        if model['score_choices'] != freeze_choices(scores): raise RuntimeError('Frozen score choices differ')
    return record


def run(out, guard, calibration_path):
    started = time.monotonic(); out = output(out)
    guard(); sources = source_check(); calibration = verify_calibration(calibration_path)
    if calibration['sources'] != sources: raise RuntimeError('Scientific sources differ from calibration')
    all_rows, all_ids = pairs(); take = all_rows[:, 3] == 1
    rows, row_ids = all_rows[take], all_ids[take]
    nsave(out / 'test-pairs.npz', rows=rows, row_ids=row_ids)
    records = []
    for model in calibration['models']:
        seed = model['seed']; guard()
        p, h, hr, intact, labels = load_model(seed)
        folder = output(out / f'test-seed-{seed}')
        logits, prob, audits = evaluate(folder / 'subsets', rows, row_ids, model['panel'], p, h, hr, guard)
        close(prob[255], intact[rows[:, 0]])
        summary, _, _ = summarize(folder, rows, logits, prob, intact, labels, model['score_choices'])
        records.append(dict(seed=seed, panel=model['panel'], minimum_size=summary['search']['minimum_size'],
                            minimum_masks=summary['search']['minimum_masks'],
                            greedy_minimum_claim_passed=summary['search']['all_greedy_orders_find_minimum'],
                            score_choices=model['score_choices'], selected_results=summary['development_selected_candidates'],
                            maximum_replay_error=max(max(a['maximum_replay_error'], a['maximum_scalar_anchor_error']) for a in audits)))
    guard(); source_check(); verify_calibration(calibration_path)
    result = dict(passed=True, status='complete', meaning='Calculation controls passed; search claims have separate recorded outcomes.',
                  models=records, source_commit=SOURCE_COMMIT, sources=sources,
                  calibration_manifest_sha256=digest(Path(calibration_path) / 'calibration-manifest.json'),
                  rows_per_seed=768, subset_output_rows=1572864, seconds=time.monotonic() - started)
    jsave(out / 'summary.json', result)
    return result


def mock_check():
    rows, ids = pairs()
    if len(np.unique(ids)) != 1536: raise AssertionError('Duplicate pair identifiers')
    if len(set(visit_orders())) != 32: raise AssertionError('Visit orders are not distinct')
    if sorted(bank_order()) != list(range(1, 255)): raise AssertionError('Bank order differs')
    # Constructed OR of a singleton and a seven-part route: minimum is one,
    # but visiting the singleton first leaves the larger route.
    passed = np.array([bool(mask & 1) or (mask & 254) == 254 for mask in range(256)])
    result = search(passed)
    if result['minimum_size'] != 1 or result['greedy'][0]['retained_size'] != 7:
        raise AssertionError('Constructed greedy control failed')
    return dict(passed=True, scientific_work=False, models_loaded=False, pair_rows=len(rows))
