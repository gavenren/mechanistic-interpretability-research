"""Fixed-mask donor and edit-order checks. Importing this file runs no model."""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
OLD = STUDY.parent / '2026-10-13-inactive-variable-audit'
SOURCE_COMMIT = '168d952'
TOL = 1e-10
BATCH = 768
SEEDS = tuple(range(8))
MASKS = ((7, 8, 11), (1, 5, 6), (1, 2, 11), (0, 1, 2),
         (2, 3, 4), (2, 3, 5), (0, 2, 3), (2, 6, 10))
ROWS_PER_MODEL = 36864 + 2304 + 12288
SOURCE_HASHES = {
    'code/run.py': '555c08943b8e847a142a8a1d103f100673ac7ce3a964619c45849505a95f77b3',
    'data/inputs.csv': '10ef255c99d31ece166cb0e8ab6675e9e9b089cadda9d8ce65b7c9b670f2a06a',
    'data/pairs.csv': '92a0af4151f9745d312d83ab93e9f76fb3acc4b0e9cc74c33424c5082d7f817d',
    'results/selected-masks.csv': '34f592a9ca723fa3e86ce368a4cec5b073cb6bc51e3f66188ba8d28bd543c3ae',
    'results/manifest.json': '635208a65e203d1e71ffc4354f81bbe538c4c79ddc72611beecef631ae55983f',
    'models/seed-0.npz': 'e2a686a71424cef3c87418874f2a43ee96803bcbd7f45ffad4a0dcd7594cbefe',
    'models/seed-1.npz': '8aea6f6f7a61e25a19aec3db513dbf37a143f58a50ba64de5a1cd27ef718d1a1',
    'models/seed-2.npz': '48d48633dbea5d87ac931c6a3440c09258d71742d461356f1ba4720759de0002',
    'models/seed-3.npz': '05fbf0c680ff50f510eda83cd0d5144e4cc4c52e4e6ee17b493a7ddf8c662aa2',
    'models/seed-4.npz': '47724400c39252f198a066d3c2e6fd0477779088c8978b6b122d1025ef076380',
    'models/seed-5.npz': '832419cf76225a7b7373130ffed52cd110ab3b7a4be2d214a35f239c5ac2d297',
    'models/seed-6.npz': 'fd6e4fb5b333c152709fb8db9b695101abbc3fbfe4f5fe1fa0a6560f3c0adc64',
    'models/seed-7.npz': '59c13f3d419cc9746799e6c40571e252ce3478fdce81c23b7aa500fb61ecc321',
}
INPUT_FILES = [OLD / name for name in SOURCE_HASHES]
LEDGER_COLUMNS = {
    'donors': ['receiver', 'donor', 'required_input', 'split', 'same_a', 'active', 'b_match', 's_match', 'n_match'],
    'controls': ['receiver', 'required_input', 'split', 'same_a', 'active', 'operation'],
    'orders': ['receiver', 'first_donor', 'second_donor', 'first_required_input', 'second_required_input',
               'split', 'requested_a_values_equal', 'active', 'first_context', 'second_context', 'order'],
}
GROUP_FIELDS = {
    'donors': ['split', 'same_a', 'active', 'b_match', 's_match', 'n_match'],
    'controls': ['split', 'same_a', 'active', 'operation'],
    'orders': ['split', 'requested_a_values_equal', 'active', 'first_context', 'second_context', 'order'],
}


def empty_output(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Selector output directory is not empty')
    return out


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def save_npz(path, **values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)


def source_check():
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((OLD / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Selector source hash differs: ' + name)
    with (OLD / 'results/selected-masks.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['regime'] == 'unrestricted'
                and row['k'] == '3' and row['method'] == 'repair_only']
    if [(int(row['seed']), tuple(map(int, row['mask'].split()))) for row in rows] != list(enumerate(MASKS)):
        raise RuntimeError('Frozen selector masks differ')
    return {str((OLD / name).relative_to(ROOT)): value for name, value in SOURCE_HASHES.items()}


def state_id(a, b, s, n):
    return ((a * 4 + b) * 2 + s) * 6 + n


def inputs():
    states = np.array(list(itertools.product(range(4), range(4), range(2), range(6))), dtype=np.int64)
    x = np.concatenate([np.eye(size, dtype=np.float64)[states[:, j]]
                        for j, size in enumerate((4, 4, 2, 6))], axis=1)
    labels = np.where(states[:, 2] == 0, states[:, 0], states[:, 1])
    return states, x, labels


def ledgers():
    states, _, _ = inputs()
    # Split whole receiver/request blocks. Every donor context in a block stays
    # together. This split never depends on model outputs or selected effects.
    split = {}
    for s in (0, 1):
        for same in (False, True):
            blocks = [(r, a) for r in range(192) for a in range(4)
                      if states[r, 2] == s and (states[r, 0] == a) == same]
            blocks.sort(key=lambda pair: hashlib.sha256(f'selector-procedure-v1:{pair[0]}:{pair[1]}'.encode()).digest())
            for index, pair in enumerate(blocks):
                split[pair] = int(index >= len(blocks) // 2)
    donors, controls, orders = [], [], []
    for r, (a, b, s, n) in enumerate(states):
        for requested in range(4):
            cf = state_id(requested, b, s, n)
            for db, ds, dn in itertools.product(range(4), range(2), range(6)):
                d = state_id(requested, db, ds, dn)
                donors.append((r, d, cf, split[r, requested], int(a == requested), int(s == 0),
                               int(b == db), int(s == ds), int(n == dn)))
            for operation in range(3):  # 0=no change, 1=zero mask, 2=coherent input
                controls.append((r, cf, split[r, requested], int(a == requested), int(s == 0), operation))
            for compatible in (True, False):
                second = requested if compatible else (requested + 1) % 4
                cf2 = state_id(second, b, s, n)
                for first_context, second_context in itertools.product(range(2), repeat=2):
                    def donor(value, context):
                        return state_id(value, (b + context) % 4, s ^ context, (n + context) % 6)
                    d1, d2 = donor(requested, first_context), donor(second, second_context)
                    for order in range(2):
                        orders.append((r, d1, d2, cf, cf2, split[r, requested], int(compatible),
                                       int(s == 0), first_context, second_context, order))
    tables = dict(donors=np.array(donors, dtype=np.int64), controls=np.array(controls, dtype=np.int64),
                  orders=np.array(orders, dtype=np.int64))
    if [len(tables[k]) for k in ('donors', 'controls', 'orders')] != [36864, 2304, 12288]:
        raise RuntimeError('Unexpected selector ledger size')
    return tables


def affine_reference(x, w, b):
    # Separate columnwise reductions; no matrix multiplication is used here.
    return np.column_stack([np.sum(x * w[:, j], axis=1) + b[j] for j in range(w.shape[1])])


def softmax(x):
    z = np.exp(x - np.max(x, axis=1, keepdims=True))
    return z / np.sum(z, axis=1, keepdims=True)


def downstream(h, p, reference=False):
    affine = affine_reference if reference else lambda x, w, b: x @ w + b
    return softmax(affine(np.tanh(affine(h, p['W2'], p['b2'])), p['W3'], p['b3']))


def forward(x, p, reference=False):
    h = np.tanh(affine_reference(x, p['W1'], p['b1']) if reference else x @ p['W1'] + p['b1'])
    return h, downstream(h, p, reference)


def close(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Selector shape or finite-value check failed')
    error = float(np.max(np.abs(a - b)))
    if error > TOL:
        raise RuntimeError('Selector calculation replay failed')
    return error


def edit(kind, rows, h, mask, reference=False):
    mixed = h[rows[:, 0]].copy()
    if kind == 'donors':
        if reference:
            for coordinate in mask:
                mixed[:, coordinate] = h[rows[:, 1], coordinate]
        else:
            mixed[:, mask] = h[rows[:, 1]][:, mask]
    elif kind == 'controls':
        zero = rows[:, 5] == 1
        coherent = rows[:, 5] == 2
        for coordinate in mask:
            mixed[zero, coordinate] = 0
        mixed[coherent] = h[rows[coherent, 1]]
    else:
        order = rows[:, 10]
        if reference:
            # On the same mask the last write sets its requested coordinates.
            # This direct reference does not compose the two operations.
            last = np.where(order == 0, rows[:, 2], rows[:, 1])
            for coordinate in mask:
                mixed[:, coordinate] = h[last, coordinate]
        else:
            first = np.where(order == 0, rows[:, 1], rows[:, 2])
            second = np.where(order == 0, rows[:, 2], rows[:, 1])
            mixed[:, mask] = h[first][:, mask]
            mixed[:, mask] = h[second][:, mask]
    return mixed


def batch(out, name, kind, rows, h, hr, p, mask, guard):
    guard()
    mixed = edit(kind, rows, h, mask)
    probabilities = downstream(mixed, p)
    path = out / (name + '.npz')
    save_npz(path, rows=rows, probabilities=probabilities, edited_mask=mixed[:, mask])
    guard()
    with np.load(path, allow_pickle=False) as saved:
        reference_state = edit(kind, saved['rows'], hr, mask, reference=True)
        state_error = close(saved['edited_mask'], reference_state[:, mask])
        probability_error = close(saved['probabilities'], downstream(reference_state, p, reference=True))
    audit = dict(passed=True, rows=len(rows), maximum_mask_replay_error=state_error,
                 maximum_probability_replay_error=probability_error, absolute_tolerance=TOL)
    write_json(out / (name + '-audit.json'), audit)
    return probabilities, mixed, audit


def prepare(out, guard):
    started = time.monotonic()
    out = empty_output(out)
    guard(); sources = source_check()
    rng = np.random.default_rng(2026102601)
    p = {f'W{i}': rng.normal(0, .1, (ni, no)) for i, (ni, no) in enumerate(((16, 12), (12, 24), (24, 4)), 1)}
    p.update({f'b{i}': np.zeros(no) for i, no in enumerate((12, 24, 4), 1)})
    x = rng.normal(size=(192, 16))  # Random dummy inputs, no task model weights.
    h, prob = forward(x, p); hr, probr = forward(x, p, True)
    close(h, hr); close(prob, probr)
    tables = ledgers()
    begun = time.monotonic()
    audits = []
    for kind in ('donors', 'controls', 'orders'):
        _, _, audit = batch(out, 'dummy-' + kind, kind, tables[kind][:BATCH], h, hr, p, MASKS[0], guard)
        audits.append(audit)
    elapsed = time.monotonic() - begun
    # All three operation types, save/read, reference replay and guard calls are
    # timed once. No choice between repeated timing methods is permitted.
    projected = 4 * (elapsed / (3 * BATCH) * ROWS_PER_MODEL * len(SEEDS) + 60)
    result = dict(passed=True, scientific_work=False, dummy_rows=3 * BATCH,
                  measured_batch_seconds=elapsed, projected_seconds=projected,
                  projected_run_seconds=projected, safety_factor=4,
                  fixed_setup_and_summary_allowance_seconds=60, total_output_rows=ROWS_PER_MODEL * len(SEEDS),
                  tolerance=TOL, sources=sources, audits=audits, seconds=time.monotonic() - started)
    guard(); write_json(out / 'summary.json', result)
    return result


def run(out, guard):
    started = time.monotonic()
    out = empty_output(out)
    guard(); sources = source_check()
    tables = ledgers()
    # The complete fixed ledger is saved before any trained weights are loaded.
    save_npz(out / 'pair-ledgers.npz', **tables)
    write_json(out / 'ledger-schema.json', dict(columns=LEDGER_COLUMNS, group_fields=GROUP_FIELDS,
               split={'0': 'development', '1': 'test'}, operation={'0': 'no change', '1': 'zero mask', '2': 'coherent input'},
               context={'0': 'receiver context', '1': 'all nuisance inputs changed by the fixed rule'},
               order={'0': 'first request then second request', '1': 'second request then first request'}))
    states, x, labels = inputs()
    with (OLD / 'data/inputs.csv').open() as stream:
        saved_states = list(csv.DictReader(stream))
    if not np.array_equal(states, np.array([[int(row[key]) for key in ('a', 'b', 's', 'n')] for row in saved_states])):
        raise RuntimeError('Generated inputs differ from the fixed input file')
    records, controls = [], []
    rows_done = 0
    for seed in SEEDS:
        guard()
        with np.load(OLD / f'models/seed-{seed}.npz', allow_pickle=False) as saved:
            p = {key: saved[key].copy() for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')}
        if any(value.dtype != np.float64 or not np.isfinite(value).all() for value in p.values()):
            raise RuntimeError('Expected finite float64 selector weights')
        h, prob = forward(x, p); hr, probr = forward(x, p, True)
        errors = [close(h, hr), close(prob, probr)]
        accuracy = float(np.mean(prob.argmax(1) == labels))
        if accuracy < .99:
            raise RuntimeError('Saved selector model failed its ordinary task rule')
        mask = MASKS[seed]
        save_npz(out / f'intact-seed-{seed}.npz', states=states, labels=labels, probabilities=prob,
                 reference_probabilities=probr, hidden=h, mask=np.array(mask))
        accumulators = {}
        for kind in ('donors', 'controls', 'orders'):
            table = tables[kind]
            for start in range(0, len(table), BATCH):
                rows = table[start:start + BATCH]
                name = f'{kind}-seed-{seed}-row-{start:05d}'
                values, mixed, audit = batch(out, name, kind, rows, h, hr, p, mask, guard)
                errors.extend([audit['maximum_mask_replay_error'], audit['maximum_probability_replay_error']])
                if kind == 'donors':
                    target = rows[:, 2]
                    group_columns = (3, 4, 5, 6, 7, 8)
                    identity = rows[:, 0] == rows[:, 1]
                    if np.any(identity): errors.append(close(values[identity], prob[rows[identity, 0]]))
                elif kind == 'controls':
                    target = rows[:, 1]
                    group_columns = (2, 3, 4, 5)
                    no_op = rows[:, 5] == 0
                    coherent = rows[:, 5] == 2
                    if np.any(no_op): errors.append(close(values[no_op], prob[rows[no_op, 0]]))
                    if np.any(coherent): errors.append(close(values[coherent], prob[target[coherent]]))
                else:
                    target = rows[:, 3]
                    group_columns = (5, 6, 7, 8, 9, 10)
                measures = dict(tv_target=np.abs(values - prob[target]).sum(1) / 2,
                                correct_target=(values.argmax(1) == labels[target]).astype(float),
                                changed_answer=(values.argmax(1) != prob[rows[:, 0]].argmax(1)).astype(float))
                if kind == 'orders':
                    if not np.array_equal(rows[::2, :10], rows[1::2, :10]):
                        raise RuntimeError('Opposite-order pair ledger differs')
                    request1 = h[rows[:, 1]][:, mask]
                    request2 = h[rows[:, 2]][:, mask]
                    measures.update(tv_second_target=np.abs(values - prob[rows[:, 4]]).sum(1) / 2,
                                    correct_second_target=(values.argmax(1) == labels[rows[:, 4]]).astype(float),
                                    first_request_met=(np.max(np.abs(mixed[:, mask] - request1), axis=1) <= TOL).astype(float),
                                    second_request_met=(np.max(np.abs(mixed[:, mask] - request2), axis=1) <= TOL).astype(float),
                                    coordinate_requests_compatible=(np.max(np.abs(request1 - request2), axis=1) <= TOL).astype(float),
                                    order_probability_distance=np.repeat(np.abs(values[::2] - values[1::2]).sum(1) / 2, 2),
                                    order_answer_disagreement=np.repeat(values[::2].argmax(1) != values[1::2].argmax(1), 2).astype(float))
                    last_met = np.where(rows[:, 10] == 0, measures['second_request_met'], measures['first_request_met'])
                    if not np.all(last_met == 1): raise RuntimeError('Last-write control failed')
                for group in np.unique(rows[:, group_columns], axis=0):
                    take = np.all(rows[:, group_columns] == group, axis=1)
                    key = (kind,) + tuple(int(value) for value in group)
                    entry = accumulators.setdefault(key, dict(n=0, **{key: 0. for key in measures}))
                    entry['n'] += int(take.sum())
                    for measure, values_for_measure in measures.items():
                        entry[measure] += float(values_for_measure[take].sum())
                rows_done += len(rows)
        seed_rows = []
        for group, values in sorted(accumulators.items()):
            seed_rows.append(dict(seed=seed, kind=group[0], group=list(group[1:]), n=values['n'],
                                  **{key: value / values['n'] for key, value in values.items() if key != 'n'}))
        control = dict(seed=seed, intact_accuracy=accuracy, maximum_replay_error=max(errors),
                       mask=list(mask), passed=True)
        guard(); write_json(out / f'summary-seed-{seed}.json', dict(controls=control, groups=seed_rows))
        records.extend(seed_rows); controls.append(control)
    guard(); source_check()
    result = dict(status='complete', passed=True, meaning='Measurement controls passed; this is not an explanation acceptance rule.',
                  rows=rows_done, models=8, source_commit=SOURCE_COMMIT, sources=sources,
                  tolerance=TOL, controls=controls, group_fields=GROUP_FIELDS,
                  groups=records, seconds=time.monotonic() - started)
    if rows_done != ROWS_PER_MODEL * len(SEEDS): raise RuntimeError('Incomplete selector ledger')
    write_json(out / 'summary.json', result)
    return result


def mock_check():
    """Small index/arithmetic controls only. No saved model is loaded."""
    tables = ledgers()
    states, _, _ = inputs()
    donors = tables['donors']
    if not np.array_equal(states[donors[:, 1], 0], states[donors[:, 2], 0]):
        raise AssertionError('Donor does not have the required variable value')
    for split in (0, 1):
        if int(np.sum(donors[:, 3] == split)) != 18432:
            raise AssertionError('Unexpected split count')
    for receiver, target in np.unique(donors[:, (0, 2)], axis=0):
        rows = donors[(donors[:, 0] == receiver) & (donors[:, 2] == target)]
        if len(rows) != 48 or len(np.unique(rows[:, 3])) != 1:
            raise AssertionError('A donor comparison block was split or reduced')
    h = np.arange(192 * 12, dtype=np.float64).reshape(192, 12)
    mask = MASKS[0]
    order_rows = tables['orders'][:64]
    first = edit('orders', order_rows, h, mask)
    reference = edit('orders', order_rows, h, mask, True)
    if not np.array_equal(first, reference):
        raise AssertionError('Order implementation differs from last-write reference')
    untouched = sorted(set(range(12)) - set(mask))
    if not np.array_equal(first[:, untouched], h[order_rows[:, 0]][:, untouched]):
        raise AssertionError('An unselected coordinate changed')
    x = np.arange(12, dtype=np.float64).reshape(3, 4) / 7
    w = np.arange(8, dtype=np.float64).reshape(4, 2) / 11
    close(x @ w + np.array([.2, -.3]), affine_reference(x, w, np.array([.2, -.3])))
    return dict(passed=True, saved_models_loaded=False, scientific_outputs=False,
                donor_rows=len(donors), total_rows_per_model=sum(len(v) for v in tables.values()))
