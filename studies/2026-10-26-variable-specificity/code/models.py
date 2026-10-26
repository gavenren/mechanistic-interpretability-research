"""Fresh fixed-recipe selector models and a fixed three-way intervention split.

Importing this module does not train or evaluate a model. The controller owns
usage permits, resource limits, the public lock, and the complete stage budget.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
OLD = STUDY.parent / '2026-10-13-inactive-variable-audit'
SOURCE_COMMIT = '352f1551dde4be89c0e603402a54b7adc534b604'
INPUT_FILES = [OLD / n for n in ('code/run.py', 'analysis-plan.md', 'data/inputs.csv', 'requirements.txt')]
SEEDS = tuple(range(91001, 91009))
STEPS = 4000
LR = .02
TOL = 1e-10
SHAPES = {'W1': (16, 12), 'b1': (12,), 'W2': (12, 24), 'b2': (24,), 'W3': (24, 4), 'b3': (4,)}
SPLITS = {'learning': 0, 'selection': 1, 'test': 2}


def output(path):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()): raise FileExistsError('Model output directory is not empty')
    return path


def jsave(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def nsave(path, **values):
    with Path(path).open('xb') as stream: np.savez_compressed(stream, **values)


def source_check():
    sources = {}
    for path in INPUT_FILES:
        relative = path.relative_to(ROOT).as_posix()
        local = path.read_bytes()
        if subprocess.check_output(['git', 'show', SOURCE_COMMIT + ':' + relative], cwd=ROOT) != local:
            raise RuntimeError('Original selector source differs: ' + relative)
        sources[relative] = hashlib.sha256(local).hexdigest()
    return sources


def inputs():
    states = np.array(list(itertools.product(range(4), range(4), range(2), range(6))), dtype=np.int64)
    x = np.concatenate([np.eye(size, dtype=np.float64)[states[:, j]]
                        for j, size in enumerate((4, 4, 2, 6))], axis=1)
    labels = np.where(states[:, 2] == 0, states[:, 0], states[:, 1])
    return states, x, labels


def state_id(a, b, s, n):
    return ((a * 4 + b) * 2 + s) * 6 + n


def pairs():
    states, _, _ = inputs()
    block_split = {}
    for s in (0, 1):
        for same in (False, True):
            blocks = [(r, a) for r in range(192) for a in range(4)
                      if states[r, 2] == s and (states[r, 0] == a) == same]
            blocks.sort(key=lambda pair: hashlib.sha256(f'selector-specificity-v1:{pair[0]}:{pair[1]}'.encode()).digest())
            for index, block in enumerate(blocks):
                block_split[block] = 0 if index < len(blocks) // 2 else (1 if index < 3 * len(blocks) // 4 else 2)
    r, d = np.indices((192, 192)); r, d = r.ravel(), d.ravel()
    cf = state_id(states[d, 0], states[r, 1], states[r, 2], states[r, 3])
    split = np.array([block_split[int(ri), int(states[di, 0])] for ri, di in zip(r, d)], dtype=np.int64)
    result = dict(r=r, d=d, cf=cf, split=split, active=states[r, 2] == 0,
                  same_value=states[r, 0] == states[d, 0])
    if [int(np.sum(split == k)) for k in range(3)] != [18432, 9216, 9216]:
        raise RuntimeError('Unexpected intervention split counts')
    return result


def split_pairs(name):
    data = pairs(); keep = data['split'] == SPLITS[name]
    return {key: value[keep] for key, value in data.items()}


def _initialize(seed):
    rng = np.random.default_rng(seed)
    p = {}
    for i, (ni, no) in enumerate(((16, 12), (12, 24), (24, 4)), 1):
        p[f'W{i}'] = rng.normal(0, 1 / np.sqrt(ni), (ni, no))
        p[f'b{i}'] = np.zeros(no)
    return p, rng.bit_generator.state


def initialize(seed):
    return _initialize(seed)[0]


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=1, keepdims=True)


def downstream(h1, p):
    h2 = np.tanh(h1 @ p['W2'] + p['b2'])
    return softmax(h2 @ p['W3'] + p['b3'])


def forward(x, p):
    h1 = np.tanh(x @ p['W1'] + p['b1'])
    h2 = np.tanh(h1 @ p['W2'] + p['b2'])
    logits = h2 @ p['W3'] + p['b3']
    return softmax(logits), h1, h2, logits


def loss_grad(x, y, p):
    prob, h1, h2, logits = forward(x, p)
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    dz = prob.copy(); dz[np.arange(len(y)), y] -= 1; dz /= len(y)
    g = {'W3': h2.T @ dz, 'b3': dz.sum(axis=0)}
    dh2 = (dz @ p['W3'].T) * (1 - h2 * h2)
    g.update(W2=h1.T @ dh2, b2=dh2.sum(axis=0))
    dh1 = (dh2 @ p['W2'].T) * (1 - h1 * h1)
    g.update(W1=x.T @ dh1, b1=dh1.sum(axis=0))
    return float(loss), g


def reference_downstream(h, p):
    logits, probabilities = [], []
    for row in h:
        hidden = [math.tanh(math.fsum(float(row[i]) * float(p['W2'][i, j]) for i in range(12))
                            + float(p['b2'][j])) for j in range(24)]
        z = [math.fsum(hidden[i] * float(p['W3'][i, j]) for i in range(24)) + float(p['b3'][j]) for j in range(4)]
        ex = [math.exp(value - max(z)) for value in z]
        logits.append(z); probabilities.append([value / math.fsum(ex) for value in ex])
    return np.array(probabilities, dtype=np.float64), np.array(logits, dtype=np.float64)


def reference_hidden(x, p):
    return np.array([[math.tanh(math.fsum(float(row[i]) * float(p['W1'][i, j]) for i in range(16))
                                + float(p['b1'][j])) for j in range(12)] for row in x])


def reference_forward(x, p):
    return reference_downstream(reference_hidden(x, p), p)


def close(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Model shape or finite-value control failed')
    error = float(np.max(np.abs(a - b)))
    if error > TOL: raise RuntimeError('Model scalar replay failed')
    return error


def checked(values):
    if set(values) != set(SHAPES): raise RuntimeError('Model array keys differ from the fixed architecture')
    for key, value in values.items():
        if value.shape != SHAPES[key] or value.dtype != np.float64 or not np.isfinite(value).all():
            raise RuntimeError('A model or optimizer array failed its constraint check')


def snapshot(p, m, v):
    return tuple({key: value.copy() for key, value in group.items()} for group in (p, m, v))


def update(p, m, v, grad, step):
    # Identical update order and arithmetic to Study 1.
    for name in p:
        m[name] = .9 * m[name] + .1 * grad[name]
        v[name] = .999 * v[name] + .001 * grad[name] * grad[name]
        p[name] -= LR * (m[name] / (1 - .9**step)) / (np.sqrt(v[name] / (1 - .999**step)) + 1e-8)


def save_checkpoint(folder, label, state, rng_state, completed, reason=None):
    p, m, v = state
    nsave(folder / f'{label}.npz', **p,
          **{'adam_m_' + key: value for key, value in m.items()},
          **{'adam_v_' + key: value for key, value in v.items()})
    jsave(folder / f'{label}.json', dict(completed_updates=completed, learning_rate=LR,
          betas=[.9, .999], epsilon=1e-8, rng_state_after_initialization=rng_state,
          training_random_draws=False, full_batch_order='All 192 states in the original product order',
          checkpoint_boundary='End of last fully completed parameter update',
          automatic_resume_allowed=False, stop_type=reason))


def train_one(folder, seed, x, labels, guard):
    started = time.monotonic(); folder = output(folder)
    p, rng_state = _initialize(seed)
    m = {key: np.zeros_like(value) for key, value in p.items()}
    v = {key: np.zeros_like(value) for key, value in p.items()}
    checked(p); nsave(folder / 'initial.npz', **p)
    completed = 0; committed = (0, snapshot(p, m, v))
    try:
        with (folder / 'training.csv').open('x', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(('step', 'loss_before_update', 'accuracy_after_update'))
            for step in range(1, STEPS + 1):
                guard()
                loss, grad = loss_grad(x, labels, p)
                if not math.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                checked(grad)
                update(p, m, v, grad, step)
                checked(p); checked(m); checked(v)
                # Keep a complete in-memory boundary. If a signal interrupts a
                # later update, save this state, not half-updated parameters.
                committed = (step, snapshot(p, m, v)); completed = step
                accuracy = float(np.mean(forward(x, p)[0].argmax(1) == labels)) if step == 1 or step % 100 == 0 else ''
                writer.writerow((step, loss, accuracy))
                if step % 100 == 0: stream.flush()
        guard(); nsave(folder / 'final.npz', **p)
        save_checkpoint(folder, 'optimizer-final', committed[1], rng_state, committed[0])
        prob, hidden, _, logits = forward(x, p)
        nsave(folder / 'evaluation.npz', probabilities=prob, logits=logits, hidden=hidden, labels=labels)
        guard()
        with np.load(folder / 'final.npz', allow_pickle=False) as saved:
            weights = {key: saved[key] for key in SHAPES}
            reference_prob, reference_logits = reference_forward(x, weights)
        with np.load(folder / 'evaluation.npz', allow_pickle=False) as saved:
            error = max(close(saved['probabilities'], reference_prob), close(saved['logits'], reference_logits),
                        close(saved['hidden'], reference_hidden(x, weights)))
            accuracy = float(np.mean(saved['probabilities'].argmax(1) == saved['labels']))
        record = dict(seed=seed, status='complete', numerical_checks_passed=True, updates=completed,
                      ordinary_accuracy=accuracy, ability_gate_pass=accuracy >= .99,
                      max_scalar_replay_error=error, tolerance=TOL, seconds=time.monotonic() - started)
        jsave(folder / 'summary.json', record)
        return record
    except BaseException as exc:
        # No permit check in emergency saving. This stores the last complete
        # update and never exposes a private permit path or account details.
        save_checkpoint(folder, 'stopped-checkpoint', committed[1], rng_state, committed[0], type(exc).__name__)
        jsave(folder / 'stopped.json', dict(seed=seed, status='stopped', completed_updates=committed[0],
              stop_type=type(exc).__name__, reason='Training or verification stopped. See the controller record.',
              seconds=time.monotonic() - started, automatic_resume_allowed=False))
        raise


def prepare(out, guard):
    started = time.monotonic(); out = output(out)
    guard(); sources = source_check()
    rng = np.random.default_rng(2026102603)
    x = rng.normal(size=(192, 16)); labels = rng.integers(0, 4, size=192)
    p, _ = _initialize(2026102604)
    m = {key: np.zeros_like(value) for key, value in p.items()}
    v = {key: np.zeros_like(value) for key, value in p.items()}
    checked(p)
    # Check gradients on fixed dummy coordinates before timing.
    _, grad = loss_grad(x, labels, p)
    locations = [(key, index) for key, value in p.items() for index in np.ndindex(value.shape)]
    chosen = rng.choice(len(locations), size=24, replace=False)
    max_gradient_error = 0.
    for location in chosen:
        guard(); key, index = locations[location]; old = p[key][index]
        p[key][index] = old + 1e-5; plus = loss_grad(x, labels, p)[0]
        p[key][index] = old - 1e-5; minus = loss_grad(x, labels, p)[0]
        p[key][index] = old
        estimate = (plus - minus) / 2e-5
        max_gradient_error = max(max_gradient_error, abs(estimate - grad[key][index]) / max(1., abs(estimate), abs(grad[key][index])))
    if max_gradient_error > 1e-6: raise RuntimeError('Dummy gradient check failed')
    step = 0
    def dummy_step(writer):
        nonlocal step
        guard(); step += 1
        loss, grad = loss_grad(x, labels, p)
        if not math.isfinite(loss): raise RuntimeError('Nonfinite dummy loss')
        checked(grad)
        update(p, m, v, grad, step); checked(p); checked(m); checked(v)
        snapshot(p, m, v)
        accuracy = float(np.mean(forward(x, p)[0].argmax(1) == labels)) if step == 1 or step % 100 == 0 else ''
        writer.writerow((step, loss, accuracy))
    blocks = []
    with (out / 'dummy-training.csv').open('x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(('step', 'loss_before_update', 'accuracy_after_update'))
        for _ in range(20): dummy_step(writer)
        for _ in range(3):
            began = time.monotonic()
            for _ in range(100): dummy_step(writer)
            stream.flush(); blocks.append(time.monotonic() - began)
    began = time.monotonic()
    prob, _, _, logits = forward(x, p)
    nsave(out / 'dummy-final.npz', **p)
    nsave(out / 'dummy-evaluation.npz', probabilities=prob, logits=logits)
    with np.load(out / 'dummy-final.npz', allow_pickle=False) as saved:
        reference_prob, reference_logits = reference_forward(x, {key: saved[key] for key in SHAPES})
    with np.load(out / 'dummy-evaluation.npz', allow_pickle=False) as saved:
        replay_error = max(close(saved['probabilities'], reference_prob), close(saved['logits'], reference_logits))
    replay_seconds = time.monotonic() - began
    projected = 4 * (max(blocks) / 100 * STEPS * len(SEEDS) + replay_seconds * len(SEEDS) + 60)
    result = dict(passed=True, scientific_work=False, projected_seconds=projected,
                  warmup_updates=20, timed_blocks=3, updates_per_block=100, block_seconds=blocks,
                  max_dummy_gradient_error=float(max_gradient_error), max_scalar_replay_error=replay_error,
                  measured_save_replay_seconds=replay_seconds, safety_factor=4, setup_allowance_seconds=60,
                  total_scientific_updates=STEPS * len(SEEDS), sources=sources, seconds=time.monotonic() - started)
    guard(); jsave(out / 'summary.json', result)
    return result


def run(out, guard):
    started = time.monotonic(); out = output(out)
    guard(); sources = source_check()
    states, x, labels = inputs(); pair_arrays = pairs()
    with (OLD / 'data/inputs.csv').open() as stream:
        original = list(csv.DictReader(stream))
    if not np.array_equal(states, np.array([[int(row[key]) for key in ('a', 'b', 's', 'n')] for row in original])):
        raise RuntimeError('Task states differ from the original input record')
    nsave(out / 'data.npz', states=states, x=x, labels=labels, **pair_arrays)
    jsave(out / 'data-schema.json', dict(split_codes=SPLITS,
          pair_fields=['r', 'd', 'cf', 'split', 'active', 'same_value'],
          rows_by_split={'learning': 18432, 'selection': 9216, 'test': 9216},
          input_fields=['a', 'b', 's', 'n'], model_seeds=list(SEEDS),
          required_input='Only a takes its donor value; b, s and n stay with the receiver.'))
    records = []
    for seed in SEEDS:
        # A scientific accuracy failure does not skip any later seed.
        records.append(train_one(out / f'seed-{seed}', seed, x, labels, guard))
    guard(); source_check()
    result = dict(passed=True, status='complete', numerical_checks_passed=True,
                  ability_gate_pass=all(row['ability_gate_pass'] for row in records),
                  ability_gate_threshold=.99, seeds=list(SEEDS), models=records,
                  total_updates=STEPS * len(SEEDS), sources=sources,
                  source_commit=SOURCE_COMMIT, seconds=time.monotonic() - started)
    jsave(out / 'summary.json', result)
    return result
