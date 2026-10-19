"""Experiment 1: intact register task ability, without hidden-state edits."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import check_stop

WIDTHS = {'A': 64, 'B': 128}
TOL, UPDATES, BATCH = 1e-9, 6000, 64
GROUPS = [('current_h0', 0, 'current'), ('copy_first_h1', 1, 'copy'),
          ('write_first_h1', 1, 'write')] + [(f'{order}_h{h}', h, order)
          for h in (2, 8, 32) for order in ('copy_write', 'write_copy')]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def check(deadline, *tensors):
    check_stop(deadline)
    if time.monotonic() >= deadline:
        raise RuntimeError('Fixed deadline reached')
    if any(not torch.isfinite(t).all().item() for t in tensors):
        raise RuntimeError('Nonfinite value')


def transition(states, operations):
    states, operations = np.broadcast_arrays(np.asarray(states, dtype=np.int64),
                                             np.asarray(operations, dtype=np.int64))
    if np.any((operations < 0) | (operations > 13)):
        raise ValueError('Invalid operation')
    a, b = states // 4, states % 4
    aa = np.where(operations < 4, operations, a)
    bb = np.where((operations >= 4) & (operations < 8), operations-4, b)
    bb = np.where(operations == 8, a, bb)
    aa = np.where(operations == 9, b, aa)
    aa = np.where(operations == 10, (a+1) % 4, aa)
    bb = np.where(operations == 11, (b+1) % 4, bb)
    aa = np.where(operations == 12, b, aa)
    bb = np.where(operations == 12, a, bb)
    return 4*aa+bb


def labels(tokens):
    state = np.zeros(len(tokens), dtype=np.int64)
    result = []
    for operations in np.asarray(tokens).T:
        state = transition(state, operations)
        result.append(state.copy())
    return np.stack(result, axis=1)


def unique_histories(seed, count, used):
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        row = tuple(rng.integers(0, 14, size=12).tolist())
        if row not in used:
            used.add(row)
            rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def inputs(split):
    used = set()
    train = unique_histories(71201, 1024, used)
    # Generate both pools in a fixed order, so disjointness is checked before use.
    banks = {s: unique_histories(seed, len(GROUPS)*1024, used)
             for s, seed in [('development', 89001), ('final', 89002)]}
    assert len(used) == 1024+2*len(GROUPS)*1024
    histories = banks[split]
    rng = np.random.default_rng(89003 if split == 'development' else 89004)
    all_tokens = np.full((len(histories), 44), -1, dtype=np.int16)
    all_labels = np.full_like(all_tokens, -1)
    lengths = np.zeros(len(histories), dtype=np.int16)
    group_ids = np.repeat(np.arange(len(GROUPS), dtype=np.int16), 1024)
    for group, (_, horizon, order) in enumerate(GROUPS):
        rows = slice(group*1024, (group+1)*1024)
        base = histories[rows]
        a = labels(base)[:, -1]//4
        # The write always changes A. Offsets cycle through 1, 2, and 3.
        write = (a+1+np.arange(1024)%3)%4
        suffix = rng.integers(0, 14, size=(1024, horizon))
        if horizon:
            suffix[:, 0] = 8 if order in ('copy', 'copy_write') else write
        if horizon >= 2:
            suffix[:, 1] = write if order == 'copy_write' else 8
        tokens = np.concatenate([base, suffix], axis=1)
        n = 12+horizon
        all_tokens[rows, :n], all_labels[rows, :n] = tokens, labels(tokens)
        lengths[rows] = n
    return {'train_tokens': train, 'train_labels': labels(train), 'tokens': all_tokens,
            'labels': all_labels, 'lengths': lengths, 'group_ids': group_ids}


class Model(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gru = nn.GRU(14, width, batch_first=True, dtype=torch.float64)
        self.readout = nn.Linear(width, 16, dtype=torch.float64)

    def forward(self, tokens):
        x = F.one_hot(tokens, 14).to(torch.float64)
        return self.readout(self.gru(x)[0])


def save(path, model):
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})


def explicit(tokens, w):
    """GRU equations without nn.GRU or the model forward method."""
    w = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in w.items()}
    x = F.one_hot(torch.as_tensor(tokens, dtype=torch.long), 14).to(torch.float64)
    h = torch.zeros(len(x), w['readout.weight'].shape[1], dtype=torch.float64)
    outputs = []
    for column in x.transpose(0, 1):
        ir, iz, inn = F.linear(column, w['gru.weight_ih_l0'], w['gru.bias_ih_l0']).chunk(3, dim=-1)
        hr, hz, hn = F.linear(h, w['gru.weight_hh_l0'], w['gru.bias_hh_l0']).chunk(3, dim=-1)
        r, z = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
        candidate = torch.tanh(inn+r*hn)
        h = (1-z)*candidate+z*h
        outputs.append(F.linear(h, w['readout.weight'], w['readout.bias']))
    return torch.stack(outputs, dim=1).numpy()


def run_job(recipe: str, seed: int, split: str, out: Path, deadline: float) -> dict:
    if recipe not in WIDTHS or split not in ('development', 'final'):
        raise ValueError('Unknown recipe or split')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    arrays = inputs(split)
    np.savez_compressed(out / 'inputs.npz', **arrays)
    torch.manual_seed(seed)
    model = Model(WIDTHS[recipe])
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters <= 100000
    save(out / 'initial.npz', model)
    x, y = torch.from_numpy(arrays['train_tokens']), torch.from_numpy(arrays['train_labels'])
    schedule_seed = 71202 if split == 'development' else 71208
    schedule = np.random.default_rng(schedule_seed).integers(0, 1024, (UPDATES, BATCH))
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    train_started = time.monotonic()
    try:
        with (out / 'training.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['update', 'loss', 'elapsed_seconds'])
            for step, ids in enumerate(schedule, 1):
                check(deadline, *model.parameters())
                value = model(x[ids])
                loss = F.cross_entropy(value.reshape(-1, 16), y[ids].reshape(-1))
                check(deadline, loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                check(deadline, *(p.grad for p in model.parameters()))
                opt.step()
                check(deadline, *model.parameters())
                writer.writerow([step, loss.item(), time.monotonic()-train_started])
                if step % 250 == 0:
                    stream.flush()
    except BaseException:
        save(out / 'partial.npz', model)
        raise
    train_seconds = time.monotonic()-train_started
    save(out / 'final.npz', model)
    eval_started = time.monotonic()
    predictions = np.full_like(arrays['labels'], -1)
    endpoint_logits = np.zeros((len(predictions), 16), dtype=np.float64)
    metrics, complete_metrics = {}, {}
    with torch.no_grad():
        for group, (name, horizon, _) in enumerate(GROUPS):
            n = 12+horizon
            for first in range(group*1024, (group+1)*1024, 256):
                check(deadline)
                value = model(torch.from_numpy(arrays['tokens'][first:first+256, :n].astype(np.int64)))
                check(deadline, value)
                predictions[first:first+256, :n] = value.argmax(-1).numpy()
                endpoint_logits[first:first+256] = value[:, -1].numpy()
            rows = slice(group*1024, (group+1)*1024)
            correct = predictions[rows, :n] == arrays['labels'][rows, :n]
            metrics[name] = float(correct[:, -1].mean())
            # Supporting result: all checkpoints from the initial state through the suffix.
            complete_metrics[name] = float(correct[:, 11:n].all(axis=1).mean())
    np.savez_compressed(out / 'evaluation.npz', predictions=predictions,
                        endpoint_logits=endpoint_logits, labels=arrays['labels'],
                        group_ids=arrays['group_ids'], lengths=arrays['lengths'])
    result = {'task': 'register', 'recipe': recipe, 'seed': seed, 'split': split,
              'metrics': metrics, 'passed': all(v >= .99 for v in metrics.values()),
              'selection_score': min(metrics.values()), 'eligible': True,
              'seconds': time.monotonic()-started, 'train_seconds': train_seconds,
              'evaluation_seconds': time.monotonic()-eval_started, 'updates': UPDATES,
              'parameters': parameters, 'width': WIDTHS[recipe], 'precision': 'float64',
              'schedule_seed': schedule_seed, 'batch_size': BATCH,
              'rows_per_group': 1024, 'complete_checkpoint_accuracy_supporting': complete_metrics,
              'gate_definition': 'Joint-state accuracy at the final checkpoint of each group.'}
    dump(out / 'metrics.json', result)
    return result


def replay(out: Path, deadline: float = float('inf')) -> dict:
    started = time.monotonic()
    out = Path(out)
    w = dict(np.load(out / 'final.npz'))
    inp, saved = np.load(out / 'inputs.npz'), np.load(out / 'evaluation.npz')
    error, same, labels_same = 0., True, True
    metrics = {}
    for group, (name, horizon, _) in enumerate(GROUPS):
        n = 12+horizon
        endpoint_correct = []
        for first in range(group*1024, (group+1)*1024, 256):
            check(deadline)
            tokens = inp['tokens'][first:first+256, :n]
            values = explicit(tokens.astype(np.int64), w)
            error = max(error, float(np.max(np.abs(values[:, -1]-saved['endpoint_logits'][first:first+256]))))
            pred = values.argmax(-1)
            same &= np.array_equal(pred, saved['predictions'][first:first+256, :n])
            truth = labels(tokens)
            labels_same &= np.array_equal(truth, saved['labels'][first:first+256, :n])
            endpoint_correct.extend((pred[:, -1] == truth[:, -1]).tolist())
        metrics[name] = float(np.mean(endpoint_correct))
    same_metrics = metrics == json.loads((out / 'metrics.json').read_text())['metrics']
    return {'passed': bool(error <= TOL and same and labels_same and same_metrics),
            'max_endpoint_logit_error': error, 'tolerance': TOL,
            'same_predictions': bool(same), 'same_labels': bool(labels_same),
            'same_metrics': same_metrics, 'rows': len(inp['tokens']),
            'seconds': time.monotonic()-started,
            'method': 'Explicit GRU and affine equations from saved weights; no nn.GRU and no retraining.'}


def preflight(out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    check(float("inf"))
    torch.manual_seed(99201)
    rng = np.random.default_rng(99202)
    model = Model(128)
    # Independent random labels: this timing test does not train the register task.
    tokens = torch.from_numpy(rng.integers(0, 14, (64, 12)))
    truth = torch.from_numpy(rng.integers(0, 16, (64, 12)))
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    began = time.monotonic()
    for _ in range(20):
        check(float("inf"))
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tokens).reshape(-1, 16), truth.reshape(-1))
        loss.backward()
        check(float('inf'), loss, *(p.grad for p in model.parameters()))
        opt.step()
        check(float('inf'), *model.parameters())
    train_seconds = time.monotonic()-began
    dummy = rng.integers(0, 14, (256, 44))
    w = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    began = time.monotonic()
    with torch.no_grad():
        native = model(torch.from_numpy(dummy)).numpy()
        noop = model(torch.from_numpy(dummy.copy())).numpy()
    eval_seconds = (time.monotonic()-began)/2
    began = time.monotonic()
    direct = explicit(dummy, w)
    replay_seconds = time.monotonic()-began
    error = float(np.max(np.abs(native-direct)))
    oracle_ok = True
    for a in range(4):
        for b in range(4):
            expected = [(v,b) for v in range(4)] + [(a,v) for v in range(4)] + [
                (a,a), (b,b), ((a+1)%4,b), (a,(b+1)%4), (b,a), (a,b)]
            oracle_ok &= transition(4*a+b, np.arange(14)).tolist() == [4*x+y for x,y in expected]
    parameters = sum(p.numel() for p in model.parameters())
    projected = 4*(train_seconds/20*UPDATES+(eval_seconds+replay_seconds)*4*len(GROUPS)+10)
    passed = bool(parameters <= 100000 and native.shape == (256,44,16)
                  and native.dtype == np.float64 and error <= TOL
                  and np.array_equal(native, noop) and oracle_ok)
    result = {'passed': passed, 'parameters': parameters, 'dummy_steps': 20,
              'dummy_train_seconds': train_seconds, 'dummy_forward_seconds': eval_seconds,
              'dummy_replay_seconds': replay_seconds, 'max_replay_error': error,
              'noop_exact': bool(np.array_equal(native, noop)), 'oracle_224_cases_pass': bool(oracle_ok),
              'projected_job_seconds': projected, 'projected_eight_job_seconds': 8*projected,
              'projection': 'Four times larger-recipe training, 44-step eval/replay for all rows, and 10-second setup allowance.',
              'scientific_training_or_evaluation': False}
    dump(out / 'preflight.json', result)
    return result
